from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json, sha256_json
from miniwebwork.m5_webshop_protocol import task_id_for_goal_index
from miniwebwork.m6_mini import (
    audit_mini_rl,
    build_evaluation_semantics,
    build_mini_chain_report,
    build_mini_sft_gate,
    summarize_closed_loop_identity,
)
from miniwebwork.m6_posttraining_protocol import (
    audit_behavior_sampling_logprobs,
    build_exposure_registry,
    build_split_lock,
    load_protocol,
    validate_split_lock,
)
from miniwebwork.m6_pilot import (
    canonical_sft_audit_input_key,
    load_pilot_waiver,
    validate_artifact_git_compatibility,
    validate_pilot_authorization,
    validate_pilot_method,
    validate_sft_corpus_git_compatibility,
)
from miniwebwork.m6_power import build_power_report
from miniwebwork.webshop_rl import prompt
from miniwebwork.webshop_rl.m6_corpus import (
    audit_conditional_learnability,
    build_policy_visible_corpus,
    build_retention_states,
    flatten_corpus_rows,
    validate_retention_states,
)
from miniwebwork.webshop_rl.m6_online_training import replay_parity_checks, validate_replay_parity
from miniwebwork.webshop_rl.verifier_td import (
    ANCHOR_METHOD,
    BASELINE_METHOD,
    annotate_episode_with_verifier,
    assign_group_credit,
    public_stage_potential,
    score_public_observation,
    trajectory_td_credit,
)


def _registry(tmp_path: Path) -> dict:
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        '{"task_id":"webshop_goal_01003"}\n{"task_id":"webshop_goal_00004"}\n'
        '{"goal_index":5}\n',
        encoding="utf-8",
    )
    return build_exposure_registry(evidence_files=[("sft_corpus", evidence)])


def _goals() -> list[dict]:
    return [
        {
            "goal_index": index,
            "instruction": f"buy unique product {index}",
            "asin": f"B{index:09d}",
        }
        for index in range(12087)
    ]


def test_m6_protocol_split_and_exposure_are_self_hashed_and_isolated(tmp_path: Path):
    protocol = load_protocol()
    assert protocol["payload"]["formal_submission_allowed"] is False
    registry = _registry(tmp_path)
    assert [item["goal_index"] for item in registry["exposures"]] == [4, 5, 1003]
    split = build_split_lock(
        goals=_goals(),
        exposure_registry=registry,
        n_eval=1000,
        protocol=protocol["payload"],
    )
    assert validate_split_lock(split)["roles"]["mini_train"]["count"] == 256
    assert set(split["roles"]["mini_train"]["goal_indices"]) <= set(split["roles"]["train"]["goal_indices"])
    assert 1003 not in split["roles"]["promotion"]["goal_indices"]
    assert 1003 not in split["roles"]["holdout"]["goal_indices"]

    changed = copy.deepcopy(split)
    changed["roles"]["holdout"]["goal_indices"][0] = changed["roles"]["promotion"]["goal_indices"][0]
    with pytest.raises(ValueError, match="hash|overlap"):
        validate_split_lock(changed)


def test_m6_protocol_rejects_training_recipe_drift():
    protocol = load_protocol()["payload"]
    changed = copy.deepcopy(protocol)
    changed["sft"]["lora"]["dropout"] = 0.0
    from miniwebwork.m6_posttraining_protocol import validate_protocol

    with pytest.raises(ValueError, match="LoRA contract drift"):
        validate_protocol(changed)


def test_m6_cpu_artifact_tools_do_not_require_torch_at_import_time():
    # Phase A curriculum construction and post-run audits are CPU-only.  They
    # import this module for schema validation, so importing it must not pull
    # in the GPU learner runtime.
    from miniwebwork.webshop_rl.m6_online_training import (
        update_adaptive_kl_coefficient,
    )

    assert update_adaptive_kl_coefficient(
        0.03,
        0.005,
        target_minimum=0.002,
        target_maximum=0.02,
    ) == pytest.approx(0.03)


def test_m6_immutable_publication_allows_only_identical_retry(tmp_path: Path):
    target = tmp_path / "lock.json"
    first = {"schema_version": "fixture", "value": 1}
    assert publish_immutable_json(target, first) == publish_immutable_json(target, first)
    with pytest.raises(ValueError, match="immutable artifact already differs"):
        publish_immutable_json(target, {"schema_version": "fixture", "value": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == first


def _credit_group() -> list[dict]:
    output = []
    for rollout in range(8):
        success = rollout % 2 == 0
        if success:
            potentials = [0.1, 0.7, 1.0]
            score = 1.0
        else:
            potentials = [0.1, 0.7, 0.0]
            score = 0.65
        output.append(
            {
                "task_id": "webshop_goal_01000",
                "task_score": score,
                "success": success,
                "turns": [
                    {
                        "verifier_public_state_sha256": sha256_json({"rollout": rollout, "turn": turn}),
                        "verifier_progress_after_action": value,
                    }
                    for turn, value in enumerate(potentials)
                ],
            }
        )
    return output


def test_m6_verifier_td_is_strict_telescoping_and_cancels_partial_match():
    assert public_stage_potential(
        page_type="options",
        candidate_attribute_match_fraction=1.0,
        item_attribute_match_fraction=1.0,
        selected_option_fraction=1.0,
    ) == pytest.approx(0.9)
    failed = trajectory_td_credit(task_score=0.65, post_transition_potentials=[0.1, 0.7, 0.0])
    assert sum(failed["dense_rewards"]) == pytest.approx(0.0)
    assert failed["dense_rewards"][-1] == pytest.approx(-0.7)
    assert sum(failed["td_deviations"]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="terminal potential"):
        trajectory_td_credit(task_score=0.65, post_transition_potentials=[0.1, 0.7, 0.65])

    report = assign_group_credit(_credit_group())
    assert report["metrics"]["mixed_strict_reward_signal"] is True
    assert report["metrics"]["maximum_absolute_telescoping_error"] <= 1e-8
    assert report["metrics"]["positive_progress_failure_terminal_cancellation_complete"] is True
    assert all(
        sum(item["td_deviations"]) == pytest.approx(0.0)
        for item in report["trajectory_td"]
    )


def _public_observation(*, page_type: str, visible_text: str, terminal: bool = False) -> dict:
    return {
        "schema_version": "m5-webshop-1.0",
        "task_id": "webshop_goal_01000",
        "instruction": "buy a navy blue mug under $30",
        "page_type": page_type,
        "visible_text": visible_text,
        "text_truncated": False,
        "available_actions": [] if terminal else ["search[<your query>]"],
        "terminal": terminal,
    }


def test_m6_public_progress_uses_post_action_visible_text_and_strict_boundary():
    goal = {
        "instruction": "buy a navy blue mug under $30",
        "name": "hidden exact product title",
        "asin": "B000TARGET",
        "query": "mug",
        "attributes": ["navy blue"],
        "goal_options": ["navy"],
        "price_upper": 30.0,
    }
    search = _public_observation(
        page_type="search_results",
        visible_text=(
            "Search Results\nInstruction: buy a navy blue mug under $30\n"
            "[1] B000000001 | Navy blue travel mug | $24.99 | mugs"
        ),
    )
    item = _public_observation(
        page_type="item",
        visible_text=(
            "Product Page\nInstruction: buy a navy blue mug under $30\n"
            "ASIN: B000000001\nTitle: Navy blue travel mug\nPrice: $24.99\n"
            "Category: mugs\nOptions:\n- color (selected: navy): navy, red"
        ),
    )
    search_evidence = score_public_observation(goal=goal, observation=search)
    item_evidence = score_public_observation(goal=goal, observation=item)
    assert 0.0 < search_evidence["potential"] <= 0.15
    assert item_evidence["potential"] == pytest.approx(0.9)
    serialized = json.dumps(item_evidence)
    assert "hidden exact product title" not in serialized
    assert "B000TARGET" not in serialized

    done = _public_observation(page_type="done", visible_text="Episode complete.", terminal=True)
    episode = {
        "task_id": "webshop_goal_01000",
        "rollout_valid": True,
        "task_score": 0.65,
        "reward": 0.0,
        "success": False,
        "turns": [
            {
                "schema_valid": True,
                "action": {"command": "search[navy blue mug]"},
                "action_result": {"success": True, "error_code": ""},
                "post_action_observation": search,
            },
            {
                "schema_valid": True,
                "action": {"command": "click[B000000001]"},
                "action_result": {"success": True, "error_code": ""},
                "post_action_observation": item,
            },
            {
                "schema_valid": True,
                "action": {"command": "click[Buy Now]"},
                "action_result": {"success": True, "error_code": ""},
                "post_action_observation": done,
            },
        ],
    }
    annotated = annotate_episode_with_verifier(episode, goal=goal)
    potentials = [turn["verifier_progress_after_action"] for turn in annotated["turns"]]
    assert potentials[-1] == 0.0
    td = trajectory_td_credit(task_score=0.65, post_transition_potentials=potentials)
    assert sum(td["dense_rewards"]) == pytest.approx(0.0)
    assert td["dense_rewards"][-1] == pytest.approx(-0.9)


def test_m6_invalid_output_has_zero_transition_credit_until_strict_boundary():
    goal = {
        "instruction": "buy a navy blue mug under $30",
        "attributes": ["navy blue"],
        "price_upper": 30.0,
    }
    observation = _public_observation(page_type="home", visible_text="Welcome")
    episode = {
        "task_id": "webshop_goal_01000",
        "rollout_valid": True,
        "task_score": 0.0,
        "success": False,
        "turns": [
            {"schema_valid": False, "action": None, "post_action_observation": observation},
            {"schema_valid": False, "action": None, "post_action_observation": observation},
        ],
    }
    annotated = annotate_episode_with_verifier(episode, goal=goal)
    assert [turn["verifier_progress_after_action"] for turn in annotated["turns"]] == [0.0, 0.0]
    assert annotated["turns"][0]["verifier_progress_evidence"]["boundary"] == "no_environment_transition"


def test_m6_missing_action_result_cannot_claim_a_transition():
    goal = {
        "instruction": "buy a navy blue mug under $30",
        "attributes": ["navy blue"],
        "price_upper": 30.0,
    }
    item = _public_observation(
        page_type="item",
        visible_text="Product Page navy blue mug $24.99",
    )
    episode = {
        "task_id": "webshop_goal_01000",
        "rollout_valid": True,
        "task_score": 0.0,
        "success": False,
        "turns": [
            {
                "schema_valid": True,
                "action": {"command": "click[B000000001]"},
                "post_action_observation": item,
            }
        ],
    }
    annotated = annotate_episode_with_verifier(episode, goal=goal)
    turn = annotated["turns"][0]
    assert turn["verifier_progress_after_action"] == 0.0
    # A one-turn episode is simultaneously a no-op model failure and the
    # strict terminal boundary.  The zero potential proves that the visible
    # item page did not receive invented transition credit.
    assert turn["verifier_progress_evidence"]["boundary"] == "strict_episode_boundary"


def _source_trajectory(index: int, *, query: str = "blue mug") -> dict:
    instruction = "buy a blue mug"
    first_observation = {
        "schema_version": "m5-webshop-1.0",
        "task_id": "webshop_goal_01000",
        "episode_id": f"episode-{index}",
        "instruction": instruction,
        "step_index": 0,
        "page_type": "home",
        "visible_text": "Welcome to WebShop blue mug",
        "text_truncated": False,
        "available_actions": ["search[<your query>]"],
        "terminal": False,
    }
    second_observation = {
        **first_observation,
        "step_index": 1,
        "page_type": "search_results",
        "visible_text": "Blue mug B000000001",
        "available_actions": ["click[B000000001]"],
    }
    first_messages = prompt.build_messages(type("Observation", (), first_observation)(), [])
    second_messages = prompt.build_messages(
        type("Observation", (), second_observation)(),
        [{"turn": 1, "command": f"search[{query}]", "success": True, "error_code": "", "page_type": "search_results"}],
    )
    return {
        "trajectory_id": f"trajectory-{index}",
        "task_id": "webshop_goal_01000",
        "task_score": 1.0,
        "success": True,
        "replay_success": True,
        "turns": [
            {
                "observation": first_observation,
                "rendered_prompt_sha256": prompt.compute_message_hash(first_messages),
                "schema_valid": True,
                "action": {"command": f"search[{query}]"},
                "raw_output": json.dumps({"command": f"search[{query}]"}),
                "action_result": {"success": True, "error_code": ""},
                "terminated": False,
            },
            {
                "observation": second_observation,
                "rendered_prompt_sha256": prompt.compute_message_hash(second_messages),
                "schema_valid": True,
                "action": {"command": "click[B000000001]"},
                "raw_output": '{"command":"click[B000000001]"}',
                "action_result": {"success": True, "error_code": ""},
                "terminated": True,
            },
        ],
    }


def test_m6_policy_visible_corpus_rejects_target_asin_leak():
    goal = {
        "goal_index": 1000,
        "instruction": "buy a blue mug",
        "asin": "B000000001",
    }
    corpus = build_policy_visible_corpus(
        trajectories=[_source_trajectory(index) for index in range(4)],
        goal_by_task_id={"webshop_goal_01000": goal},
        completion_token_counter=lambda text: len(text),
    )
    assert corpus["hidden_metadata_used_for_label_construction"] is False
    audit = audit_conditional_learnability(
        corpus=corpus,
        goal_by_task_id={"webshop_goal_01000": goal},
        mini=True,
    )
    # The small fixture intentionally fails scale, diversity and recovery, but
    # policy-visible label checks pass.
    assert audit["checks"]["hidden_policy_fields"] is True
    assert audit["checks"]["target_asin_search_labels"] is True
    assert audit["checks"]["public_query_token_fraction"] is True
    assert audit["metrics"]["strict_success_fraction"] == 1.0
    assert audit["metrics"]["replay_success_fraction"] == 1.0

    leaked = build_policy_visible_corpus(
        trajectories=[_source_trajectory(10, query="B000000001")],
        goal_by_task_id={"webshop_goal_01000": goal},
        completion_token_counter=lambda text: len(text),
    )
    leaked_audit = audit_conditional_learnability(
        corpus=leaked,
        goal_by_task_id={"webshop_goal_01000": goal},
        mini=True,
    )
    assert leaked_audit["checks"]["target_asin_search_labels"] is False
    assert leaked_audit["passed"] is False


def test_m6_retention_states_have_raw_action_positions_but_no_labels():
    source = _source_trajectory(20)
    source.update(
        rollout_valid=True,
        trajectory_id="raw-retention-20",
    )
    for index, turn in enumerate(source["turns"]):
        turn["prompt_token_ids"] = [10, 11, 12 + index]
        turn["generated_token_ids"] = [20 + index, 30 + index]
    retention = build_retention_states(trajectories=[source], maximum_states=10)
    validated = validate_retention_states(retention)
    assert validated["state_count"] == 2
    assert validated["supervised_label_count"] == 0
    assert all(state["supervised_label_present"] is False for state in validated["states"])


def test_m6_flattened_corpus_preserves_trajectory_provenance():
    goal = {"goal_index": 1000, "instruction": "buy a blue mug", "asin": "B000000001"}
    corpus = build_policy_visible_corpus(
        trajectories=[_source_trajectory(30)],
        goal_by_task_id={"webshop_goal_01000": goal},
        completion_token_counter=lambda text: len(text),
    )
    rows = flatten_corpus_rows(corpus)
    assert len(rows) == 2
    assert {row["trajectory_id"] for row in rows} == {"trajectory-30"}
    assert all("trajectory_command_sequence_sha256" in row for row in rows)


def test_m6_sft_update_schedule_is_exact_nine_to_one():
    pytest.importorskip("torch")
    from miniwebwork.webshop_rl.m6_sft_training import build_update_schedule

    schedule = build_update_schedule(20, 3, seed=7)
    assert len(schedule) == 2
    assert all(len(item["imitation_indices"]) == 9 for item in schedule)
    used = [index for item in schedule for index in item["imitation_indices"]]
    assert len(used) == len(set(used)) == 18


def _eval_groups(identity: str, success_tasks: int) -> list[dict]:
    groups = []
    for task_index in range(200):
        success = task_index < success_tasks
        groups.append(
            {
                "task_id": task_id_for_goal_index(2000 + task_index),
                "trajectories": [
                    {
                        "rollout_seed": rollout,
                        "success": success,
                        "task_score": 1.0 if success else 0.0,
                        "termination_reason": "purchase" if success else "max_model_turns",
                        "generated_action_tokens": 10,
                        "environment_steps": 2,
                        "turns": [
                            {
                                "schema_valid": True,
                                "action_result": {"error_code": ""},
                                "observation": {"page_type": "item"},
                            }
                        ],
                    }
                    for rollout in range(4)
                ],
            }
        )
    return groups


def test_m6_mini_gate_requires_paired_raw_sft_rl_chain():
    evaluation_contract = "e" * 64
    raw = summarize_closed_loop_identity(identity="raw", groups=_eval_groups("raw", 60), development_only=False, evaluation_contract_sha256=evaluation_contract)
    sft = summarize_closed_loop_identity(identity="mini_sft", groups=_eval_groups("mini_sft", 70), development_only=True, evaluation_contract_sha256=evaluation_contract)
    rl = summarize_closed_loop_identity(identity="mini_rl", groups=_eval_groups("mini_rl", 80), development_only=True, evaluation_contract_sha256=evaluation_contract)
    corpus_audit = {"schema_version": "m6_conditional_learnability_audit_v1", "passed": True, "checks": {"ok": True}}
    corpus_audit["content_sha256"] = sha256_json(corpus_audit)
    rl_audit = {"schema_version": "m6_mini_rl_audit_v1", "passed": True, "checks": {"ok": True}}
    rl_audit["content_sha256"] = sha256_json(rl_audit)
    report = build_mini_chain_report(
        raw=raw,
        sft=sft,
        rl=rl,
        corpus_audit=corpus_audit,
        rl_audit=rl_audit,
    )
    assert report["passed"] is True
    assert report["decision"] == "READY_FOR_FULL_SFT_APPROVAL"
    assert report["mini_checkpoints_reusable_for_formal_training"] is False


def test_m6_sft_stage_gate_stops_rl_until_sft_beats_raw():
    evaluation_contract = "e" * 64
    raw = summarize_closed_loop_identity(identity="raw", groups=_eval_groups("raw", 60), development_only=False, evaluation_contract_sha256=evaluation_contract)
    sft = summarize_closed_loop_identity(identity="mini_sft", groups=_eval_groups("mini_sft", 70), development_only=True, evaluation_contract_sha256=evaluation_contract)
    corpus = {"schema_version": "m6_conditional_learnability_audit_v1", "passed": True, "checks": {"ok": True}}
    corpus["content_sha256"] = sha256_json(corpus)
    passed = build_mini_sft_gate(raw=raw, sft=sft, corpus_audit=corpus)
    assert passed["passed"] is True
    assert passed["decision"] == "ALLOW_MINI_RL"
    regressed = summarize_closed_loop_identity(identity="mini_sft", groups=_eval_groups("mini_sft", 50), development_only=True, evaluation_contract_sha256=evaluation_contract)
    stopped = build_mini_sft_gate(raw=raw, sft=regressed, corpus_audit=corpus)
    assert stopped["passed"] is False
    assert stopped["decision"] == "STOP_AND_BURN_MINI_DEV"


def test_m6_paired_gate_rejects_evaluation_contract_drift():
    raw = summarize_closed_loop_identity(
        identity="raw",
        groups=_eval_groups("raw", 60),
        development_only=False,
        evaluation_contract_sha256="a" * 64,
    )
    sft = summarize_closed_loop_identity(
        identity="mini_sft",
        groups=_eval_groups("mini_sft", 70),
        development_only=True,
        evaluation_contract_sha256="b" * 64,
    )
    corpus = {"schema_version": "m6_conditional_learnability_audit_v1", "passed": True, "checks": {"ok": True}}
    corpus["content_sha256"] = sha256_json(corpus)
    with pytest.raises(ValueError, match="evaluation contract drift"):
        build_mini_sft_gate(raw=raw, sft=sft, corpus_audit=corpus)


def test_m6_evaluation_semantics_separate_policy_and_git_lineage():
    protocol = load_protocol()["payload"]
    split_hash = "c" * 64

    def evidence(git_sha: str, adapter: str | None) -> tuple[dict, dict]:
        invocation = {
            "mode": "evaluation",
            "role": "mini_dev",
            "K": 4,
            "task_order_sha256": "d" * 64,
            "seed": 20260812,
            "iteration_index": 0,
            "max_model_turns": 18,
            "max_environment_steps": 15,
            "protocol_sha256": "e" * 64,
            "git_sha": git_sha,
            "split_lock_content_sha256": split_hash,
            "base_model": protocol["sft"]["base_model"],
            "adapter": adapter,
        }
        collection = {
            key: invocation[key]
            for key in ("mode", "role", "K", "task_order_sha256", "protocol_sha256", "git_sha", "split_lock_content_sha256")
        }
        return collection, invocation

    raw_collection, raw_invocation = evidence("a" * 40, None)
    sft_collection, sft_invocation = evidence("b" * 40, "/tmp/sft-adapter")
    raw = build_evaluation_semantics(
        collection=raw_collection,
        invocation=raw_invocation,
        split_lock_content_sha256=split_hash,
        protocol=protocol,
    )
    sft = build_evaluation_semantics(
        collection=sft_collection,
        invocation=sft_invocation,
        split_lock_content_sha256=split_hash,
        protocol=protocol,
    )
    assert raw == sft
    changed = copy.deepcopy(sft_invocation)
    changed["seed"] += 1
    assert build_evaluation_semantics(
        collection=sft_collection,
        invocation=changed,
        split_lock_content_sha256=split_hash,
        protocol=protocol,
    )["content_sha256"] != raw["content_sha256"]


def test_m6_rl_audit_requires_real_updates_and_credit():
    credit = assign_group_credit(_credit_group())
    learner = {
        "development_only": True,
        "method": "strict_grpo_verifier_td",
        "group_size": 8,
        "generated_action_tokens": 40_000,
        "optimizer_updates": 2,
        "parameters_changed": True,
        "lineage_chain_complete": True,
        "frozen_reference_sft_adapter_sha256": "c" * 64,
        "parameter_sha256_before": "a" * 64,
        "parameter_sha256_after": "b" * 64,
        "iterations": [
            {
                "loss": 0.1,
                "gradient_norm": 1.0,
                "observed_kl": 0.01,
                "adaptive_kl_coefficient_before": 0.03,
                "adaptive_kl_coefficient_after": 0.03,
                "mixed_strict_reward_signal": True,
            }
            for _ in range(5)
        ],
    }
    learner["content_sha256"] = sha256_json(learner)
    report = audit_mini_rl(learner_report=learner, credit_assignments=[credit] * 5)
    assert report["passed"] is True
    changed = copy.deepcopy(learner)
    changed["parameter_sha256_after"] = changed["parameter_sha256_before"]
    changed["content_sha256"] = sha256_json({key: value for key, value in changed.items() if key != "content_sha256"})
    assert audit_mini_rl(learner_report=changed, credit_assignments=[credit] * 5)["passed"] is False


def test_m6_pilot_waiver_is_exact_and_development_only():
    waiver_bundle = load_pilot_waiver()
    waiver = waiver_bundle["payload"]
    assert waiver["formal_training_allowed"] is False
    assert waiver["only_waived_check"] == "mini_success_task_count"
    assert tuple(waiver["approved_rl_methods"]) == (BASELINE_METHOD, ANCHOR_METHOD)
    authorization = {
        "schema_version": "m6_mini_pilot_authorization_v1",
        "study_id": waiver["study_id"],
        "development_only": True,
        "formal_training_allowed": False,
        "passed": True,
        "decision": waiver["decision"],
        "waiver_file_sha256": waiver_bundle["sha256"],
        "original_minimum_success_tasks": 160,
        "authorized_minimum_success_tasks": 156,
        "observed_success_tasks": 156,
        "observed_replay_success_trajectories": 493,
        "observed_completion_label_tokens": 31362,
        "only_waived_check": waiver["only_waived_check"],
        "failed_corpus_audit_content_sha256": waiver["failed_corpus_audit_content_sha256"],
        "source_protocol_sha256": waiver["source_protocol_sha256"],
        "source_producer_git_sha": waiver["source_producer_git_sha"],
        "source_collection_report_content_sha256": waiver["source_collection_report_content_sha256"],
        "source_collection_seeds": waiver["source_collection_seeds"],
        "approved_rl_methods": waiver["approved_rl_methods"],
        "shared_rl_controls": waiver["shared_rl_controls"],
    }
    authorization["content_sha256"] = sha256_json(authorization)
    assert validate_pilot_authorization(authorization)["passed"] is True
    assert validate_pilot_method(BASELINE_METHOD, authorization)["verifier_td_lambda"] == 0.0
    changed = copy.deepcopy(authorization)
    changed["observed_success_tasks"] = 155
    changed["content_sha256"] = sha256_json(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="task count"):
        validate_pilot_authorization(changed)


def test_m6_sft_runtime_rebuild_uses_corpus_audit_input_keys():
    assert canonical_sft_audit_input_key("train.jsonl") == "train"
    assert canonical_sft_audit_input_key("dev.jsonl") == "dev"
    assert canonical_sft_audit_input_key("pilot_authorization.json") == "pilot_authorization"
    with pytest.raises(ValueError, match="filename drift"):
        canonical_sft_audit_input_key("trainl")


def test_m6_sft_corpus_git_bridge_is_explicit_and_fail_closed():
    producer = "a" * 40
    consumer = "b" * 40
    assert validate_sft_corpus_git_compatibility(
        corpus_producer_git_sha=producer,
        consumer_git_sha=consumer,
        explicitly_authorized_producer_git_sha=producer,
    ) == producer
    with pytest.raises(ValueError, match="explicitly authorized"):
        validate_sft_corpus_git_compatibility(
            corpus_producer_git_sha=producer,
            consumer_git_sha=consumer,
            explicitly_authorized_producer_git_sha=None,
        )
    with pytest.raises(ValueError, match="explicitly authorized"):
        validate_sft_corpus_git_compatibility(
            corpus_producer_git_sha=producer,
            consumer_git_sha=consumer,
            explicitly_authorized_producer_git_sha="c" * 40,
        )


def test_m6_derived_artifact_git_bridge_is_explicit_and_scoped():
    producer = "a" * 40
    consumer = "b" * 40
    assert validate_artifact_git_compatibility(
        artifact_name="RL curriculum",
        producer_git_sha=producer,
        consumer_git_sha=consumer,
        explicitly_authorized_producer_git_sha=producer,
    ) == producer
    with pytest.raises(ValueError, match="RL curriculum.*explicitly authorized"):
        validate_artifact_git_compatibility(
            artifact_name="RL curriculum",
            producer_git_sha=producer,
            consumer_git_sha=consumer,
            explicitly_authorized_producer_git_sha=None,
        )


def test_m6_dual_methods_share_macro_credit_but_differ_within_trajectory():
    group = _credit_group()
    baseline = assign_group_credit(group, method=BASELINE_METHOD)
    anchor = assign_group_credit(group, method=ANCHOR_METHOD)
    assert baseline["macro_advantages"] == anchor["macro_advantages"]
    assert baseline["verifier_td_lambda"] == 0.0
    assert anchor["verifier_td_lambda"] == 0.5
    for macro, turns in zip(baseline["macro_advantages"], baseline["turn_credit"]):
        assert all(turn["turn_advantage"] == pytest.approx(macro) for turn in turns)
    assert any(
        baseline_turn["turn_advantage"] != pytest.approx(anchor_turn["turn_advantage"])
        for baseline_trajectory, anchor_trajectory in zip(
            baseline["turn_credit"], anchor["turn_credit"]
        )
        for baseline_turn, anchor_turn in zip(baseline_trajectory, anchor_trajectory)
    )


def test_m6_power_report_fails_closed_without_empirical_backend(monkeypatch):
    differences = [float((index % 3) - 1) for index in range(100)]
    report = build_power_report(
        comparisons={"raw_sft": differences, "sft_rl": differences, "raw_rl": differences},
        simulations=20_000,
    )
    assert report["selected_n_eval"] in {1000, 1500, 2000, None}
    assert report["decision"] in {"FREEZE_N_EVAL", "STOP_BEFORE_TRAINING"}


def test_m6_behavior_sampling_logprob_parity_is_fail_closed():
    report = audit_behavior_sampling_logprobs(
        [-0.5, -0.7],
        [-0.5, -0.7],
        maximum_absolute_difference=1e-6,
    )
    assert report["passed"] is True
    with pytest.raises(ValueError, match="behavior/sampling parity"):
        audit_behavior_sampling_logprobs(
            [-0.5, -0.7],
            [-0.5, -0.69],
            maximum_absolute_difference=1e-6,
        )


def test_m6_replay_parity_failure_preserves_numeric_evidence():
    contract = load_protocol()["payload"]["rl"]["parity_contract"]
    parity = {
        "mean_absolute_logprob_difference": 0.01,
        "p95_absolute_logprob_difference": 0.04,
        "p99_absolute_logprob_difference": 0.101,
        "p999_absolute_logprob_difference": 0.2,
        "initial_ratio_clip_fraction": 0.0,
        "mean_importance_ratio": 1.0,
    }
    checks = replay_parity_checks(parity, contract=contract)
    assert checks["p99_absolute_difference"] is False
    assert sum(not value for value in checks.values()) == 1
    with pytest.raises(ValueError) as captured:
        validate_replay_parity(parity, contract=contract)
    failure = json.loads(str(captured.value).split(": ", 1)[1])
    assert failure["parity"]["p99_absolute_logprob_difference"] == pytest.approx(0.101)
    assert failure["thresholds"]["replay_p99_absolute_difference"] == pytest.approx(0.1)
    assert failure["checks"] == checks
