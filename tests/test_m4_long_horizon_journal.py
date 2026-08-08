from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from miniwebwork.long_horizon_rl.contracts import (
    TRAJECTORY_SCHEMA,
    TURN_SCHEMA,
    RunIdentity,
    public_anchor_signature,
    token_ids_sha256,
)
from miniwebwork.long_horizon_rl.credit import CREDIT_FORMULA_VERSION
from miniwebwork.long_horizon_rl.journal import (
    AppendOnlyAttemptJournal,
    CollectionStore,
    build_committed_group,
    read_journal,
)


def _identity(adapter="6" * 64):
    return RunIdentity(
        study_id="m4_long_horizon_credit_v1",
        git_sha="1" * 64,
        method="multi_turn_grpo",
        seed=20260801,
        dataset_manifest_sha256="2" * 64,
        seed_manifest_sha256="3" * 64,
        prompt_contract="browser_agent_v4_long_memory",
        prompt_sha256="4" * 64,
        credit_formula_version=CREDIT_FORMULA_VERSION,
        task_order_sha256="5" * 64,
        input_adapter_sha256=adapter,
    )


def _turn(identity, rollout_index):
    observation = {
        "schema_version": "1.0",
        "task_id": "TASK-1",
        "instruction": "instruction",
        "path": "/products",
        "page_type": "products",
        "title": "Products",
        "visible_text": "Products",
        "elements": [],
        "last_action_result": None,
        "terminal": False,
    }
    prompt_ids = [1, 2]
    generated_ids = [10 + rollout_index, 20]
    return {
        "schema_version": TURN_SCHEMA,
        "study_id": identity.study_id,
        "method": identity.method,
        "seed": identity.seed,
        "iteration_index": 0,
        "policy_version": "policy_0000",
        "group_id": "group-0000",
        "trajectory_id": f"trajectory-{rollout_index}",
        "task_id": "TASK-1",
        "rollout_index": rollout_index,
        "turn_index": 1,
        "adapter_sha256": identity.input_adapter_sha256,
        "rendered_prompt_sha256": "7" * 64,
        "prompt_token_ids": prompt_ids,
        "prompt_token_sha256": token_ids_sha256(prompt_ids),
        "generated_token_ids": generated_ids,
        "behavior_logprobs": [-0.1, -0.2],
        "sampling_logprobs": [-0.1, -0.2],
        "observation": observation,
        "anchor_signature": public_anchor_signature(observation, prompt_token_ids=prompt_ids),
    }


def _trajectory(identity, rollout_index, reward):
    turn = _turn(identity, rollout_index)
    return {
        "schema_version": TRAJECTORY_SCHEMA,
        "study_id": identity.study_id,
        "method": identity.method,
        "seed": identity.seed,
        "iteration_index": 0,
        "trajectory_id": f"trajectory-{rollout_index}",
        "group_id": "group-0000",
        "task_id": "TASK-1",
        "policy_version": "policy_0000",
        "adapter_sha256": identity.input_adapter_sha256,
        "rollout_index": rollout_index,
        "rollout_valid": True,
        "success": reward == 1.0,
        "reward": reward,
        "failure_origin": "none" if reward == 1.0 else "policy",
        "turns": [turn],
        "generated_action_tokens": 2,
    }


def test_partial_group_cost_survives_restart_and_entire_group_is_resampled(tmp_path):
    identity = _identity()
    store = CollectionStore(tmp_path / "collection", identity)
    attempt = store.start_group(group_id="group-0000", task_id="TASK-1", iteration_index=0)
    assert attempt == 0
    store.journal.append_turn_generated(
        group_id="group-0000",
        attempt_index=attempt,
        trajectory_id="partial-trajectory",
        rollout_index=0,
        turn_index=1,
        generated_token_ids=[1, 2, 3],
        turn_sha256="8" * 64,
    )
    store.mark_group_invalid(group_id="group-0000", attempt_index=attempt, reason="worker crashed")

    resumed = CollectionStore(tmp_path / "collection", identity)
    assert resumed.journal.generated_action_tokens == 3
    assert resumed.start_group(
        group_id="group-0000", task_id="TASK-1", iteration_index=0
    ) == 1
    trajectories = [_trajectory(identity, index, float(index % 2 == 0)) for index in range(4)]
    for trajectory in trajectories:
        turn = trajectory["turns"][0]
        resumed.journal.append_turn_generated(
            group_id="group-0000",
            attempt_index=1,
            trajectory_id=trajectory["trajectory_id"],
            rollout_index=trajectory["rollout_index"],
            turn_index=1,
            generated_token_ids=turn["generated_token_ids"],
            turn_sha256="9" * 64,
        )
        resumed.mark_trajectory_completed(
            group_id="group-0000", attempt_index=1, trajectory=trajectory
        )
    group = build_committed_group(
        identity=identity,
        iteration_index=0,
        group_id="group-0000",
        task_id="TASK-1",
        policy_version="policy_0000",
        trajectories=trajectories,
    )
    resumed.commit_group(group, attempt_index=1)
    assert resumed.journal.generated_action_tokens == 11
    assert group["generated_action_tokens"] == 8
    assert resumed.journal.committed_group_ids == ("group-0000",)
    assert resumed.load_committed_groups()[0]["group_sha256"] == group["group_sha256"]
    manifest = resumed.freeze_collection(
        iteration_index=0,
        stopped_for_token_budget=False,
        task_sampler_state={"sampler_version": "test"},
    )
    assert manifest["committed_group_action_tokens"] == 8
    assert manifest["all_generated_action_tokens"] == 11
    assert resumed.freeze_collection(
        iteration_index=0,
        stopped_for_token_budget=False,
        task_sampler_state={"sampler_version": "test"},
    ) == manifest


def test_group_commit_requires_durable_exact_k_trajectory_events(tmp_path):
    identity = _identity()
    store = CollectionStore(tmp_path / "collection", identity)
    attempt = store.start_group(group_id="group-0000", task_id="TASK-1", iteration_index=0)
    trajectories = [_trajectory(identity, index, float(index % 2 == 0)) for index in range(4)]
    group = build_committed_group(
        identity=identity,
        iteration_index=0,
        group_id="group-0000",
        task_id="TASK-1",
        policy_version="policy_0000",
        trajectories=trajectories,
    )
    with pytest.raises(ValueError, match="exactly K"):
        store.commit_group(group, attempt_index=attempt)


def test_group_artifact_identifier_rejects_path_traversal(tmp_path):
    store = CollectionStore(tmp_path / "collection", _identity())
    with pytest.raises(ValueError, match="safe artifact"):
        store.start_group(group_id="../escape", task_id="TASK-1", iteration_index=0)


def test_journal_identity_drift_fails_closed(tmp_path):
    path = tmp_path / "attempts.jsonl"
    AppendOnlyAttemptJournal(path, _identity())
    with pytest.raises(ValueError, match="identity mismatch"):
        AppendOnlyAttemptJournal(path, _identity(adapter="a" * 64))


def test_journal_hash_chain_detects_corruption(tmp_path):
    path = tmp_path / "attempts.jsonl"
    journal = AppendOnlyAttemptJournal(path, _identity())
    journal.append("group_started", {"group_id": "g", "attempt_index": 0})
    rows = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(rows[-1])
    event["payload"]["attempt_index"] = 99
    rows[-1] = json.dumps(event, sort_keys=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_journal(path)


def test_journal_serializes_concurrent_turn_events_without_losing_tokens(tmp_path):
    journal = AppendOnlyAttemptJournal(tmp_path / "attempts.jsonl", _identity())

    def append(index):
        journal.append_turn_generated(
            group_id="g",
            attempt_index=0,
            trajectory_id=f"t-{index // 10}",
            rollout_index=index // 10,
            turn_index=index % 10 + 1,
            generated_token_ids=[index + 1],
            turn_sha256="b" * 64,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append, range(40)))
    events = read_journal(journal.path)
    assert [event["sequence"] for event in events] == list(range(41))
    assert journal.generated_action_tokens == 40
