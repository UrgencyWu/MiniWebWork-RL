from __future__ import annotations

import copy

import pytest

from miniwebwork.long_horizon_rl.contracts import (
    TRAJECTORY_SCHEMA,
    TURN_SCHEMA,
    RunIdentity,
    public_anchor_signature,
    token_ids_sha256,
)
from miniwebwork.long_horizon_rl.credit import (
    BASELINE_METHOD,
    CREDIT_FORMULA_VERSION,
    STEP_AWARE_METHOD,
    assign_group_credit,
    standardized_advantages,
)
from miniwebwork.long_horizon_rl.journal import build_committed_group


def _identity(method: str = STEP_AWARE_METHOD) -> RunIdentity:
    return RunIdentity(
        study_id="m4_long_horizon_credit_v1",
        git_sha="1" * 64,
        method=method,
        seed=20260801,
        iteration_index=0,
        policy_version="policy_0000",
        dataset_manifest_sha256="2" * 64,
        seed_manifest_sha256="3" * 64,
        prompt_contract="browser_agent_v4_long_memory",
        prompt_sha256="4" * 64,
        credit_formula_version=CREDIT_FORMULA_VERSION,
        task_order_sha256="5" * 64,
        base_model_manifest_sha256="8" * 64,
        runtime_contract_sha256="9" * 64,
        input_adapter_sha256="6" * 64,
        input_rollout_adapter_sha256="a" * 64,
        input_adapter_semantic_sha256="b" * 64,
    )


def _observation(path: str = "/products") -> dict:
    return {
        "schema_version": "1.0",
        "task_id": "TASK-1",
        "episode_id": "EP-RUNTIME-SECRET",
        "instruction": "Choose the cheapest feasible product",
        "step_index": 0,
        "url": f"http://127.0.0.1:9999{path}?episode_id=SECRET",
        "path": path,
        "page_type": "products",
        "title": "Products",
        "visible_text": "Product P1 price 10",
        "text_truncated": False,
        "elements": [{
            "element_id": "runtime-generated-id",
            "role": "link",
            "tag": "a",
            "name": "P1",
            "text": "P1",
            "value": "",
            "input_type": "",
            "testid": "product-P1",
            "options": [],
            "disabled": False,
        }],
        "last_action_result": None,
        "terminal": False,
        "oracle_answer": "P1",
        "verifier": {"success": True},
    }


def _turn(identity: RunIdentity, rollout_index: int, turn_index: int, *, shared: bool) -> dict:
    observation = _observation("/products" if shared else f"/products/P{rollout_index}")
    prompt_ids = [11, 12] if shared else [11, 12, 100 + rollout_index]
    generated_ids = [20 + rollout_index, 30 + turn_index]
    return {
        "schema_version": TURN_SCHEMA,
        "study_id": identity.study_id,
        "method": identity.method,
        "seed": identity.seed,
        "iteration_index": 0,
        "attempt_index": 0,
        "policy_version": "policy_0000",
        "group_id": "group-0000",
        "trajectory_id": f"trajectory-{rollout_index}",
        "task_id": "TASK-1",
        "rollout_index": rollout_index,
        "turn_index": turn_index,
        "request_id": f"request-{rollout_index}-{turn_index}",
        "sampling_seed": 2026080100 + rollout_index * 20 + turn_index,
        "generation_backend": "vllm_async",
        "adapter_sha256": identity.input_adapter_sha256,
        "rollout_adapter_sha256": identity.input_rollout_adapter_sha256,
        "adapter_semantic_sha256": identity.input_adapter_semantic_sha256,
        "rendered_prompt_sha256": "7" * 64,
        "prompt_token_ids": prompt_ids,
        "prompt_token_sha256": token_ids_sha256(prompt_ids),
        "generated_token_ids": generated_ids,
        "generated_token_sha256": token_ids_sha256(generated_ids),
        "behavior_logprobs": [-0.2, -0.3],
        "sampling_logprobs": [-0.2, -0.3],
        "observation": observation,
        "anchor_signature": public_anchor_signature(observation, prompt_token_ids=prompt_ids),
        "schema_valid": True,
        "parsed_action": {"action": "click", "target": "product-P1"},
    }


def _group(method: str, rewards=(1.0, 0.0, 1.0, 0.0)) -> dict:
    identity = _identity(method)
    trajectories = []
    for rollout_index, reward in enumerate(rewards):
        turns = [
            _turn(identity, rollout_index, 1, shared=True),
            _turn(identity, rollout_index, 2, shared=False),
        ]
        trajectories.append({
            "schema_version": TRAJECTORY_SCHEMA,
            "study_id": identity.study_id,
            "method": identity.method,
            "seed": identity.seed,
            "iteration_index": 0,
            "attempt_index": 0,
            "trajectory_id": f"trajectory-{rollout_index}",
            "group_id": "group-0000",
            "task_id": "TASK-1",
            "policy_version": "policy_0000",
            "adapter_sha256": identity.input_adapter_sha256,
            "rollout_adapter_sha256": identity.input_rollout_adapter_sha256,
            "adapter_semantic_sha256": identity.input_adapter_semantic_sha256,
            "rollout_index": rollout_index,
            "rollout_valid": True,
            "success": reward == 1.0,
            "reward": reward,
            "failure_origin": "none" if reward == 1.0 else "policy",
            "turns": turns,
            "generated_action_tokens": sum(len(turn["generated_token_ids"]) for turn in turns),
        })
    return build_committed_group(
        identity=identity,
        iteration_index=0,
        attempt_index=0,
        group_id="group-0000",
        task_id="TASK-1",
        policy_version="policy_0000",
        trajectories=trajectories,
    )


def test_public_anchor_ignores_runtime_or_hidden_fields_but_changes_with_public_state():
    left = _observation()
    right = copy.deepcopy(left)
    right["episode_id"] = "OTHER"
    right["url"] = "http://another-origin/products?episode_id=OTHER"
    right["elements"][0]["element_id"] = "another-runtime-id"
    right["oracle_answer"] = "HIDDEN-DIFFERENT"
    right["verifier"] = {"success": False}
    assert public_anchor_signature(left, prompt_token_ids=[1, 2]) == public_anchor_signature(
        right, prompt_token_ids=[1, 2]
    )
    right["visible_text"] = "A different public product"
    assert public_anchor_signature(left, prompt_token_ids=[1, 2]) != public_anchor_signature(
        right, prompt_token_ids=[1, 2]
    )
    assert public_anchor_signature(left, prompt_token_ids=[1, 2]) != public_anchor_signature(
        left, prompt_token_ids=[1, 3]
    )


def test_standardized_advantage_zero_variance_falls_back_to_exact_zero():
    assert standardized_advantages([1.0, 1.0, 1.0, 1.0]) == (0.0, 0.0, 0.0, 0.0)
    mixed = standardized_advantages([1.0, 0.0, 1.0, 0.0])
    assert mixed[0] == pytest.approx(1.0, abs=3e-6)
    assert mixed[1] == pytest.approx(-1.0, abs=3e-6)


def test_grpo_broadcasts_only_macro_advantage_to_every_turn():
    report = assign_group_credit(_group(BASELINE_METHOD), BASELINE_METHOD)
    assert report["formula_version"] == CREDIT_FORMULA_VERSION
    assert report["metrics"]["informative_micro_turn_count"] == 0
    for trajectory_index, turns in enumerate(report["turn_credit"]):
        assert all(turn["micro_advantage"] == 0.0 for turn in turns)
        assert all(
            turn["turn_advantage"] == pytest.approx(report["macro_advantages"][trajectory_index])
            for turn in turns
        )


def test_step_aware_adds_micro_only_at_shared_first_visit_and_falls_back_elsewhere():
    report = assign_group_credit(_group(STEP_AWARE_METHOD), STEP_AWARE_METHOD)
    assert report["metrics"]["shared_anchor_count"] == 1
    assert report["metrics"]["informative_anchor_count"] == 1
    assert report["metrics"]["informative_micro_turn_count"] == 4
    for trajectory_index, turns in enumerate(report["turn_credit"]):
        first, second = turns
        assert abs(first["micro_advantage"]) > 0.9
        assert first["turn_advantage"] == pytest.approx(
            first["macro_advantage"] + first["micro_advantage"]
        )
        assert second["micro_advantage"] == 0.0
        assert second["turn_advantage"] == pytest.approx(second["macro_advantage"])


def test_step_aware_all_equal_rewards_has_no_fake_process_signal():
    report = assign_group_credit(
        _group(STEP_AWARE_METHOD, rewards=(0.0, 0.0, 0.0, 0.0)),
        STEP_AWARE_METHOD,
    )
    assert report["macro_advantages"] == [0.0, 0.0, 0.0, 0.0]
    assert report["metrics"]["informative_anchor_count"] == 0
    assert all(
        turn["turn_advantage"] == 0.0
        for trajectory in report["turn_credit"]
        for turn in trajectory
    )
