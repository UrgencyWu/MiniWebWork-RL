from __future__ import annotations

from pathlib import Path

from miniwebwork.long_horizon_rl.contracts import RunIdentity
from miniwebwork.long_horizon_rl.credit import CREDIT_FORMULA_VERSION
from miniwebwork.long_horizon_rl.journal import CollectionStore
from miniwebwork.long_horizon_rl.rollout import (
    admit_next_group,
    run_atomic_k4_group,
)


def _identity():
    return RunIdentity(
        study_id="m4_long_horizon_credit_v1",
        git_sha="1" * 40,
        method="multi_turn_grpo",
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
    )


def _raw_turn(rollout_index):
    prompt_ids = [1, 2]
    generated_ids = [10 + rollout_index, 20]
    return {
        "model_turn_index": 1,
        "environment_step_index": 0,
        "observation": {
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
        },
        "rendered_prompt_sha256": "7" * 64,
        "prompt_token_ids": prompt_ids,
        "input_tokens": len(prompt_ids),
        "raw_output": '{"action":"finish"}',
        "generated_token_ids": generated_ids,
        "token_logprobs": [-0.1, -0.2],
        "sampling_logprobs": [-0.1, -0.2],
        "output_tokens": len(generated_ids),
        "request_id": f"request-{rollout_index}",
        "sampling_seed": 100 + rollout_index,
        "generation_backend": "vllm_async",
        "latency_ms": 10.0,
        "queue_wait_ms": 1.0,
        "first_token_latency_ms": 2.0,
        "generation_time_ms": 7.0,
        "strict_json_success": True,
        "fallback_used": False,
        "schema_valid": True,
        "action": {"action": "finish"},
        "errors": [],
        "action_result": {"success": True},
        "reward": 0.0,
        "terminated": True,
        "truncated": False,
    }


def _result(rollout_index, reward):
    return {
        "task_id": "TASK-1",
        "episode_id": f"EP-{rollout_index}",
        "success": reward == 1.0,
        "reward": reward,
        "rollout_valid": True,
        "failure_origin": "none" if reward == 1.0 else "policy",
        "termination_reason": "finish",
        "model_turns": 1,
        "environment_steps": 1,
        "failure_reasons": [],
        "elapsed_s": 0.1,
    }


def test_four_complete_workers_commit_one_atomic_group_with_full_artifacts(tmp_path):
    identity = _identity()
    store = CollectionStore(tmp_path / "collection", identity)

    def worker(rollout_index, writer):
        turn = _raw_turn(rollout_index)
        writer.on_turn_generated(turn)
        writer.on_turn_completed(turn)
        return _result(rollout_index, float(rollout_index % 2 == 0))

    result = run_atomic_k4_group(
        store=store,
        identity=identity,
        group_id="group-0000",
        task_id="TASK-1",
        worker=worker,
    )
    assert result["committed"] is True
    assert result["group"]["K"] == 4
    assert result["group"]["generated_action_tokens"] == 8
    assert store.journal.generated_action_tokens == 8
    assert len(store.load_committed_groups()) == 1
    for rollout_index in range(4):
        trajectory_root = (
            tmp_path
            / "collection/attempts/group-0000/attempt-0000"
            / f"group-0000.a0.r{rollout_index}"
        )
        assert (trajectory_root / "turn-0001.json").is_file()
        assert (trajectory_root / "trajectory.json").is_file()


def test_worker_crash_after_charge_retains_cost_invalidates_and_archives_entire_group(tmp_path):
    identity = _identity()
    store = CollectionStore(tmp_path / "collection", identity)

    def worker(rollout_index, writer):
        turn = _raw_turn(rollout_index)
        writer.on_turn_generated(turn)
        if rollout_index == 2:
            raise RuntimeError("intentional worker crash")
        writer.on_turn_completed(turn)
        return _result(rollout_index, 0.0)

    failed = run_atomic_k4_group(
        store=store,
        identity=identity,
        group_id="group-0000",
        task_id="TASK-1",
        worker=worker,
    )
    assert failed["committed"] is False
    assert store.journal.generated_action_tokens == 8
    assert store.load_committed_groups() == ()
    archive = tmp_path / "collection/invalidated_attempts/group-0000/attempt-0000"
    assert archive.is_dir()

    def healthy_worker(rollout_index, writer):
        turn = _raw_turn(rollout_index)
        writer.on_turn_generated(turn)
        writer.on_turn_completed(turn)
        return _result(rollout_index, float(rollout_index % 2 == 0))

    retried = run_atomic_k4_group(
        store=store,
        identity=identity,
        group_id="group-0000",
        task_id="TASK-1",
        worker=healthy_worker,
    )
    assert retried["committed"] is True
    assert retried["attempt_index"] == 1
    assert store.journal.generated_action_tokens == 16


def test_budget_admission_reserves_worst_case_complete_group():
    first = admit_next_group(
        global_tokens_before_iteration=0,
        current_iteration_tokens=239760,
    )
    assert first.allowed is True
    assert first.maximum_group_reserve == 10240
    blocked = admit_next_group(
        global_tokens_before_iteration=0,
        current_iteration_tokens=239761,
    )
    assert blocked.allowed is False
    assert blocked.reason == "insufficient_worst_case_group_reserve"


def test_budget_admission_reserves_every_concurrent_inflight_group():
    first = admit_next_group(
        global_tokens_before_iteration=0,
        current_iteration_tokens=230000,
        reserved_inflight_groups=0,
    )
    assert first.allowed is True
    second = admit_next_group(
        global_tokens_before_iteration=0,
        current_iteration_tokens=230000,
        reserved_inflight_groups=1,
    )
    assert second.allowed is False
    assert second.total_inflight_reserve == 20480
