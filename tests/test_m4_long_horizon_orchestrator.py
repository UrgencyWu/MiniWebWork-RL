from __future__ import annotations

import json
import threading
import time

from miniwebwork.long_horizon_rl.contracts import RunIdentity
from miniwebwork.long_horizon_rl.credit import CREDIT_FORMULA_VERSION
from miniwebwork.long_horizon_rl.journal import CollectionStore
from miniwebwork.long_horizon_rl.orchestrator import (
    collect_iteration,
    load_public_task_roster,
    restore_sampler,
    sampler_task_order_sha256,
)
from miniwebwork.long_horizon_rl.rollout import run_atomic_k4_group
from miniwebwork.long_horizon_rl.sampler import TaskDescriptor


def _tasks():
    return (
        TaskDescriptor("TASK-A", "family-a", "basic"),
        TaskDescriptor("TASK-B", "family-b", "long"),
    )


def _identity(tasks=None):
    tasks = tasks or _tasks()
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
        task_order_sha256=sampler_task_order_sha256(tasks, study_seed=20260801),
        base_model_manifest_sha256="8" * 64,
        runtime_contract_sha256="9" * 64,
        input_adapter_sha256="6" * 64,
        input_rollout_adapter_sha256="a" * 64,
        input_adapter_semantic_sha256="b" * 64,
    )


def _raw_turn(writer, rollout_index):
    prompt_ids = [1, 2]
    generated_ids = [10 + rollout_index, 20]
    return {
        "model_turn_index": 1,
        "environment_step_index": 0,
        "observation": {
            "schema_version": "1.0",
            "task_id": writer.task_id,
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
        "request_id": (
            f"{writer.group_id}-a{writer.attempt_index}-r{rollout_index}"
        ),
        "sampling_seed": 100 + writer.attempt_index * 10 + rollout_index,
        "generation_backend": "vllm_async",
        "adapter_sha256": "6" * 64,
        "rollout_adapter_sha256": "a" * 64,
        "adapter_semantic_sha256": "b" * 64,
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


def _result(writer, rollout_index):
    reward = float(rollout_index % 2 == 0)
    return {
        "task_id": writer.task_id,
        "episode_id": f"EP-{writer.group_id}-{rollout_index}",
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


def _healthy_worker(rollout_index, writer):
    turn = _raw_turn(writer, rollout_index)
    writer.on_turn_generated(turn)
    writer.on_turn_completed(turn)
    return _result(writer, rollout_index)


def _runner(store, identity, worker=_healthy_worker):
    def run(group_id, task):
        return run_atomic_k4_group(
            store=store,
            identity=identity,
            group_id=group_id,
            task_id=task.task_id,
            worker=worker,
        )

    return run


def test_public_roster_loader_projects_only_sampler_fields(tmp_path):
    path = tmp_path / "public.jsonl"
    rows = [
        {
            "task_id": "TASK-B",
            "task_family": "family-b",
            "horizon_stratum": "long",
            "instruction": "not retained",
            "oracle_min_env_actions": 18,
        },
        {
            "task_id": "TASK-A",
            "task_family": "family-a",
            "horizon_stratum": "basic",
            "instruction": "not retained",
            "oracle_min_env_actions": 7,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert load_public_task_roster(path, expected_count=2) == _tasks()


def test_sampler_restore_uses_durable_counts_and_recomputes_derived_fields():
    initial = restore_sampler(_tasks(), study_seed=20260801, state=None)
    initial.record_committed_group("TASK-A", [1.0, 0.0, 1.0, 0.0])
    initial.record_infra_invalid_attempt("TASK-B")
    restored = restore_sampler(
        _tasks(),
        study_seed=20260801,
        state=initial.audit_dict(),
    )
    assert restored.audit_dict() == initial.audit_dict()


def test_collection_orchestrator_commits_two_groups_and_freezes_sorted_manifest(tmp_path):
    tasks = _tasks()
    identity = _identity(tasks)
    store = CollectionStore(tmp_path / "collection", identity)
    report = collect_iteration(
        store=store,
        identity=identity,
        tasks=tasks,
        initial_sampler_state=None,
        group_runner=_runner(store, identity),
        global_generated_action_tokens_before=0,
    )
    assert report["ready_for_update"] is True
    assert report["committed_group_count"] == 2
    assert report["collection_manifest"]["group_count"] == 2
    assert report["collection_manifest"]["all_generated_action_tokens"] == 16
    assert [group["group_id"] for group in store.load_committed_groups()] == [
        "i0000-g0000",
        "i0000-g0001",
    ]


def test_resume_rebuilds_invalid_cost_and_sampler_before_retry(tmp_path):
    tasks = _tasks()
    identity = _identity(tasks)
    store = CollectionStore(tmp_path / "collection", identity)

    def crash_one(rollout_index, writer):
        turn = _raw_turn(writer, rollout_index)
        writer.on_turn_generated(turn)
        if rollout_index == 2:
            raise RuntimeError("intentional crash")
        writer.on_turn_completed(turn)
        return _result(writer, rollout_index)

    failed = run_atomic_k4_group(
        store=store,
        identity=identity,
        group_id="i0000-g0000",
        task_id="TASK-A",
        worker=crash_one,
    )
    assert failed["committed"] is False
    report = collect_iteration(
        store=store,
        identity=identity,
        tasks=tasks,
        initial_sampler_state=None,
        group_runner=_runner(store, identity),
        global_generated_action_tokens_before=0,
    )
    assert report["committed_group_count"] == 2
    assert report["infra_invalid_attempt_count"] == 1
    assert report["collection_manifest"]["all_generated_action_tokens"] == 24
    signal = report["collection_manifest"]["task_sampler_state"]["signals"]["TASK-A"]
    assert signal["infra_invalid_attempts"] == 1
    assert signal["committed_groups"] == 1


def test_one_failed_group_does_not_archive_concurrent_healthy_peer(tmp_path):
    tasks = _tasks()
    identity = _identity(tasks)
    store = CollectionStore(tmp_path / "collection", identity)

    def mixed_worker(rollout_index, writer):
        turn = _raw_turn(writer, rollout_index)
        writer.on_turn_generated(turn)
        if writer.group_id == "i0000-g0000" and writer.attempt_index == 0 and rollout_index == 2:
            raise RuntimeError("one group fails once")
        if writer.group_id == "i0000-g0001":
            time.sleep(0.05)
        writer.on_turn_completed(turn)
        return _result(writer, rollout_index)

    report = collect_iteration(
        store=store,
        identity=identity,
        tasks=tasks,
        initial_sampler_state=None,
        group_runner=_runner(store, identity, worker=mixed_worker),
        global_generated_action_tokens_before=0,
    )
    assert report["committed_group_count"] == 2
    assert report["infra_invalid_attempt_count"] == 1
    assert (store.invalidated_attempts_dir / "i0000-g0000/attempt-0000").is_dir()
    assert not (store.invalidated_attempts_dir / "i0000-g0001").exists()


def test_global_budget_reservation_prevents_two_groups_being_inflight(tmp_path):
    tasks = _tasks()
    identity = _identity(tasks)
    store = CollectionStore(tmp_path / "collection", identity)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def monitored(group_id, task):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.05)
            return _runner(store, identity)(group_id, task)
        finally:
            with lock:
                active -= 1

    report = collect_iteration(
        store=store,
        identity=identity,
        tasks=tasks,
        initial_sampler_state=None,
        group_runner=monitored,
        global_generated_action_tokens_before=230000,
    )
    assert report["committed_group_count"] == 2
    assert maximum_active == 1


def test_group_launches_are_deterministically_staggered(tmp_path):
    tasks = tuple(
        TaskDescriptor(f"TASK-{index}", f"family-{index}", "long")
        for index in range(3)
    )
    identity = _identity(tasks)
    store = CollectionStore(tmp_path / "collection", identity)
    starts = []
    lock = threading.Lock()

    def monitored(group_id, task):
        with lock:
            starts.append((group_id, time.monotonic()))
        return _runner(store, identity)(group_id, task)

    report = collect_iteration(
        store=store,
        identity=identity,
        tasks=tasks,
        initial_sampler_state=None,
        group_runner=monitored,
        global_generated_action_tokens_before=0,
        maximum_concurrent_groups=3,
        group_launch_stagger_seconds=0.02,
    )
    assert report["committed_group_count"] == 3
    assert [group_id for group_id, _ in starts] == [
        "i0000-g0000",
        "i0000-g0001",
        "i0000-g0002",
    ]
    assert starts[1][1] - starts[0][1] >= 0.015
    assert starts[2][1] - starts[1][1] >= 0.015


def test_no_group_starts_when_global_budget_cannot_reserve_one_complete_k4(tmp_path):
    tasks = _tasks()
    identity = _identity(tasks)
    store = CollectionStore(tmp_path / "collection", identity)
    calls = 0

    def forbidden(_group_id, _task):
        nonlocal calls
        calls += 1
        raise AssertionError("group runner must not be called")

    report = collect_iteration(
        store=store,
        identity=identity,
        tasks=tasks,
        initial_sampler_state=None,
        group_runner=forbidden,
        global_generated_action_tokens_before=239761,
    )
    assert calls == 0
    assert report["ready_for_update"] is False
    assert report["stopped_for_token_budget"] is True
    assert report["collection_manifest"]["group_count"] == 0
    assert report["collection_manifest"]["stopped_for_token_budget"] is True
