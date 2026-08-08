from __future__ import annotations

import asyncio
import threading
import time

from miniwebwork.long_horizon_rl.browser_pool import (
    BrowserWorkerPoolConfig,
    VLLMBrowserWorkerPool,
)
from miniwebwork.long_horizon_rl.contracts import RunIdentity
from miniwebwork.long_horizon_rl.credit import CREDIT_FORMULA_VERSION
from miniwebwork.long_horizon_rl.journal import CollectionStore
from miniwebwork.long_horizon_rl.orchestrator import (
    collect_iteration,
    sampler_task_order_sha256,
)
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


class _FakeEnvironment:
    def __init__(self, lane_index):
        self.lane_index = lane_index
        self.closed = False

    def close(self):
        self.closed = True


def _turn(context, task_id):
    generated_ids = [10 + context.rollout_index, 20]
    return {
        "model_turn_index": 1,
        "environment_step_index": 0,
        "observation": {
            "schema_version": "1.0",
            "task_id": task_id,
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
        "prompt_token_ids": [1, 2],
        "input_tokens": 2,
        "raw_output": '{"action":"finish"}',
        "generated_token_ids": generated_ids,
        "token_logprobs": [-0.1, -0.2],
        "sampling_logprobs": [-0.1, -0.2],
        "output_tokens": 2,
        "request_id": (
            f"{context.group_id}-a{context.attempt_index}-r{context.rollout_index}"
        ),
        "sampling_seed": 100 + context.rollout_index,
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


def _result(context, task_id):
    reward = float(context.rollout_index % 2 == 0)
    return {
        "task_id": task_id,
        "episode_id": f"EP-{context.group_id}-{context.rollout_index}",
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


def _pool(tmp_path, *, workers, identity, activity):
    environments = []

    def environment_factory(lane_index):
        environment = _FakeEnvironment(lane_index)
        environments.append(environment)
        return environment

    def episode_runner(
        task_id,
        environment,
        context,
        _max_model_turns,
        _max_environment_steps,
        *,
        turn_generated_callback,
        turn_completed_callback,
    ):
        with activity["lock"]:
            activity["active"] += 1
            activity["maximum"] = max(activity["maximum"], activity["active"])
            activity["lanes"].add(environment.lane_index)
        try:
            time.sleep(0.04)
            turn = _turn(context, task_id)
            turn_generated_callback(turn)
            turn_completed_callback(turn)
            return _result(context, task_id)
        finally:
            with activity["lock"]:
                activity["active"] -= 1

    loop = asyncio.new_event_loop()
    pool = VLLMBrowserWorkerPool(
        identity=identity,
        engine=object(),
        event_loop=loop,
        event_loop_thread_id=threading.get_ident(),
        config=BrowserWorkerPoolConfig(total_workers=workers),
        environment_factory=environment_factory,
        backend_factory=lambda context: context,
        agent_factory=lambda backend: backend,
        episode_runner=episode_runner,
    )
    return pool, loop, environments


def test_one_worker_serializes_four_candidates_but_preserves_exact_k(tmp_path):
    tasks = (_tasks()[0],)
    identity = _identity(tasks)
    store = CollectionStore(tmp_path / "collection", identity)
    activity = {"lock": threading.Lock(), "active": 0, "maximum": 0, "lanes": set()}
    pool, loop, environments = _pool(
        tmp_path,
        workers=1,
        identity=identity,
        activity=activity,
    )
    try:
        report = collect_iteration(
            store=store,
            identity=identity,
            tasks=tasks,
            initial_sampler_state=None,
            group_runner=lambda group_id, task: pool.run_group(
                store=store,
                group_id=group_id,
                task=task,
            ),
            global_generated_action_tokens_before=0,
            maximum_concurrent_groups=1,
        )
    finally:
        pool.close()
        loop.close()
    assert report["committed_group_count"] == 1
    assert store.load_committed_groups()[0]["K"] == 4
    assert activity["maximum"] == 1
    assert activity["lanes"] == {0}
    assert all(environment.closed for environment in environments)


def test_eight_workers_fill_two_disjoint_k4_slots_concurrently(tmp_path):
    tasks = _tasks()
    identity = _identity(tasks)
    store = CollectionStore(tmp_path / "collection", identity)
    activity = {"lock": threading.Lock(), "active": 0, "maximum": 0, "lanes": set()}
    pool, loop, environments = _pool(
        tmp_path,
        workers=8,
        identity=identity,
        activity=activity,
    )
    try:
        report = collect_iteration(
            store=store,
            identity=identity,
            tasks=tasks,
            initial_sampler_state=None,
            group_runner=lambda group_id, task: pool.run_group(
                store=store,
                group_id=group_id,
                task=task,
            ),
            global_generated_action_tokens_before=0,
            maximum_concurrent_groups=2,
        )
    finally:
        pool.close()
        loop.close()
    assert report["committed_group_count"] == 2
    assert activity["maximum"] == 8
    assert activity["lanes"] == set(range(8))
    assert len(environments) == 8
    assert all(environment.closed for environment in environments)
