"""Single-GPU preflight runner joining collection, learner, and atomic recovery."""

from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..m4_long_horizon_protocol import (
    ONLINE_METHODS,
    ONLINE_SEEDS,
    PROJECT_ROOT,
    PROMPT_CONTRACT,
    PROMPT_PATH,
    STUDY_ID,
    assert_dataset_binding,
    assert_formal_submission_closed,
    assert_preflight_output,
    load_study_manifest,
)
from .adapter_view import (
    VLLM_ADAPTER_MAPPING_CONTRACT,
    build_vllm_adapter_view,
)
from .browser_pool import BrowserWorkerPoolConfig, VLLMBrowserWorkerPool
from .contracts import (
    RunIdentity,
    atomic_write_json,
    directory_sha256,
    sha256_file,
    sha256_json,
)
from .credit import CREDIT_FORMULA_VERSION
from .iteration import IterationStore
from .journal import CollectionStore
from .learner import (
    build_or_load_policy_optimizer,
    create_bootstrap_optimizer_artifact,
    load_trainable_policy_model,
    save_policy_update_artifacts,
    train_policy_groups,
)
from .model_manifest import validate_base_model_manifest
from .orchestrator import (
    collect_iteration,
    load_frozen_train_roster,
    restore_sampler,
    sampler_task_order_sha256,
)
from .runtime_contract import PARITY_THRESHOLDS, load_online_runtime_contract
from .sft_selection import load_sft_preflight_selection
from .vllm_backend import AsyncVLLMGenerationEngine, VLLMBackendConfig

ONLINE_PREFLIGHT_RUN_CONFIG_SCHEMA = "m4_long_horizon_online_preflight_run_v2"
ONLINE_PREFLIGHT_REPORT_SCHEMA = "m4_long_horizon_online_preflight_report_v2"
ONLINE_PREFLIGHT_RECOVERY_REPORT_SCHEMA = (
    "m4_long_horizon_online_preflight_recovery_report_v2"
)
TARGET_ITERATION_INDEX = 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    _require(len(value) in (40, 64), "git object id length drift")
    return value


def _assert_git_clean() -> str:
    status = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    _require(not status.stdout.strip(), "online preflight requires a clean tracked worktree")
    return _git_sha()


def _resolve_state_artifact(
    store: IterationStore,
    state: Mapping[str, Any],
    field: str,
) -> Path:
    descriptor = state[field]
    path = Path(descriptor["path"]).expanduser()
    if not path.is_absolute():
        path = store.root / path
    resolved = path.resolve()
    _require(resolved.exists(), f"run-state artifact is missing: {field}")
    return resolved


def identity_from_run_state(state: Mapping[str, Any]) -> RunIdentity:
    """Reconstruct the exact current policy identity from durable run state."""

    return RunIdentity(
        study_id=state["study_id"],
        git_sha=state["git_sha"],
        method=state["method"],
        seed=state["seed"],
        iteration_index=state["current_iteration_index"],
        policy_version=state["current_policy_version"],
        dataset_manifest_sha256=state["dataset_manifest_sha256"],
        seed_manifest_sha256=state["seed_manifest_sha256"],
        prompt_contract=PROMPT_CONTRACT,
        prompt_sha256=state["prompt_sha256"],
        credit_formula_version=state["credit_formula_version"],
        task_order_sha256=state["task_order_sha256"],
        base_model_manifest_sha256=state["base_model_manifest_sha256"],
        runtime_contract_sha256=state["runtime_contract_sha256"],
        input_adapter_sha256=state["current_adapter"]["sha256"],
        input_rollout_adapter_sha256=state["current_rollout_adapter"][
            "sha256"
        ],
        input_adapter_semantic_sha256=state[
            "current_adapter_semantic_sha256"
        ],
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    _require(bool(values), "cannot summarize an empty metric")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize_collection_performance(
    groups: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    _require(bool(groups), "cannot summarize an empty collection")
    _require(elapsed_seconds > 0, "collection elapsed time must be positive")
    trajectories = [trajectory for group in groups for trajectory in group["trajectories"]]
    turns = [turn for trajectory in trajectories for turn in trajectory["turns"]]
    generated_tokens = sum(len(turn["generated_token_ids"]) for turn in turns)
    trajectory_lengths = [len(trajectory["turns"]) for trajectory in trajectories]
    queue_wait = [float(turn["queue_wait_ms"]) for turn in turns]
    first_token = [float(turn["first_token_latency_ms"]) for turn in turns]
    generation_time = [float(turn["generation_time_ms"]) for turn in turns]
    scale = 3600.0 / elapsed_seconds
    return {
        "elapsed_seconds": elapsed_seconds,
        "group_count": len(groups),
        "trajectory_count": len(trajectories),
        "model_turn_count": len(turns),
        "generated_action_tokens": generated_tokens,
        "trajectories_per_hour": len(trajectories) * scale,
        "model_turns_per_hour": len(turns) * scale,
        "generated_action_tokens_per_hour": generated_tokens * scale,
        "trajectory_model_turns": {
            "minimum": min(trajectory_lengths),
            "p50": _percentile(trajectory_lengths, 0.5),
            "p95": _percentile(trajectory_lengths, 0.95),
            "maximum": max(trajectory_lengths),
        },
        "queue_wait_ms": {
            "mean": statistics.fmean(queue_wait),
            "p50": _percentile(queue_wait, 0.5),
            "p95": _percentile(queue_wait, 0.95),
            "maximum": max(queue_wait),
        },
        "first_token_latency_ms": {
            "mean": statistics.fmean(first_token),
            "p50": _percentile(first_token, 0.5),
            "p95": _percentile(first_token, 0.95),
            "maximum": max(first_token),
        },
        "generation_time_ms": {
            "mean": statistics.fmean(generation_time),
            "p50": _percentile(generation_time, 0.5),
            "p95": _percentile(generation_time, 0.95),
            "maximum": max(generation_time),
        },
    }


def audit_post_wake_generation(
    result: Any,
    *,
    expected_adapter_sha256: str,
    expected_rollout_adapter_sha256: str,
    expected_adapter_semantic_sha256: str,
) -> dict[str, Any]:
    """Prove that the updated adapter can generate after the same-GPU wake."""

    _require(not result.error, "post-wake generation returned an error")
    _require(result.new_tokens > 0, "post-wake generation produced no tokens")
    _require(
        result.new_tokens == len(result.generated_token_ids),
        "post-wake generated-token count drift",
    )
    _require(
        len(result.logprobs) == result.new_tokens
        and len(result.sampling_logprobs) == result.new_tokens,
        "post-wake logprob/token count drift",
    )
    _require(
        result.adapter_sha256 == expected_adapter_sha256,
        "post-wake canonical adapter identity drift",
    )
    _require(
        result.rollout_adapter_sha256 == expected_rollout_adapter_sha256,
        "post-wake rollout adapter identity drift",
    )
    _require(
        result.adapter_semantic_sha256 == expected_adapter_semantic_sha256,
        "post-wake adapter semantic identity drift",
    )
    behavior_sampling_maximum = max(
        abs(float(behavior) - float(sampling))
        for behavior, sampling in zip(result.logprobs, result.sampling_logprobs)
    )
    _require(
        behavior_sampling_maximum
        <= PARITY_THRESHOLDS["behavior_sampling_maximum_absolute_difference"],
        "post-wake behavior/sampling parity failed",
    )
    return {
        "request_id": result.request_id,
        "sampling_seed": result.sampling_seed,
        "input_tokens": result.input_tokens,
        "generated_action_tokens": result.new_tokens,
        "generated_token_ids_sha256": sha256_json(result.generated_token_ids),
        "raw_text_sha256": sha256_json({"raw_text": result.raw_text}),
        "behavior_sampling_maximum_absolute_difference": behavior_sampling_maximum,
        "adapter_sha256": result.adapter_sha256,
        "rollout_adapter_sha256": result.rollout_adapter_sha256,
        "adapter_semantic_sha256": result.adapter_semantic_sha256,
        "latency_ms": result.latency_ms,
        "queue_wait_ms": result.queue_wait_ms,
        "first_token_latency_ms": result.first_token_latency_ms,
        "generation_time_ms": result.generation_time_ms,
        "cost_scope": "preflight_phase_switch_diagnostic_not_formal_training",
        "passed": True,
    }


def _phase_event(root: Path, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    directory = Path(root) / "phase_events"
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("*.json"))
    event = {
        "schema_version": "m4_long_horizon_phase_event_v1",
        "sequence": len(existing),
        "event_type": event_type,
        "timestamp_ns": time.time_ns(),
        "payload": dict(payload),
    }
    event["event_sha256"] = sha256_json(event)
    atomic_write_json(directory / f"{len(existing):06d}-{event_type}.json", event)
    return event


def _load_or_write_run_config(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(root) / "run_config.json"
    normalized = dict(payload)
    normalized["run_config_sha256"] = sha256_json(normalized)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _require(existing == normalized, "online preflight run config drift")
        return existing
    atomic_write_json(path, normalized)
    return normalized


def _initialize_or_load_state(
    *,
    root: Path,
    run_config: Mapping[str, Any],
    initial_adapter: Path,
    initial_rollout_adapter: Path,
    initial_adapter_semantic_sha256: str,
) -> tuple[IterationStore, dict[str, Any]]:
    tasks = load_frozen_train_roster()
    store = IterationStore(Path(root) / "state")
    if not store.initialized:
        identity = RunIdentity(
            study_id=STUDY_ID,
            git_sha=run_config["git_sha"],
            method=run_config["method"],
            seed=run_config["seed"],
            iteration_index=0,
            policy_version="policy_0000",
            dataset_manifest_sha256=run_config["dataset_manifest_sha256"],
            seed_manifest_sha256=run_config["seed_manifest_sha256"],
            prompt_contract=PROMPT_CONTRACT,
            prompt_sha256=run_config["prompt_sha256"],
            credit_formula_version=CREDIT_FORMULA_VERSION,
            task_order_sha256=run_config["task_order_sha256"],
            base_model_manifest_sha256=run_config["base_model_manifest_sha256"],
            runtime_contract_sha256=run_config["runtime_contract_sha256"],
            input_adapter_sha256=run_config["initial_adapter_sha256"],
            input_rollout_adapter_sha256=run_config[
                "initial_rollout_adapter_sha256"
            ],
            input_adapter_semantic_sha256=run_config[
                "initial_adapter_semantic_sha256"
            ],
        )
        bootstrap_optimizer = Path(root) / "bootstrap" / "optimizer.pt"
        create_bootstrap_optimizer_artifact(path=bootstrap_optimizer, identity=identity)
        sampler = restore_sampler(tasks, study_seed=identity.seed, state=None)
        store.initialize(
            identity=identity,
            input_adapter_path=initial_adapter,
            input_rollout_adapter_path=initial_rollout_adapter,
            input_adapter_semantic_sha256=initial_adapter_semantic_sha256,
            input_optimizer_path=bootstrap_optimizer,
            task_sampler_state=sampler.audit_dict(),
        )
    else:
        store.reconcile_committed_iterations()
    state = store.load_state()
    fixed = {
        "study_id": STUDY_ID,
        "method": run_config["method"],
        "seed": run_config["seed"],
        "git_sha": run_config["git_sha"],
        "dataset_manifest_sha256": run_config["dataset_manifest_sha256"],
        "seed_manifest_sha256": run_config["seed_manifest_sha256"],
        "prompt_sha256": run_config["prompt_sha256"],
        "credit_formula_version": CREDIT_FORMULA_VERSION,
        "task_order_sha256": run_config["task_order_sha256"],
        "base_model_manifest_sha256": run_config["base_model_manifest_sha256"],
        "runtime_contract_sha256": run_config["runtime_contract_sha256"],
    }
    for field, expected in fixed.items():
        _require(state.get(field) == expected, f"run state/config {field} drift")
    return store, state


def prepare_online_preflight(
    *,
    output_dir: Path,
    method: str,
    seed: int,
    initial_adapter: Path,
    base_model_manifest: Path,
    browser_workers: int,
    maximum_tasks: int,
    learner_microbatch_size: int,
    collection_only: bool,
) -> dict[str, Any]:
    """Validate all immutable bindings and initialize a resumable run root."""

    assert_formal_submission_closed()
    study = load_study_manifest()
    runtime = load_online_runtime_contract()
    dataset = assert_dataset_binding(study["payload"])
    selection = load_sft_preflight_selection()
    root = assert_preflight_output(output_dir, study["payload"])
    root.mkdir(parents=True, exist_ok=True)
    _require(method in ONLINE_METHODS, "unsupported online preflight method")
    _require(seed in ONLINE_SEEDS, "online preflight seed drift")
    worker_config = BrowserWorkerPoolConfig(total_workers=browser_workers)
    worker_config.validate()
    _require(1 <= maximum_tasks <= 32, "online preflight task count drift")
    _require(learner_microbatch_size in (1, 2, 4, 8), "online learner microbatch candidate drift")
    adapter = Path(initial_adapter).expanduser().resolve()
    _require(adapter.is_dir(), "initial preflight adapter is missing")
    adapter_sha = directory_sha256(adapter)
    _require(
        adapter_sha == selection["payload"]["adapter_audit"]["directory_sha256"],
        "initial adapter is not the frozen disposable SFT preflight adapter",
    )
    model_manifest = validate_base_model_manifest(base_model_manifest, verify_files=True)
    base_model = Path(model_manifest["payload"]["base_model_path"])
    _require(
        str(base_model) == runtime["payload"]["model_contract"]["base_model_path"],
        "base model/runtime contract drift",
    )
    git_sha = _assert_git_clean()
    initial_rollout_adapter = root / "bootstrap" / "rollout_adapter_view"
    adapter_view = build_vllm_adapter_view(
        source_adapter=adapter,
        destination=initial_rollout_adapter,
        base_model=base_model,
    )
    tasks = load_frozen_train_roster()
    payload = {
        "schema_version": ONLINE_PREFLIGHT_RUN_CONFIG_SCHEMA,
        "study_id": STUDY_ID,
        "formal_training": False,
        "target_iteration_index": TARGET_ITERATION_INDEX,
        "method": method,
        "seed": seed,
        "git_sha": git_sha,
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "seed_manifest_sha256": dataset["seed_manifest_sha256"],
        "prompt_contract": PROMPT_CONTRACT,
        "prompt_sha256": sha256_file(PROMPT_PATH),
        "credit_formula_version": CREDIT_FORMULA_VERSION,
        "task_order_sha256": sampler_task_order_sha256(tasks, study_seed=seed),
        "base_model_path": str(base_model),
        "base_model_manifest_path": model_manifest["path"],
        "base_model_manifest_sha256": model_manifest["sha256"],
        "runtime_contract_sha256": runtime["sha256"],
        "initial_adapter_path": str(adapter),
        "initial_adapter_sha256": adapter_sha,
        "initial_rollout_adapter_path": str(initial_rollout_adapter),
        "initial_rollout_adapter_sha256": adapter_view[
            "view_directory_sha256"
        ],
        "initial_adapter_semantic_sha256": adapter_view[
            "semantic_tensor_sha256"
        ],
        "adapter_mapping_contract": VLLM_ADAPTER_MAPPING_CONTRACT,
        "browser_workers": browser_workers,
        "maximum_concurrent_k4_groups": worker_config.concurrent_group_slots,
        "maximum_tasks": maximum_tasks,
        "learner_microbatch_size": learner_microbatch_size,
        "collection_only": bool(collection_only),
        "parity_thresholds": dict(PARITY_THRESHOLDS),
    }
    run_config = _load_or_write_run_config(root, payload)
    state_store, state = _initialize_or_load_state(
        root=root,
        run_config=run_config,
        initial_adapter=adapter,
        initial_rollout_adapter=initial_rollout_adapter,
        initial_adapter_semantic_sha256=adapter_view[
            "semantic_tensor_sha256"
        ],
    )
    return {
        "root": root,
        "run_config": run_config,
        "state_store": state_store,
        "state": state,
        "tasks": tasks,
        "worker_config": worker_config,
    }


def _release_learner_memory(model: Any, optimizer: Any, tokenizer: Any) -> None:
    del optimizer
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def execute_learner_update(
    *,
    state_store: IterationStore,
    collection_store: CollectionStore,
    identity: RunIdentity,
    collection_manifest: Mapping[str, Any],
    base_model: Path,
    learner_microbatch_size: int,
) -> dict[str, Any]:
    """Replay one frozen collection and atomically advance adapter/optimizer state."""

    state = state_store.assert_identity_matches_state(identity)
    state_store.recover_interrupted_update(identity)
    paths = state_store.begin_update(
        identity=identity,
        collection_manifest=collection_manifest,
    )
    input_adapter = _resolve_state_artifact(state_store, state, "current_adapter")
    input_optimizer = _resolve_state_artifact(state_store, state, "current_optimizer")
    model = tokenizer = optimizer = None
    started = time.monotonic()
    try:
        model, tokenizer = load_trainable_policy_model(
            base_model=base_model,
            adapter_path=input_adapter,
            device="cuda:0",
        )
        optimizer = build_or_load_policy_optimizer(
            model=model,
            identity=identity,
            optimizer_artifact=input_optimizer,
        )
        groups = collection_store.load_committed_groups()
        report = train_policy_groups(
            model=model,
            optimizer=optimizer,
            groups=groups,
            method=identity.method,
            identity=identity,
            pad_token_id=tokenizer.pad_token_id,
            microbatch_size=learner_microbatch_size,
            device=torch.device("cuda:0"),
            collection_sha256=collection_manifest["collection_sha256"],
            all_generated_action_tokens=collection_manifest["all_generated_action_tokens"],
            parity_thresholds=PARITY_THRESHOLDS,
        )
        elapsed = time.monotonic() - started
        report["learner_elapsed_seconds"] = elapsed
        report["optimizer_evaluated_action_tokens_per_second"] = (
            report["optimizer_evaluated_action_tokens"] / elapsed
        )
        artifact_audit = save_policy_update_artifacts(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            identity=identity,
            learner_report=report,
            output_adapter=Path(paths["output_adapter"]),
            output_rollout_adapter=Path(paths["output_rollout_adapter"]),
            output_optimizer=Path(paths["output_optimizer"]),
            base_model=base_model,
            input_adapter=input_adapter,
            input_optimizer=input_optimizer,
        )
        report.update(artifact_audit)
        committed = state_store.commit_update(
            identity=identity,
            learner_report=report,
        )
        return {"learner_report": report, "commit": committed}
    finally:
        if model is not None:
            owned_model, owned_optimizer, owned_tokenizer = model, optimizer, tokenizer
            model = optimizer = tokenizer = None
            _release_learner_memory(owned_model, owned_optimizer, owned_tokenizer)


async def run_online_preflight_once(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Run or resume target iteration zero; never advance a second iteration."""

    root = Path(prepared["root"])
    run_config = prepared["run_config"]
    state_store: IterationStore = prepared["state_store"]
    completion_path = root / "preflight_report.json"
    if completion_path.exists():
        report = json.loads(completion_path.read_text(encoding="utf-8"))
        _require(report.get("run_config_sha256") == run_config["run_config_sha256"], "preflight report/config drift")
        _require(report.get("complete") is True, "existing preflight report is incomplete")
        return report

    state_store.reconcile_committed_iterations()
    state = state_store.load_state()
    if state["current_iteration_index"] > TARGET_ITERATION_INDEX:
        _require(
            state["current_iteration_index"] == TARGET_ITERATION_INDEX + 1,
            "single-iteration preflight advanced beyond its target",
        )
        recovery_report = {
            "schema_version": ONLINE_PREFLIGHT_RECOVERY_REPORT_SCHEMA,
            "study_id": STUDY_ID,
            "formal_training": False,
            "run_config_sha256": run_config["run_config_sha256"],
            "result": "RECOVERED_COMMITTED_UPDATE_PHASE_GATE_FAILED",
            "recovered_after_committed_update": True,
            "current_iteration_index": state["current_iteration_index"],
            "current_policy_version": state["current_policy_version"],
            "current_adapter_sha256": state["current_adapter"]["sha256"],
            "current_rollout_adapter_sha256": state[
                "current_rollout_adapter"
            ]["sha256"],
            "current_adapter_semantic_sha256": state[
                "current_adapter_semantic_sha256"
            ],
            "last_iteration_manifest_sha256": state[
                "last_iteration_manifest_sha256"
            ],
            "same_gpu_phase_switch": {
                "passed": False,
                "reason": (
                    "iteration committed without a durable complete preflight report; "
                    "the prior process may have stopped before vLLM wake/add-adapter "
                    "or post-wake generation"
                ),
            },
            "complete": False,
        }
        atomic_write_json(root / "recovered_committed_update.json", recovery_report)
        raise RuntimeError(
            "committed update recovered, but same-GPU sleep/wake/generation was not proven; "
            "use a new preflight run root for the phase-switch gate"
        )

    identity = identity_from_run_state(state)
    state_store.assert_identity_matches_state(identity)
    collection_store = CollectionStore(
        root / "collections" / f"iteration-{identity.iteration_index:04d}",
        identity,
    )
    current_adapter = _resolve_state_artifact(state_store, state, "current_adapter")
    current_rollout_adapter = _resolve_state_artifact(
        state_store,
        state,
        "current_rollout_adapter",
    )
    engine_config = VLLMBackendConfig(
        base_model=run_config["base_model_path"],
        adapter_path=str(current_adapter),
        adapter_sha256=identity.input_adapter_sha256,
        rollout_adapter_path=str(current_rollout_adapter),
        rollout_adapter_sha256=identity.input_rollout_adapter_sha256,
        adapter_semantic_sha256=identity.input_adapter_semantic_sha256,
        seed=identity.seed,
    )
    _phase_event(
        root,
        "engine_starting",
        {
            "identity_sha256": identity.sha256,
            "adapter_sha256": identity.input_adapter_sha256,
            "rollout_adapter_sha256": identity.input_rollout_adapter_sha256,
            "adapter_semantic_sha256": identity.input_adapter_semantic_sha256,
        },
    )
    engine = await AsyncVLLMGenerationEngine.create(engine_config)
    engine_in_learner_phase = False
    collection_started = time.monotonic()
    try:
        loop = asyncio.get_running_loop()
        worker_pool = VLLMBrowserWorkerPool(
            identity=identity,
            engine=engine,
            event_loop=loop,
            event_loop_thread_id=threading.get_ident(),
            config=prepared["worker_config"],
        )
        _phase_event(
            root,
            "collection_started",
            {
                "browser_workers": run_config["browser_workers"],
                "maximum_tasks": run_config["maximum_tasks"],
            },
        )
        try:
            collection = await asyncio.to_thread(
                collect_iteration,
                store=collection_store,
                identity=identity,
                tasks=prepared["tasks"],
                initial_sampler_state=state["task_sampler_state"],
                group_runner=lambda group_id, task: worker_pool.run_group(
                    store=collection_store,
                    group_id=group_id,
                    task=task,
                ),
                global_generated_action_tokens_before=state["global_generated_action_tokens"],
                maximum_concurrent_groups=run_config["maximum_concurrent_k4_groups"],
                maximum_tasks=run_config["maximum_tasks"],
            )
        finally:
            await asyncio.to_thread(worker_pool.close)
        collection_elapsed = time.monotonic() - collection_started
        _require(collection["ready_for_update"] is True, "preflight collection produced no updateable group")
        groups = collection_store.load_committed_groups()
        performance = summarize_collection_performance(
            groups,
            elapsed_seconds=collection_elapsed,
        )
        _phase_event(root, "collection_frozen", performance)
        if run_config["collection_only"]:
            report = {
                "schema_version": ONLINE_PREFLIGHT_REPORT_SCHEMA,
                "study_id": STUDY_ID,
                "formal_training": False,
                "run_config_sha256": run_config["run_config_sha256"],
                "result": "COLLECTION_COMPLETE",
                "collection": collection,
                "collection_performance": performance,
                "learner": None,
                "complete": True,
            }
            atomic_write_json(completion_path, report)
            return report

        _phase_event(root, "vllm_sleep_started", {})
        sleep_result = await engine.switch_to_learner()
        engine_in_learner_phase = True
        _phase_event(root, "learner_started", sleep_result)
        update = await asyncio.to_thread(
            execute_learner_update,
            state_store=state_store,
            collection_store=collection_store,
            identity=identity,
            collection_manifest=collection["collection_manifest"],
            base_model=Path(run_config["base_model_path"]),
            learner_microbatch_size=run_config["learner_microbatch_size"],
        )
        new_state = update["commit"]["state"]
        next_adapter = _resolve_state_artifact(state_store, new_state, "current_adapter")
        next_rollout_adapter = _resolve_state_artifact(
            state_store,
            new_state,
            "current_rollout_adapter",
        )
        _phase_event(
            root,
            "learner_committed",
            {
                "output_policy_version": new_state["current_policy_version"],
                "output_adapter_sha256": new_state["current_adapter"]["sha256"],
                "output_rollout_adapter_sha256": new_state[
                    "current_rollout_adapter"
                ]["sha256"],
                "output_adapter_semantic_sha256": new_state[
                    "current_adapter_semantic_sha256"
                ],
            },
        )
        wake_result = await engine.switch_to_generation(
            adapter_path=str(next_adapter),
            adapter_sha256=new_state["current_adapter"]["sha256"],
            rollout_adapter_path=str(next_rollout_adapter),
            rollout_adapter_sha256=new_state["current_rollout_adapter"][
                "sha256"
            ],
            adapter_semantic_sha256=new_state[
                "current_adapter_semantic_sha256"
            ],
        )
        engine_in_learner_phase = False
        _phase_event(root, "vllm_wake_complete", wake_result)
        post_wake_result = await engine.generate_messages(
            [
                {
                    "role": "system",
                    "content": "Return one compact JSON object and no other text.",
                },
                {
                    "role": "user",
                    "content": 'Return {"status":"awake"}.',
                },
            ],
            request_id=f"post-wake-{new_state['current_policy_version']}",
            sampling_seed=identity.seed + 1_000_000,
        )
        post_wake_generation = audit_post_wake_generation(
            post_wake_result,
            expected_adapter_sha256=new_state["current_adapter"]["sha256"],
            expected_rollout_adapter_sha256=new_state["current_rollout_adapter"][
                "sha256"
            ],
            expected_adapter_semantic_sha256=new_state[
                "current_adapter_semantic_sha256"
            ],
        )
        _phase_event(root, "post_wake_generation_complete", post_wake_generation)
        report = {
            "schema_version": ONLINE_PREFLIGHT_REPORT_SCHEMA,
            "study_id": STUDY_ID,
            "formal_training": False,
            "run_config_sha256": run_config["run_config_sha256"],
            "result": "PASS",
            "collection": collection,
            "collection_performance": performance,
            "learner": update["learner_report"],
            "iteration_manifest": update["commit"]["manifest"],
            "same_gpu_phase_switch": {
                "sleep": sleep_result,
                "wake": wake_result,
                "post_wake_generation": post_wake_generation,
                "passed": True,
            },
            "complete": True,
        }
        atomic_write_json(completion_path, report)
        return report
    finally:
        # A failed learner leaves its stage for explicit archive/replay on resume.
        _phase_event(root, "engine_shutdown", {"learner_phase": engine_in_learner_phase})
        engine.shutdown()
