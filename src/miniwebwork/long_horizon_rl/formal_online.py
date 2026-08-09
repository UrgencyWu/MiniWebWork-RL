"""Recoverable multi-iteration formal online RL runner."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from ..m4_long_horizon_protocol import (
    ACTION_TOKEN_CAP,
    MAX_MODEL_TURNS,
    MAX_NEW_TOKENS,
    ONLINE_METHODS,
    ONLINE_SEEDS,
    PROJECT_ROOT,
    PROMPT_CONTRACT,
    PROMPT_PATH,
    STUDY_ID,
    assert_dataset_binding,
    load_study_manifest,
)
from .browser_pool import BrowserWorkerPoolConfig, VLLMBrowserWorkerPool
from .contracts import RunIdentity, atomic_write_json, directory_sha256, sha256_file, sha256_json
from .formal_contract import (
    assert_clean_tracked_worktree,
    current_git_sha,
    formal_output_path,
    load_formal_authorization,
    validate_readiness_for_submission,
)
from .formal_sft import FORMAL_SFT_ROOT, validate_formal_sft_manifest
from .iteration import IterationStore
from .journal import CollectionStore
from .model_manifest import validate_base_model_manifest
from .online_runner import (
    _initialize_or_load_state,
    _phase_event,
    _resolve_state_artifact,
    audit_post_wake_generation,
    execute_learner_update,
    identity_from_run_state,
    summarize_collection_performance,
)
from .orchestrator import (
    collect_iteration,
    load_frozen_train_roster,
    sampler_task_order_sha256,
)
from .rollout import admit_next_group
from .runtime_contract import PARITY_THRESHOLDS, load_online_runtime_contract
from .vllm_backend import AsyncVLLMGenerationEngine, VLLMBackendConfig

FORMAL_ONLINE_RUN_CONFIG_SCHEMA = "m4_long_horizon_formal_online_run_v1"
FORMAL_ONLINE_MANIFEST_SCHEMA = "m4_long_horizon_formal_online_manifest_v1"
FORMAL_POST_UPDATE_PROBE_SCHEMA = "m4_long_horizon_formal_post_update_probe_v1"
FORMAL_ONLINE_MANIFEST_NAME = "formal_online_manifest.json"
MAXIMUM_GROUP_TOKEN_RESERVE = 4 * MAX_MODEL_TURNS * MAX_NEW_TOKENS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON payload must be an object: {path}")
    return payload


def _content_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def expected_online_root(method: str, seed: int) -> Path:
    _require(method in ONLINE_METHODS, "unsupported formal online method")
    _require(seed in ONLINE_SEEDS, "formal online seed drift")
    return PROJECT_ROOT / "outputs" / STUDY_ID / "formal" / "online" / method / f"seed_{seed}"


def _load_or_write_config(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["run_config_sha256"] = sha256_json(normalized)
    path = root / "run_config.json"
    if path.is_file():
        existing = _json(path)
        _require(existing == normalized, "formal online run-config identity drift")
        return existing
    atomic_write_json(path, normalized)
    return normalized


def prepare_online_formal(
    *,
    output_dir: Path,
    method: str,
    seed: int,
    expected_git_sha: str,
    readiness_path: Path | None,
    shared_sft_root: Path = FORMAL_SFT_ROOT,
    base_model_manifest: Path,
    browser_workers: int = 32,
    maximum_tasks: int = 32,
    learner_microbatch_size: int = 8,
) -> dict[str, Any]:
    """Validate every upstream artifact before creating a formal run state."""

    _require(method in ONLINE_METHODS, "unsupported formal online method")
    _require(seed in ONLINE_SEEDS, "formal online seed drift")
    assert_clean_tracked_worktree()
    git_sha = current_git_sha()
    _require(git_sha == expected_git_sha, "formal online Git SHA drift")
    authorization = load_formal_authorization()
    readiness = validate_readiness_for_submission(readiness_path, expected_git_sha=git_sha)
    study = load_study_manifest()
    runtime = load_online_runtime_contract()
    dataset = assert_dataset_binding(study["payload"])
    shared = validate_formal_sft_manifest(shared_sft_root)
    _require(shared["payload"]["git_sha"] == git_sha, "shared SFT/formal online Git drift")
    _require(shared["payload"]["authorization_sha256"] == authorization["sha256"], "shared SFT authorization drift")
    _require(shared["payload"]["readiness_sha256"] == readiness["sha256"], "shared SFT readiness drift")
    expected_root = expected_online_root(method, seed).resolve()
    root = formal_output_path(output_dir)
    _require(root == expected_root, f"formal online run must use frozen root: {expected_root}")
    worker_config = BrowserWorkerPoolConfig(total_workers=browser_workers)
    worker_config.validate()
    _require(browser_workers == 32, "formal online runner requires the selected 32 browser workers")
    _require(worker_config.concurrent_group_slots == 8, "formal online K4 slot selection drift")
    _require(maximum_tasks == 32, "formal online iterations require the selected 32-task roster slice")
    _require(learner_microbatch_size == 8, "formal online learner microbatch must be 8")
    model = validate_base_model_manifest(base_model_manifest, verify_files=True)
    base_model = Path(model["payload"]["base_model_path"])
    _require(str(base_model) == runtime["payload"]["model_contract"]["base_model_path"], "formal online base model/runtime drift")
    shared_root = Path(shared_sft_root).resolve()
    initial_adapter = shared_root / shared["payload"]["canonical_adapter"]["relative_path"]
    initial_rollout_adapter = shared_root / shared["payload"]["rollout_adapter"]["relative_path"]
    _require(directory_sha256(initial_adapter) == shared["payload"]["canonical_adapter"]["directory_sha256"], "shared SFT adapter drift before online")
    _require(directory_sha256(initial_rollout_adapter) == shared["payload"]["rollout_adapter"]["directory_sha256"], "shared SFT rollout adapter drift before online")
    tasks = load_frozen_train_roster()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FORMAL_ONLINE_RUN_CONFIG_SCHEMA,
        "study_id": STUDY_ID,
        "formal_training": True,
        "method": method,
        "seed": seed,
        "git_sha": git_sha,
        "authorization_sha256": authorization["sha256"],
        "readiness_sha256": readiness["sha256"],
        "shared_sft_manifest_sha256": shared["sha256"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "seed_manifest_sha256": dataset["seed_manifest_sha256"],
        "prompt_contract": PROMPT_CONTRACT,
        "prompt_sha256": sha256_file(PROMPT_PATH),
        "credit_formula_version": study["payload"]["credit_assignment_contract"]["formula_version"],
        "task_order_sha256": sampler_task_order_sha256(tasks, study_seed=seed),
        "base_model_path": str(base_model),
        "base_model_manifest_path": model["path"],
        "base_model_manifest_sha256": model["sha256"],
        "runtime_contract_sha256": runtime["sha256"],
        "initial_adapter_path": str(initial_adapter),
        "initial_adapter_sha256": shared["payload"]["canonical_adapter"]["directory_sha256"],
        "initial_rollout_adapter_path": str(initial_rollout_adapter),
        "initial_rollout_adapter_sha256": shared["payload"]["rollout_adapter"]["directory_sha256"],
        "initial_adapter_semantic_sha256": shared["payload"]["rollout_adapter"]["semantic_tensor_sha256"],
        "browser_workers": browser_workers,
        "maximum_concurrent_k4_groups": worker_config.concurrent_group_slots,
        "maximum_tasks_per_iteration": maximum_tasks,
        "learner_microbatch_size": learner_microbatch_size,
        "action_token_cap": ACTION_TOKEN_CAP,
        "maximum_group_token_reserve": MAXIMUM_GROUP_TOKEN_RESERVE,
        "parity_thresholds": dict(PARITY_THRESHOLDS),
        "recovery": "same_root_resume_collection_replay_learner_reconcile_commit_and_probe",
        "output_root": str(root),
    }
    run_config = _load_or_write_config(root, payload)
    state_store, state = _initialize_or_load_state(
        root=root,
        run_config=run_config,
        initial_adapter=initial_adapter,
        initial_rollout_adapter=initial_rollout_adapter,
        initial_adapter_semantic_sha256=shared["payload"]["rollout_adapter"]["semantic_tensor_sha256"],
    )
    return {
        "root": root,
        "run_config": run_config,
        "state_store": state_store,
        "state": state,
        "tasks": tasks,
        "worker_config": worker_config,
    }


def _proof_path(root: Path, iteration_index: int) -> Path:
    return root / "post_update_generation" / f"iteration-{iteration_index:04d}.json"


def _validate_probe(payload: Mapping[str, Any], *, iteration_index: int, state: Mapping[str, Any]) -> dict[str, Any]:
    proof = dict(payload)
    _require(proof.get("schema_version") == FORMAL_POST_UPDATE_PROBE_SCHEMA, "formal post-update probe schema drift")
    _require(proof.get("iteration_index") == iteration_index, "formal post-update probe iteration drift")
    _require(proof.get("passed") is True, "formal post-update probe failed")
    _require(proof.get("proof_mode") in {"same_process_sleep_wake", "process_restart_recovery"}, "formal post-update probe mode drift")
    _require(proof.get("output_policy_version") == state["current_policy_version"], "formal post-update policy drift")
    _require(proof.get("adapter_sha256") == state["current_adapter"]["sha256"], "formal post-update adapter drift")
    _require(proof.get("rollout_adapter_sha256") == state["current_rollout_adapter"]["sha256"], "formal post-update rollout adapter drift")
    _require(proof.get("adapter_semantic_sha256") == state["current_adapter_semantic_sha256"], "formal post-update semantic drift")
    _require(proof.get("proof_content_sha256") == _content_sha(proof, "proof_content_sha256"), "formal post-update probe self-hash drift")
    return proof


async def _write_generation_probe(
    *,
    root: Path,
    engine: AsyncVLLMGenerationEngine,
    state: Mapping[str, Any],
    iteration_index: int,
    seed: int,
    proof_mode: str,
    sleep_result: Mapping[str, Any] | None = None,
    wake_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = _proof_path(root, iteration_index)
    if path.is_file():
        return _validate_probe(_json(path), iteration_index=iteration_index, state=state)
    result = await engine.generate_messages(
        [
            {"role": "system", "content": "Return one compact JSON object and no other text."},
            {"role": "user", "content": '{"request":"confirm updated policy is generating"}'},
        ],
        request_id=f"formal-post-update-{iteration_index:04d}-{proof_mode}",
        sampling_seed=seed + 1_000_000 + iteration_index,
    )
    audit = audit_post_wake_generation(
        result,
        expected_adapter_sha256=state["current_adapter"]["sha256"],
        expected_rollout_adapter_sha256=state["current_rollout_adapter"]["sha256"],
        expected_adapter_semantic_sha256=state["current_adapter_semantic_sha256"],
        cost_scope="formal_recovery_probe_excluded_from_training_action_token_budget",
    )
    proof = {
        "schema_version": FORMAL_POST_UPDATE_PROBE_SCHEMA,
        "study_id": STUDY_ID,
        "formal_training": True,
        "iteration_index": iteration_index,
        "output_policy_version": state["current_policy_version"],
        "proof_mode": proof_mode,
        "same_process_sleep_wake_proven": proof_mode == "same_process_sleep_wake",
        "sleep": dict(sleep_result) if sleep_result is not None else None,
        "wake": dict(wake_result) if wake_result is not None else None,
        **audit,
    }
    proof["proof_content_sha256"] = _content_sha(proof, "proof_content_sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, proof)
    return _validate_probe(proof, iteration_index=iteration_index, state=state)


async def _run_one_iteration(prepared: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(prepared["root"])
    config = prepared["run_config"]
    store: IterationStore = prepared["state_store"]
    store.reconcile_committed_iterations()
    state = store.load_state()
    identity = identity_from_run_state(state)
    store.assert_identity_matches_state(identity)
    current_adapter = _resolve_state_artifact(store, state, "current_adapter")
    current_rollout_adapter = _resolve_state_artifact(store, state, "current_rollout_adapter")
    engine = await AsyncVLLMGenerationEngine.create(
        VLLMBackendConfig(
            base_model=config["base_model_path"],
            adapter_path=str(current_adapter),
            adapter_sha256=identity.input_adapter_sha256,
            rollout_adapter_path=str(current_rollout_adapter),
            rollout_adapter_sha256=identity.input_rollout_adapter_sha256,
            adapter_semantic_sha256=identity.input_adapter_semantic_sha256,
            seed=identity.seed,
        )
    )
    learner_phase = False
    try:
        # A committed iteration may outlive the process that was about to wake
        # vLLM.  The new engine already loaded that committed adapter; prove it
        # generates before collecting another training token.
        if state["current_iteration_index"] > 0:
            prior_index = state["current_iteration_index"] - 1
            prior_path = _proof_path(root, prior_index)
            if not prior_path.is_file():
                proof = await _write_generation_probe(
                    root=root,
                    engine=engine,
                    state=state,
                    iteration_index=prior_index,
                    seed=config["seed"],
                    proof_mode="process_restart_recovery",
                )
                _phase_event(root, "post_restart_generation_recovered", proof)
            else:
                _validate_probe(_json(prior_path), iteration_index=prior_index, state=state)

        collection_store = CollectionStore(
            root / "collections" / f"iteration-{identity.iteration_index:04d}",
            identity,
        )
        loop = asyncio.get_running_loop()
        pool = VLLMBrowserWorkerPool(
            identity=identity,
            engine=engine,
            event_loop=loop,
            event_loop_thread_id=threading.get_ident(),
            config=prepared["worker_config"],
        )
        _phase_event(root, "formal_collection_started", {"iteration_index": identity.iteration_index, "browser_workers": config["browser_workers"]})
        started = time.monotonic()
        try:
            collection = await asyncio.to_thread(
                collect_iteration,
                store=collection_store,
                identity=identity,
                tasks=prepared["tasks"],
                initial_sampler_state=state["task_sampler_state"],
                group_runner=lambda group_id, task: pool.run_group(store=collection_store, group_id=group_id, task=task),
                global_generated_action_tokens_before=state["global_generated_action_tokens"],
                maximum_concurrent_groups=config["maximum_concurrent_k4_groups"],
                maximum_tasks=config["maximum_tasks_per_iteration"],
                token_cap=config["action_token_cap"],
            )
        finally:
            await asyncio.to_thread(pool.close)
        elapsed = time.monotonic() - started
        if collection["ready_for_update"] is False:
            _require(collection["stopped_for_token_budget"] is True, "formal collection ended without update or budget stop")
            terminal = collection.get("collection_manifest")
            _require(terminal is not None and terminal["group_count"] == 0, "formal terminal collection drift")
            _phase_event(root, "formal_terminal_cost_frozen", {"iteration_index": identity.iteration_index, "all_generated_action_tokens": terminal["all_generated_action_tokens"], "collection_sha256": terminal["collection_sha256"]})
            return {"terminal_collection": terminal, "iteration_committed": False}
        groups = collection_store.load_committed_groups()
        performance = summarize_collection_performance(groups, elapsed_seconds=elapsed)
        _phase_event(root, "formal_collection_frozen", {"iteration_index": identity.iteration_index, **performance})
        sleep_result = await engine.switch_to_learner()
        learner_phase = True
        _phase_event(root, "formal_learner_started", {"iteration_index": identity.iteration_index, **sleep_result})
        update = await asyncio.to_thread(
            execute_learner_update,
            state_store=store,
            collection_store=collection_store,
            identity=identity,
            collection_manifest=collection["collection_manifest"],
            base_model=Path(config["base_model_path"]),
            learner_microbatch_size=config["learner_microbatch_size"],
        )
        new_state = update["commit"]["state"]
        next_adapter = _resolve_state_artifact(store, new_state, "current_adapter")
        next_rollout_adapter = _resolve_state_artifact(store, new_state, "current_rollout_adapter")
        _phase_event(root, "formal_learner_committed", {"iteration_index": identity.iteration_index, "output_policy_version": new_state["current_policy_version"], "global_generated_action_tokens": new_state["global_generated_action_tokens"]})
        wake_result = await engine.switch_to_generation(
            adapter_path=str(next_adapter),
            adapter_sha256=new_state["current_adapter"]["sha256"],
            rollout_adapter_path=str(next_rollout_adapter),
            rollout_adapter_sha256=new_state["current_rollout_adapter"]["sha256"],
            adapter_semantic_sha256=new_state["current_adapter_semantic_sha256"],
        )
        learner_phase = False
        proof = await _write_generation_probe(
            root=root,
            engine=engine,
            state=new_state,
            iteration_index=identity.iteration_index,
            seed=config["seed"],
            proof_mode="same_process_sleep_wake",
            sleep_result=sleep_result,
            wake_result=wake_result,
        )
        _phase_event(root, "formal_post_update_generation_complete", proof)
        return {
            "terminal_collection": None,
            "iteration_committed": True,
            "collection": collection,
            "collection_performance": performance,
            "learner": update["learner_report"],
            "iteration_manifest": update["commit"]["manifest"],
            "proof": proof,
        }
    finally:
        _phase_event(root, "formal_engine_shutdown", {"learner_phase": learner_phase})
        engine.shutdown()


async def _recover_latest_committed_probe(prepared: Mapping[str, Any]) -> dict[str, Any] | None:
    """Prove the latest committed policy after a restart without collecting tokens."""

    root = Path(prepared["root"])
    config = prepared["run_config"]
    store: IterationStore = prepared["state_store"]
    store.reconcile_committed_iterations()
    state = store.load_state()
    if state["current_iteration_index"] == 0:
        return None
    iteration_index = state["current_iteration_index"] - 1
    path = _proof_path(root, iteration_index)
    if path.is_file():
        return _validate_probe(_json(path), iteration_index=iteration_index, state=state)
    identity = identity_from_run_state(state)
    current_adapter = _resolve_state_artifact(store, state, "current_adapter")
    current_rollout_adapter = _resolve_state_artifact(store, state, "current_rollout_adapter")
    engine = await AsyncVLLMGenerationEngine.create(
        VLLMBackendConfig(
            base_model=config["base_model_path"],
            adapter_path=str(current_adapter),
            adapter_sha256=identity.input_adapter_sha256,
            rollout_adapter_path=str(current_rollout_adapter),
            rollout_adapter_sha256=identity.input_rollout_adapter_sha256,
            adapter_semantic_sha256=identity.input_adapter_semantic_sha256,
            seed=identity.seed,
        )
    )
    try:
        proof = await _write_generation_probe(
            root=root,
            engine=engine,
            state=state,
            iteration_index=iteration_index,
            seed=config["seed"],
            proof_mode="process_restart_recovery",
        )
        _phase_event(root, "post_restart_generation_recovered", proof)
        return proof
    finally:
        _phase_event(root, "formal_recovery_engine_shutdown", {"iteration_index": iteration_index})
        engine.shutdown()


def _terminal_collection_descriptor(root: Path, result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not result or result.get("terminal_collection") is None:
        return None
    manifest = result["terminal_collection"]
    index = manifest["iteration_index"]
    path = root / "collections" / f"iteration-{index:04d}" / "collection_manifest.json"
    _require(path.is_file() and sha256_file(path), "formal terminal collection file is missing")
    return {
        "relative_path": str(path.relative_to(root)),
        "file_sha256": sha256_file(path),
        "collection_sha256": manifest["collection_sha256"],
        "all_generated_action_tokens": manifest["all_generated_action_tokens"],
        "group_count": manifest["group_count"],
    }


def finalize_formal_online(root: Path, terminal_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    config = _json(root / "run_config.json")
    store = IterationStore(root / "state")
    state = store.load_state()
    manifests = store.load_committed_iteration_manifests()
    terminal = _terminal_collection_descriptor(root, terminal_result)
    terminal_tokens = terminal["all_generated_action_tokens"] if terminal else 0
    accounted_tokens = state["global_generated_action_tokens"] + terminal_tokens
    admission = admit_next_group(
        global_tokens_before_iteration=accounted_tokens,
        current_iteration_tokens=0,
        token_cap=config["action_token_cap"],
    )
    _require(admission.allowed is False, "formal online finalized while another complete K4 group was admissible")
    _require(accounted_tokens <= ACTION_TOKEN_CAP, "formal online exceeded its action-token cap")
    _require(len(manifests) >= 2, "formal online did not exercise multiple policy iterations")
    summaries = []
    optimizer_updates = 0
    effective_tokens = 0
    group_count = 0
    zero_advantage_groups = 0
    learner_elapsed_seconds = 0.0
    committed_tokens = 0
    invalidated_tokens = 0
    for manifest in manifests:
        index = manifest["iteration_index"]
        iteration_root = root / "state" / "iterations" / f"iteration-{index:04d}"
        learner = _json(iteration_root / "learner_report.json")
        collection = _json(iteration_root / "collection_manifest.json")
        proof_path = _proof_path(root, index)
        proof = _validate_probe(_json(proof_path), iteration_index=index, state={
            "current_policy_version": manifest["output_policy_version"],
            "current_adapter": {"sha256": manifest["output_adapter"]["sha256"]},
            "current_rollout_adapter": {"sha256": manifest["output_rollout_adapter"]["sha256"]},
            "current_adapter_semantic_sha256": manifest["output_adapter_semantic_sha256"],
        })
        optimizer_updates += learner["optimizer_updates"]
        effective_tokens += learner["effective_optimizer_action_tokens"]
        group_count += manifest["collection_group_count"]
        zero_advantage_groups += learner["zero_advantage_group_count"]
        learner_elapsed_seconds += learner["learner_elapsed_seconds"]
        committed_tokens += collection["committed_group_action_tokens"]
        invalidated_tokens += collection["all_generated_action_tokens"] - collection["committed_group_action_tokens"]
        summaries.append({
            "iteration_index": index,
            "iteration_manifest_sha256": manifest["iteration_manifest_sha256"],
            "collection_sha256": manifest["collection_sha256"],
            "group_count": manifest["collection_group_count"],
            "generated_action_tokens": manifest["iteration_generated_action_tokens"],
            "optimizer_updates": learner["optimizer_updates"],
            "effective_optimizer_action_tokens": learner["effective_optimizer_action_tokens"],
            "zero_advantage_group_count": learner["zero_advantage_group_count"],
            "committed_group_action_tokens": collection["committed_group_action_tokens"],
            "invalidated_action_tokens": collection["all_generated_action_tokens"] - collection["committed_group_action_tokens"],
            "learner_elapsed_seconds": learner["learner_elapsed_seconds"],
            "mean_ratio": learner["mean_ratio"],
            "clip_fraction": learner["clip_fraction"],
            "approx_kl": learner["approx_kl"],
            "entropy": learner["entropy"],
            "parameter_change_norm": learner["parameter_change_norm"],
            "credit_assignment": learner["credit_assignment"],
            "post_update_probe_sha256": sha256_file(proof_path),
            "post_update_probe_mode": proof["proof_mode"],
        })
    _require(optimizer_updates > 0, "formal online completed without any optimizer update")
    manifest = {
        "schema_version": FORMAL_ONLINE_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "formal_training": True,
        "complete": True,
        "method": config["method"],
        "seed": config["seed"],
        "git_sha": config["git_sha"],
        "authorization_sha256": config["authorization_sha256"],
        "readiness_sha256": config["readiness_sha256"],
        "shared_sft_manifest_sha256": config["shared_sft_manifest_sha256"],
        "run_config_sha256": sha256_file(root / "run_config.json"),
        "stop_reason": "insufficient_worst_case_complete_k4_group_reserve",
        "action_token_budget": {
            "cap": ACTION_TOKEN_CAP,
            "committed_iteration_tokens": state["global_generated_action_tokens"],
            "terminal_invalid_attempt_tokens": terminal_tokens,
            "accounted_generated_action_tokens": accounted_tokens,
            "unused_tokens": ACTION_TOKEN_CAP - accounted_tokens,
            "maximum_group_reserve": MAXIMUM_GROUP_TOKEN_RESERVE,
        },
        "iteration_count": len(manifests),
        "group_count": group_count,
        "optimizer_updates": optimizer_updates,
        "effective_optimizer_action_tokens": effective_tokens,
        "effective_optimizer_action_token_fraction": effective_tokens / accounted_tokens,
        "optimizer_action_token_fraction_target_diagnostic": 0.2,
        "zero_advantage_group_count": zero_advantage_groups,
        "training_cost": {
            "committed_group_action_tokens": committed_tokens,
            "invalidated_action_tokens": invalidated_tokens + terminal_tokens,
            "learner_elapsed_seconds": learner_elapsed_seconds,
            "slurm_accounting_deferred_to_final_analysis": True,
        },
        "iterations": summaries,
        "terminal_collection": terminal,
        "final_policy": {
            "policy_version": state["current_policy_version"],
            "canonical_adapter_relative_path": str((root / "state" / state["current_adapter"]["path"]).resolve().relative_to(root)),
            "canonical_adapter_sha256": state["current_adapter"]["sha256"],
            "rollout_adapter_relative_path": str((root / "state" / state["current_rollout_adapter"]["path"]).resolve().relative_to(root)),
            "rollout_adapter_sha256": state["current_rollout_adapter"]["sha256"],
            "adapter_semantic_sha256": state["current_adapter_semantic_sha256"],
            "optimizer_relative_path": str((root / "state" / state["current_optimizer"]["path"]).resolve().relative_to(root)),
            "optimizer_sha256": state["current_optimizer"]["sha256"],
        },
        "state": {
            "relative_path": "state/run_state.json",
            "file_sha256": sha256_file(root / "state" / "run_state.json"),
            "state_sha256": state["state_sha256"],
            "iteration_ledger_sha256": sha256_file(root / "state" / "iteration_ledger.jsonl"),
            "task_sampler_state_sha256": sha256_json(state["task_sampler_state"]),
        },
    }
    manifest["manifest_content_sha256"] = _content_sha(manifest, "manifest_content_sha256")
    atomic_write_json(root / FORMAL_ONLINE_MANIFEST_NAME, manifest)
    return validate_formal_online_manifest(root)


def validate_formal_online_manifest(root: Path) -> dict[str, Any]:
    root = formal_output_path(root)
    path = root / FORMAL_ONLINE_MANIFEST_NAME
    _require(path.is_file(), "formal online manifest is missing")
    payload = _json(path)
    _require(payload.get("schema_version") == FORMAL_ONLINE_MANIFEST_SCHEMA, "formal online manifest schema drift")
    _require(payload.get("formal_training") is True and payload.get("complete") is True, "formal online run is incomplete")
    method, seed = payload.get("method"), payload.get("seed")
    _require(root == expected_online_root(method, seed).resolve(), "formal online root/identity drift")
    _require(payload.get("manifest_content_sha256") == _content_sha(payload, "manifest_content_sha256"), "formal online manifest self-hash drift")
    config_path = root / "run_config.json"
    config = _json(config_path)
    _require(payload.get("run_config_sha256") == sha256_file(config_path), "formal online run-config hash drift")
    for field in ("method", "seed", "git_sha", "authorization_sha256", "readiness_sha256", "shared_sft_manifest_sha256"):
        _require(payload.get(field) == config.get(field), f"formal online {field} drift")
    budget = payload.get("action_token_budget", {})
    _require(budget.get("cap") == ACTION_TOKEN_CAP, "formal online token cap drift")
    _require(budget.get("accounted_generated_action_tokens") == budget.get("committed_iteration_tokens") + budget.get("terminal_invalid_attempt_tokens"), "formal online token accounting drift")
    _require(0 <= budget.get("unused_tokens", -1) < MAXIMUM_GROUP_TOKEN_RESERVE, "formal online stopped outside the full-group reserve boundary")
    _require(budget["accounted_generated_action_tokens"] + budget["unused_tokens"] == ACTION_TOKEN_CAP, "formal online unused-token ledger drift")
    _require(payload.get("iteration_count", 0) >= 2, "formal online is not multi-iteration")
    _require(payload.get("optimizer_updates", 0) > 0, "formal online has no optimizer updates")
    _require(len(payload.get("iterations", [])) == payload["iteration_count"], "formal online iteration summary count drift")
    cost = payload.get("training_cost", {})
    _require(cost.get("committed_group_action_tokens", -1) + cost.get("invalidated_action_tokens", -1) == budget["accounted_generated_action_tokens"], "formal online training-cost token ledger drift")
    _require(cost.get("learner_elapsed_seconds", 0) > 0, "formal online learner-time ledger is empty")
    store = IterationStore(root / "state")
    state = store.load_state()
    manifests = store.load_committed_iteration_manifests()
    _require(len(manifests) == payload["iteration_count"] == state["current_iteration_index"], "formal online state/iteration count drift")
    _require(state["global_generated_action_tokens"] == budget["committed_iteration_tokens"], "formal online state token drift")
    final = payload["final_policy"]
    adapter = root / final["canonical_adapter_relative_path"]
    view = root / final["rollout_adapter_relative_path"]
    optimizer = root / final["optimizer_relative_path"]
    _require(directory_sha256(adapter) == final["canonical_adapter_sha256"] == state["current_adapter"]["sha256"], "formal online final adapter drift")
    _require(directory_sha256(view) == final["rollout_adapter_sha256"] == state["current_rollout_adapter"]["sha256"], "formal online final rollout adapter drift")
    _require(sha256_file(optimizer) == final["optimizer_sha256"] == state["current_optimizer"]["sha256"], "formal online final optimizer drift")
    for summary, iteration in zip(payload["iterations"], manifests):
        _require(summary["iteration_manifest_sha256"] == iteration["iteration_manifest_sha256"], "formal online iteration hash drift")
        proof_path = _proof_path(root, iteration["iteration_index"])
        _require(sha256_file(proof_path) == summary["post_update_probe_sha256"], "formal online post-update proof file drift")
        proof = _validate_probe(_json(proof_path), iteration_index=iteration["iteration_index"], state={
            "current_policy_version": iteration["output_policy_version"],
            "current_adapter": {"sha256": iteration["output_adapter"]["sha256"]},
            "current_rollout_adapter": {"sha256": iteration["output_rollout_adapter"]["sha256"]},
            "current_adapter_semantic_sha256": iteration["output_adapter_semantic_sha256"],
        })
        _require(summary["post_update_probe_mode"] == proof["proof_mode"], "formal online post-update proof mode drift")
        iteration_root = root / "state" / "iterations" / f"iteration-{iteration['iteration_index']:04d}"
        learner = _json(iteration_root / "learner_report.json")
        collection = _json(iteration_root / "collection_manifest.json")
        identity = RunIdentity.from_mapping(iteration["identity"])
        original_store = CollectionStore(root / "collections" / f"iteration-{iteration['iteration_index']:04d}", identity)
        original_groups = original_store.load_committed_groups()
        _require(len(original_groups) == iteration["collection_group_count"], "formal online raw training group count drift")
        original_collection = original_store.freeze_collection(
            iteration_index=iteration["iteration_index"],
            stopped_for_token_budget=collection["stopped_for_token_budget"],
            task_sampler_state=collection["task_sampler_state"],
        )
        _require(original_collection["collection_sha256"] == iteration["collection_sha256"], "formal online raw training collection drift")
        expected_summary = {
            "group_count": iteration["collection_group_count"],
            "generated_action_tokens": iteration["iteration_generated_action_tokens"],
            "optimizer_updates": learner["optimizer_updates"],
            "effective_optimizer_action_tokens": learner["effective_optimizer_action_tokens"],
            "zero_advantage_group_count": learner["zero_advantage_group_count"],
            "committed_group_action_tokens": collection["committed_group_action_tokens"],
            "invalidated_action_tokens": collection["all_generated_action_tokens"] - collection["committed_group_action_tokens"],
            "learner_elapsed_seconds": learner["learner_elapsed_seconds"],
            "mean_ratio": learner["mean_ratio"],
            "clip_fraction": learner["clip_fraction"],
            "approx_kl": learner["approx_kl"],
            "entropy": learner["entropy"],
            "parameter_change_norm": learner["parameter_change_norm"],
            "credit_assignment": learner["credit_assignment"],
        }
        for field, expected in expected_summary.items():
            _require(summary.get(field) == expected, f"formal online iteration summary drift: {field}")
    terminal = payload.get("terminal_collection")
    if terminal is not None:
        terminal_path = root / terminal["relative_path"]
        terminal_payload = _json(terminal_path)
        _require(sha256_file(terminal_path) == terminal["file_sha256"], "formal terminal collection file drift")
        _require(terminal_payload["collection_sha256"] == terminal["collection_sha256"], "formal terminal collection content drift")
        _require(terminal_payload["group_count"] == 0 and terminal_payload["stopped_for_token_budget"] is True, "formal terminal collection semantics drift")
        terminal_identity = RunIdentity.from_mapping(_json(terminal_path.parent / "run_identity.json"))
        terminal_store = CollectionStore(terminal_path.parent, terminal_identity)
        recomputed_terminal = terminal_store.freeze_collection(
            iteration_index=terminal_payload["iteration_index"],
            stopped_for_token_budget=True,
            task_sampler_state=terminal_payload["task_sampler_state"],
        )
        _require(recomputed_terminal["collection_sha256"] == terminal["collection_sha256"], "formal terminal journal/manifest drift")
    return {"path": str(path), "sha256": sha256_file(path), "payload": payload}


async def run_online_formal(prepared: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(prepared["root"])
    completion = root / FORMAL_ONLINE_MANIFEST_NAME
    if completion.is_file():
        return validate_formal_online_manifest(root)
    terminal_result = None
    while True:
        store: IterationStore = prepared["state_store"]
        store.reconcile_committed_iterations()
        state = store.load_state()
        admission = admit_next_group(
            global_tokens_before_iteration=state["global_generated_action_tokens"],
            current_iteration_tokens=0,
            token_cap=prepared["run_config"]["action_token_cap"],
        )
        if not admission.allowed:
            break
        result = await _run_one_iteration(prepared)
        if not result["iteration_committed"]:
            terminal_result = result
            break
    # The final learner commit can survive a 24-hour Slurm termination while
    # the same-process wake/probe does not.  Closing the token budget must not
    # bypass that lineage proof merely because no further K4 group is allowed.
    await _recover_latest_committed_probe(prepared)
    return finalize_formal_online(root, terminal_result)
