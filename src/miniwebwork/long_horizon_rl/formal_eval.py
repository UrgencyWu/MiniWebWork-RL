"""Frozen 120-task K=4 evaluation with durable task-group resume."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from ..m4_long_horizon_protocol import (
    CREDIT_FORMULA_VERSION,
    DATASET_ROOT,
    FORMAL_METHODS,
    GROUP_SIZE,
    ONLINE_METHODS,
    ONLINE_SEEDS,
    PROJECT_ROOT,
    PROMPT_CONTRACT,
    PROMPT_PATH,
    SEED_ROOT,
    SFT_SEED,
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
from .formal_online import expected_online_root, validate_formal_online_manifest
from .formal_sft import FORMAL_SFT_ROOT, validate_formal_sft_manifest
from .journal import CollectionStore
from .model_manifest import validate_base_model_manifest
from .orchestrator import load_public_task_roster
from .runtime_contract import load_online_runtime_contract
from .vllm_backend import AsyncVLLMGenerationEngine, VLLMBackendConfig

FORMAL_EVAL_RUN_CONFIG_SCHEMA = "m4_long_horizon_formal_eval_run_v1"
FORMAL_EVAL_MANIFEST_SCHEMA = "m4_long_horizon_formal_eval_manifest_v1"
FORMAL_EVAL_MANIFEST_NAME = "formal_eval_manifest.json"
TEST_PUBLIC_PATH = DATASET_ROOT / "test" / "test_public.jsonl"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON payload must be an object: {path}")
    return value


def _content_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def expected_eval_root(method: str, seed: int) -> Path:
    _require(method in FORMAL_METHODS, "unsupported formal evaluation method")
    if method == "verified_sft":
        _require(seed == SFT_SEED, "shared SFT evaluation seed drift")
    else:
        _require(seed in ONLINE_SEEDS, "online evaluation seed drift")
    return PROJECT_ROOT / "outputs" / STUDY_ID / "formal" / "frozen_test" / method / f"seed_{seed}"


def _all_training_lineage() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    shared = validate_formal_sft_manifest(FORMAL_SFT_ROOT)
    online = {
        (method, seed): validate_formal_online_manifest(expected_online_root(method, seed))
        for method in ONLINE_METHODS
        for seed in ONLINE_SEEDS
    }
    _require(len(online) == 6, "formal evaluation requires all six online runs")
    return shared, online


def _resolve_policy(
    method: str,
    seed: int,
    shared: Mapping[str, Any],
    online: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    if method == "verified_sft":
        payload = shared["payload"]
        root = FORMAL_SFT_ROOT.resolve()
        return {
            "training_manifest_path": shared["path"],
            "training_manifest_sha256": shared["sha256"],
            "policy_version": "policy_0000",
            "iteration_index": 0,
            "adapter": root / payload["canonical_adapter"]["relative_path"],
            "adapter_sha256": payload["canonical_adapter"]["directory_sha256"],
            "rollout_adapter": root / payload["rollout_adapter"]["relative_path"],
            "rollout_adapter_sha256": payload["rollout_adapter"]["directory_sha256"],
            "semantic_sha256": payload["rollout_adapter"]["semantic_tensor_sha256"],
        }
    result = online[(method, seed)]
    payload = result["payload"]
    root = expected_online_root(method, seed).resolve()
    final = payload["final_policy"]
    return {
        "training_manifest_path": result["path"],
        "training_manifest_sha256": result["sha256"],
        "policy_version": final["policy_version"],
        "iteration_index": payload["iteration_count"],
        "adapter": root / final["canonical_adapter_relative_path"],
        "adapter_sha256": final["canonical_adapter_sha256"],
        "rollout_adapter": root / final["rollout_adapter_relative_path"],
        "rollout_adapter_sha256": final["rollout_adapter_sha256"],
        "semantic_sha256": final["adapter_semantic_sha256"],
    }


def prepare_formal_eval(
    *,
    method: str,
    seed: int,
    output_dir: Path,
    expected_git_sha: str,
    readiness_path: Path | None,
    base_model_manifest: Path,
) -> dict[str, Any]:
    _require(method in FORMAL_METHODS, "unsupported formal evaluation method")
    assert_clean_tracked_worktree()
    git_sha = current_git_sha()
    _require(git_sha == expected_git_sha, "formal evaluation Git SHA drift")
    authorization = load_formal_authorization()
    readiness = validate_readiness_for_submission(readiness_path, expected_git_sha=git_sha)
    study = load_study_manifest()
    runtime = load_online_runtime_contract()
    dataset = assert_dataset_binding(study["payload"])
    shared, online = _all_training_lineage()
    lineage_artifacts = [shared, *online.values()]
    for artifact in lineage_artifacts:
        _require(artifact["payload"]["git_sha"] == git_sha, "evaluation training/Git lineage drift")
        _require(artifact["payload"]["authorization_sha256"] == authorization["sha256"], "evaluation training/authorization lineage drift")
        _require(artifact["payload"]["readiness_sha256"] == readiness["sha256"], "evaluation training/readiness lineage drift")
    lineage = _resolve_policy(method, seed, shared, online)
    _require(directory_sha256(lineage["adapter"]) == lineage["adapter_sha256"], "evaluation canonical adapter drift")
    _require(directory_sha256(lineage["rollout_adapter"]) == lineage["rollout_adapter_sha256"], "evaluation rollout adapter drift")
    root = formal_output_path(output_dir)
    _require(root == expected_eval_root(method, seed).resolve(), "formal evaluation root drift")
    model = validate_base_model_manifest(base_model_manifest, verify_files=True)
    base_model = Path(model["payload"]["base_model_path"])
    _require(str(base_model) == runtime["payload"]["model_contract"]["base_model_path"], "evaluation base model/runtime drift")
    tasks = load_public_task_roster(TEST_PUBLIC_PATH, expected_count=120)
    task_order_sha = sha256_json([task.task_id for task in tasks])
    identity = RunIdentity(
        study_id=STUDY_ID,
        git_sha=git_sha,
        method=method,
        seed=seed,
        iteration_index=lineage["iteration_index"],
        policy_version=lineage["policy_version"],
        dataset_manifest_sha256=dataset["dataset_manifest_sha256"],
        seed_manifest_sha256=dataset["seed_manifest_sha256"],
        prompt_contract=PROMPT_CONTRACT,
        prompt_sha256=sha256_file(PROMPT_PATH),
        credit_formula_version=CREDIT_FORMULA_VERSION,
        task_order_sha256=task_order_sha,
        base_model_manifest_sha256=model["sha256"],
        runtime_contract_sha256=runtime["sha256"],
        input_adapter_sha256=lineage["adapter_sha256"],
        input_rollout_adapter_sha256=lineage["rollout_adapter_sha256"],
        input_adapter_semantic_sha256=lineage["semantic_sha256"],
    )
    identity.validate()
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": FORMAL_EVAL_RUN_CONFIG_SCHEMA,
        "study_id": STUDY_ID,
        "formal_evaluation": True,
        "training_updates_allowed": False,
        "method": method,
        "seed": seed,
        "git_sha": git_sha,
        "authorization_sha256": authorization["sha256"],
        "readiness_sha256": readiness["sha256"],
        "training_manifest_path": lineage["training_manifest_path"],
        "training_manifest_sha256": lineage["training_manifest_sha256"],
        "all_online_training_manifest_sha256": {
            f"{key[0]}:{key[1]}": value["sha256"] for key, value in sorted(online.items())
        },
        "shared_sft_manifest_sha256": shared["sha256"],
        "base_model_path": str(base_model),
        "base_model_manifest_sha256": model["sha256"],
        "runtime_contract_sha256": runtime["sha256"],
        "adapter_path": str(lineage["adapter"]),
        "adapter_sha256": lineage["adapter_sha256"],
        "rollout_adapter_path": str(lineage["rollout_adapter"]),
        "rollout_adapter_sha256": lineage["rollout_adapter_sha256"],
        "adapter_semantic_sha256": lineage["semantic_sha256"],
        "identity": identity.to_dict(),
        "identity_sha256": identity.sha256,
        "test_public_path": str(TEST_PUBLIC_PATH.resolve()),
        "test_public_sha256": sha256_file(TEST_PUBLIC_PATH),
        "test_task_order_sha256": task_order_sha,
        "task_count": 120,
        "K": GROUP_SIZE,
        "expected_trajectory_count": 480,
        "browser_workers": 32,
        "concurrent_k4_groups": 8,
        "recovery": "resume_only_missing_task_groups_and_resample_infra_invalid_K4",
        "output_root": str(root),
    }
    config["run_config_sha256"] = sha256_json(config)
    config_path = root / "run_config.json"
    if config_path.is_file():
        _require(_json(config_path) == config, "formal evaluation run-config drift")
    else:
        atomic_write_json(config_path, config)
    store = CollectionStore(root / "collection", identity)
    store.recover_incomplete_attempts()
    worker_config = BrowserWorkerPoolConfig(
        total_workers=32,
        task_dir=DATASET_ROOT / "test",
        split="test",
        seed_dir=SEED_ROOT,
    )
    worker_config.validate()
    return {"root": root, "config": config, "identity": identity, "tasks": tasks, "store": store, "worker_config": worker_config}


def _group_id(index: int) -> str:
    return f"eval-g{index:04d}"


def _run_eval_batch(pool: VLLMBrowserWorkerPool, store: CollectionStore, batch: list[tuple[int, Any]]) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
        futures = [
            executor.submit(pool.run_group, store=store, group_id=_group_id(index), task=task)
            for index, task in batch
        ]
        return [future.result() for future in futures]


def finalize_formal_eval(prepared: Mapping[str, Any], *, elapsed_seconds: float) -> dict[str, Any]:
    root = Path(prepared["root"])
    config = prepared["config"]
    store: CollectionStore = prepared["store"]
    groups = store.load_committed_groups()
    _require(len(groups) == 120, "formal evaluation does not contain 120 committed task groups")
    expected = {_group_id(index): task.task_id for index, task in enumerate(prepared["tasks"])}
    _require({group["group_id"]: group["task_id"] for group in groups} == expected, "formal evaluation task/group roster drift")
    pairs = {(group["task_id"], trajectory["rollout_index"]) for group in groups for trajectory in group["trajectories"]}
    expected_pairs = {(task.task_id, rollout) for task in prepared["tasks"] for rollout in range(GROUP_SIZE)}
    _require(pairs == expected_pairs, "formal evaluation task/K roster is incomplete")
    roster_state = {
        "schema_version": "m4_long_horizon_eval_roster_state_v1",
        "task_count": 120,
        "task_order_sha256": config["test_task_order_sha256"],
        "complete": True,
    }
    collection = store.freeze_collection(
        iteration_index=prepared["identity"].iteration_index,
        stopped_for_token_budget=False,
        task_sampler_state=roster_state,
    )
    task_success = {
        group["task_id"]: sum(float(item["reward"]) for item in group["trajectories"]) / GROUP_SIZE
        for group in groups
    }
    strict_turns = [turn for group in groups for trajectory in group["trajectories"] for turn in trajectory["turns"]]
    manifest = {
        "schema_version": FORMAL_EVAL_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "formal_evaluation": True,
        "training_updates_allowed": False,
        "complete": True,
        "method": config["method"],
        "seed": config["seed"],
        "git_sha": config["git_sha"],
        "authorization_sha256": config["authorization_sha256"],
        "readiness_sha256": config["readiness_sha256"],
        "training_manifest_sha256": config["training_manifest_sha256"],
        "run_config_sha256": sha256_file(root / "run_config.json"),
        "identity_sha256": prepared["identity"].sha256,
        "test_public_sha256": config["test_public_sha256"],
        "test_task_order_sha256": config["test_task_order_sha256"],
        "task_count": 120,
        "K": GROUP_SIZE,
        "trajectory_count": len(pairs),
        "collection": {
            "relative_path": "collection/collection_manifest.json",
            "file_sha256": sha256_file(root / "collection" / "collection_manifest.json"),
            "collection_sha256": collection["collection_sha256"],
            "group_set_sha256": collection["group_set_sha256"],
            "attempt_journal_sha256": sha256_file(root / "collection" / "attempt_journal.jsonl"),
            "all_generated_action_tokens": collection["all_generated_action_tokens"],
            "committed_group_action_tokens": collection["committed_group_action_tokens"],
            "invalidated_action_tokens": collection["all_generated_action_tokens"] - collection["committed_group_action_tokens"],
        },
        "summary": {
            "task_macro_success": sum(task_success.values()) / 120,
            "trajectory_success": sum(trajectory["success"] for group in groups for trajectory in group["trajectories"]) / 480,
            "model_turns": sum(trajectory["model_turns"] for group in groups for trajectory in group["trajectories"]),
            "environment_steps": sum(trajectory["environment_steps"] for group in groups for trajectory in group["trajectories"]),
            "strict_json_success_fraction": sum(turn["strict_json_success"] for turn in strict_turns) / len(strict_turns),
            "schema_valid_fraction": sum(turn["schema_valid"] for turn in strict_turns) / len(strict_turns),
            "fallback_fraction": sum(turn["fallback_used"] for turn in strict_turns) / len(strict_turns),
            "finalization_process_elapsed_seconds": elapsed_seconds,
            "sum_trajectory_elapsed_seconds": sum(float(trajectory.get("elapsed_s", 0.0)) for group in groups for trajectory in group["trajectories"]),
        },
    }
    manifest["manifest_content_sha256"] = _content_sha(manifest, "manifest_content_sha256")
    atomic_write_json(root / FORMAL_EVAL_MANIFEST_NAME, manifest)
    return validate_formal_eval_manifest(root)


def validate_formal_eval_manifest(root: Path) -> dict[str, Any]:
    root = formal_output_path(root)
    path = root / FORMAL_EVAL_MANIFEST_NAME
    _require(path.is_file(), "formal evaluation manifest is missing")
    payload = _json(path)
    _require(payload.get("schema_version") == FORMAL_EVAL_MANIFEST_SCHEMA, "formal evaluation schema drift")
    _require(payload.get("formal_evaluation") is True and payload.get("training_updates_allowed") is False, "formal evaluation/training boundary drift")
    _require(payload.get("complete") is True, "formal evaluation is incomplete")
    _require(root == expected_eval_root(payload.get("method"), payload.get("seed")).resolve(), "formal evaluation root/identity drift")
    _require(payload.get("manifest_content_sha256") == _content_sha(payload, "manifest_content_sha256"), "formal evaluation self-hash drift")
    config_path = root / "run_config.json"
    config = _json(config_path)
    _require(payload.get("run_config_sha256") == sha256_file(config_path), "formal evaluation run-config hash drift")
    for field in ("method", "seed", "git_sha", "authorization_sha256", "readiness_sha256", "training_manifest_sha256", "identity_sha256", "test_public_sha256", "test_task_order_sha256"):
        _require(payload.get(field) == config.get(field), f"formal evaluation {field} drift")
    _require(payload.get("task_count") == 120 and payload.get("K") == 4 and payload.get("trajectory_count") == 480, "formal evaluation frozen matrix drift")
    identity = RunIdentity.from_mapping(config["identity"])
    shared, online = _all_training_lineage()
    _require(config.get("shared_sft_manifest_sha256") == shared["sha256"], "formal evaluation shared-SFT lineage drift")
    expected_online = {f"{key[0]}:{key[1]}": value["sha256"] for key, value in sorted(online.items())}
    _require(config.get("all_online_training_manifest_sha256") == expected_online, "formal evaluation online lineage matrix drift")
    target = shared if payload["method"] == "verified_sft" else online[(payload["method"], payload["seed"])]
    _require(payload["training_manifest_sha256"] == target["sha256"], "formal evaluation target training lineage drift")
    store = CollectionStore(root / "collection", identity)
    groups = store.load_committed_groups()
    _require(len(groups) == 120, "formal evaluation group count drift")
    _require(config.get("test_public_sha256") == sha256_file(TEST_PUBLIC_PATH), "formal evaluation current test-file drift")
    tasks = load_public_task_roster(TEST_PUBLIC_PATH, expected_count=120)
    expected = {_group_id(index): task.task_id for index, task in enumerate(tasks)}
    _require({group["group_id"]: group["task_id"] for group in groups} == expected, "formal evaluation task/group roster drift")
    expected_pairs = {(task.task_id, rollout) for task in tasks for rollout in range(GROUP_SIZE)}
    actual_pairs = {(group["task_id"], trajectory["rollout_index"]) for group in groups for trajectory in group["trajectories"]}
    _require(actual_pairs == expected_pairs, "formal evaluation task/K roster drift")
    collection_path = root / payload["collection"]["relative_path"]
    collection = _json(collection_path)
    _require(sha256_file(collection_path) == payload["collection"]["file_sha256"], "formal evaluation collection file drift")
    _require(collection["collection_sha256"] == payload["collection"]["collection_sha256"], "formal evaluation collection content drift")
    _require(
        sha256_file(store.journal.path) == payload["collection"]["attempt_journal_sha256"],
        "formal evaluation attempt-journal hash drift",
    )
    recomputed = store.freeze_collection(
        iteration_index=identity.iteration_index,
        stopped_for_token_budget=False,
        task_sampler_state=collection["task_sampler_state"],
    )
    _require(recomputed["collection_sha256"] == collection["collection_sha256"], "formal evaluation journal/collection drift")
    return {"path": str(path), "sha256": sha256_file(path), "payload": payload}


async def run_formal_eval(prepared: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(prepared["root"])
    completion = root / FORMAL_EVAL_MANIFEST_NAME
    if completion.is_file():
        return validate_formal_eval_manifest(root)
    store: CollectionStore = prepared["store"]
    collection_manifest = root / "collection" / "collection_manifest.json"
    started = time.monotonic()
    if not collection_manifest.is_file():
        identity: RunIdentity = prepared["identity"]
        config = prepared["config"]
        engine = await AsyncVLLMGenerationEngine.create(
            VLLMBackendConfig(
                base_model=config["base_model_path"],
                adapter_path=config["adapter_path"],
                adapter_sha256=config["adapter_sha256"],
                rollout_adapter_path=config["rollout_adapter_path"],
                rollout_adapter_sha256=config["rollout_adapter_sha256"],
                adapter_semantic_sha256=config["adapter_semantic_sha256"],
                seed=config["seed"],
            )
        )
        try:
            pool = VLLMBrowserWorkerPool(
                identity=identity,
                engine=engine,
                event_loop=asyncio.get_running_loop(),
                event_loop_thread_id=threading.get_ident(),
                config=prepared["worker_config"],
            )
            try:
                while True:
                    committed = set(store.journal.committed_group_ids)
                    pending = [
                        (index, task) for index, task in enumerate(prepared["tasks"])
                        if _group_id(index) not in committed
                    ]
                    if not pending:
                        break
                    batch = pending[: prepared["worker_config"].concurrent_group_slots]
                    await asyncio.to_thread(_run_eval_batch, pool, store, batch)
            finally:
                await asyncio.to_thread(pool.close)
        finally:
            engine.shutdown()
    return finalize_formal_eval(prepared, elapsed_seconds=time.monotonic() - started)
