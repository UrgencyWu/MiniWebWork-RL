#!/usr/bin/env python3
"""Run or resume one authorized 500k-token M5 WebShop online RL run."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.adapter_view import build_vllm_adapter_view
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_file, sha256_json
from miniwebwork.long_horizon_rl.vllm_backend import AsyncVLLMGenerationEngine, RolloutRequestContext, ThreadsafeVLLMBackend
from miniwebwork.m5_webshop_protocol import load_protocol
from miniwebwork.model_agent.agent_loop import run_model_episode
from miniwebwork.webshop_rl.agent import QwenWebShopAgent
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment
from miniwebwork.webshop_rl.formal_training import (
    AtomicTokenBudget,
    ITERATION_REPORT_SCHEMA,
    RUN_REPORT_SCHEMA,
    formal_run_root,
    formal_task_order,
    load_authorization,
    load_formal_plan,
    load_readiness,
    self_hash,
    validate_iteration_report,
    validate_run_report,
)
from miniwebwork.webshop_rl.online_training import (
    M5VLLMBackendConfig,
    audit_collection,
    build_committed_group,
    train_policy_iteration,
    trajectory_from_episode,
    validate_committed_group,
    validate_formal_iteration_learner_report,
)

# Reuse the already GPU-proven minimal compatibility and telemetry mechanics.
from m5_webshop_online_preflight import (  # noqa: E402
    DurableTokenLedger,
    TelemetryRecorder,
    _assert_clean,
    _git_sha,
    _minimal_sft_compatibility,
    _telemetry_audit,
)

GROUPS_PER_BATCH = 8
MAX_ATTEMPTS_PER_GROUP = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["content_sha256"] = self_hash(value)
    return value


def _load_hashed(path: Path, *, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("schema_version") == schema, f"M5 artifact schema drift: {path}")
    _require(value.get("content_sha256") == self_hash(value), f"M5 artifact self-hash drift: {path}")
    return value


def _append_allocation(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    with path.open("ab", buffering=0) as handle:
        handle.write(line)
        os.fsync(handle.fileno())


def _attempt_files(root: Path, group_id: str) -> list[Path]:
    return sorted(root.glob(f"{group_id}.a*.json"))


def _attempt_audit(attempts_root: Path) -> dict[str, Any]:
    valid = total = 0
    files = sorted(attempts_root.glob("*.json"))
    for path in files:
        payload = _load_hashed(path, schema="m5_webshop_formal_k4_attempt_v1")
        for rollout in payload["rollouts"]:
            total += 1
            valid += rollout["rollout_valid"] is True
    _require(total > 0, "M5 formal attempt audit is empty")
    return {
        "attempt_file_count": len(files),
        "trajectory_attempt_count": total,
        "infrastructure_valid_trajectory_count": valid,
        "infrastructure_valid_fraction": valid / total,
    }


async def _attempt_group(
    *,
    engine: AsyncVLLMGenerationEngine,
    loop: asyncio.AbstractEventLoop,
    event_loop_thread_id: int,
    task_id: str,
    group_id: str,
    iteration_index: int,
    group_index: int,
    seed: int,
    shared_prefix_turns: int,
    groups_root: Path,
    attempts_root: Path,
    ledger: DurableTokenLedger,
    budget: AtomicTokenBudget,
    lineage: Mapping[str, str],
    base_url: str,
) -> dict[str, Any] | None:
    destination = groups_root / f"{group_id}.json"
    if destination.is_file():
        return validate_committed_group(json.loads(destination.read_text(encoding="utf-8")))
    prior = _attempt_files(attempts_root, group_id)
    _require(len(prior) < MAX_ATTEMPTS_PER_GROUP, f"M5 group exhausted infrastructure retries: {group_id}")
    attempt_index = len(prior)
    reservation = budget.reserve(f"{group_id}.a{attempt_index}")
    if reservation is None:
        return None

    async def one_rollout(rollout_index: int):
        trajectory_id = f"{group_id}.a{attempt_index}.r{rollout_index}"
        context = RolloutRequestContext(
            run_seed=seed,
            iteration_index=iteration_index,
            group_id=group_id,
            attempt_index=attempt_index,
            trajectory_id=trajectory_id,
            rollout_index=rollout_index,
            shared_prefix_turns=shared_prefix_turns,
        )
        backend = ThreadsafeVLLMBackend(
            engine=engine,
            event_loop=loop,
            context=context,
            timeout_seconds=900,
            event_loop_thread_id=event_loop_thread_id,
        )
        env = WebShopHTTPEnvironment(base_url=base_url, split="train", timeout_seconds=120)
        agent = QwenWebShopAgent(backend)

        def charge(turn: dict[str, Any]) -> None:
            generated_ids = turn.get("generated_token_ids", [])
            reservation.charge(len(generated_ids))
            ledger.append(
                {
                    "schema_version": "m5_webshop_formal_generated_turn_v1",
                    "method": lineage["method"],
                    "seed": seed,
                    "iteration_index": iteration_index,
                    "group_id": group_id,
                    "task_id": task_id,
                    "attempt_index": attempt_index,
                    "rollout_index": rollout_index,
                    "turn_index": int(turn.get("model_turn_index", 0)),
                    "request_id": str(turn.get("request_id", "")),
                    "generated_action_tokens": len(generated_ids),
                    "generated_token_sha256": sha256_json(generated_ids),
                    "adapter_sha256": lineage["adapter_sha256"],
                }
            )

        try:
            return await asyncio.to_thread(run_model_episode, task_id, env, agent, 18, 15, charge, None)
        finally:
            env.close()

    try:
        episodes = await asyncio.gather(*(one_rollout(index) for index in range(4)))
    finally:
        reservation.release()
    attempt = _hashed(
        {
            "schema_version": "m5_webshop_formal_k4_attempt_v1",
            "formal_training": True,
            "git_sha": lineage["git_sha"],
            "protocol_sha256": lineage["protocol_sha256"],
            "method": lineage["method"],
            "seed": seed,
            "iteration_index": iteration_index,
            "group_id": group_id,
            "group_index": group_index,
            "task_id": task_id,
            "attempt_index": attempt_index,
            "rollouts": [
                {
                    "rollout_index": index,
                    "rollout_valid": episode.get("rollout_valid"),
                    "reward": episode.get("reward"),
                    "task_score": episode.get("task_score"),
                    "termination_reason": episode.get("termination_reason"),
                    "error": episode.get("error"),
                    "model_turns": episode.get("model_turns"),
                }
                for index, episode in enumerate(episodes)
            ],
        }
    )
    atomic_write_json(attempts_root / f"{group_id}.a{attempt_index}.json", attempt)
    if not all(episode.get("rollout_valid") is True for episode in episodes):
        return None
    trajectories = [
        trajectory_from_episode(
            episode,
            trajectory_id=f"{group_id}.a{attempt_index}.r{index}",
            rollout_index=index,
            adapter_sha256=lineage["adapter_sha256"],
            rollout_adapter_sha256=lineage["rollout_adapter_sha256"],
            adapter_semantic_sha256=lineage["adapter_semantic_sha256"],
        )
        for index, episode in enumerate(episodes)
    ]
    group = build_committed_group(
        task_id=task_id,
        group_id=group_id,
        attempt_index=attempt_index,
        trajectories=trajectories,
        git_sha=lineage["git_sha"],
        protocol_sha256=lineage["protocol_sha256"],
        adapter_sha256=lineage["adapter_sha256"],
        rollout_adapter_sha256=lineage["rollout_adapter_sha256"],
        adapter_semantic_sha256=lineage["adapter_semantic_sha256"],
    )
    atomic_write_json(destination, group)
    return group


async def _collect_iteration(
    *,
    engine: AsyncVLLMGenerationEngine,
    task_ids: list[str],
    global_start: int,
    iteration_index: int,
    seed: int,
    shared_prefix_turns: int,
    iteration_root: Path,
    ledger: DurableTokenLedger,
    budget: AtomicTokenBudget,
    lineage: Mapping[str, str],
    base_url: str,
) -> list[dict[str, Any]]:
    groups_root = iteration_root / "groups"
    attempts_root = iteration_root / "attempts"
    groups_root.mkdir(parents=True, exist_ok=True)
    attempts_root.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    pending = list(enumerate(task_ids))
    while pending:
        completed_ids = {
            path.stem for path in groups_root.glob("g*.json")
        }
        pending = [
            (offset, task_id)
            for offset, task_id in pending
            if f"g{global_start + offset:06d}" not in completed_ids
        ]
        if not pending:
            break
        available_slots = (budget.cap - budget.spent) // budget.reservation_size
        if available_slots <= 0:
            break
        wave = pending[: min(GROUPS_PER_BATCH, available_slots)]
        results = await asyncio.gather(
            *(
                _attempt_group(
                    engine=engine,
                    loop=loop,
                    event_loop_thread_id=loop_thread,
                    task_id=task_id,
                    group_id=f"g{global_start + offset:06d}",
                    iteration_index=iteration_index,
                    group_index=global_start + offset,
                    seed=seed,
                    shared_prefix_turns=shared_prefix_turns,
                    groups_root=groups_root,
                    attempts_root=attempts_root,
                    ledger=ledger,
                    budget=budget,
                    lineage=lineage,
                    base_url=base_url,
                )
                for offset, task_id in wave
            )
        )
        if not any(result is not None for result in results) and (budget.cap - budget.spent) < budget.reservation_size:
            break
    return [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(groups_root.glob("g*.json"))
    ]


def _completed_iterations(run_root: Path, *, method: str, seed: int) -> list[dict[str, Any]]:
    output = []
    previous_output_sha = None
    previous_optimizer_sha = None
    index = 0
    while True:
        path = run_root / "iterations" / f"i{index:04d}" / "iteration_report.json"
        if not path.is_file():
            break
        report = validate_iteration_report(json.loads(path.read_text(encoding="utf-8")))
        _require(report["method"] == method and report["seed"] == seed and report["iteration_index"] == index, "M5 iteration roster drift")
        learner = validate_formal_iteration_learner_report(Path(report["learner_report_path"]))
        _require(sha256_file(Path(report["learner_report_path"])) == report["learner_report_sha256"], "M5 learner report file hash drift")
        _require(learner["output_adapter_sha256"] == report["output_adapter_sha256"], "M5 iteration output adapter drift")
        if previous_output_sha is not None:
            _require(report["input_adapter_sha256"] == previous_output_sha, "M5 iteration adapter chain drift")
            _require(report["input_optimizer_sha256"] == previous_optimizer_sha, "M5 iteration optimizer chain drift")
        else:
            _require(report["input_optimizer_sha256"] is None, "M5 first iteration unexpectedly has optimizer state")
        previous_output_sha = report["output_adapter_sha256"]
        previous_optimizer_sha = report["output_optimizer_sha256"]
        output.append(report)
        index += 1
    return output


async def run(args: argparse.Namespace) -> dict[str, Any]:
    _assert_clean()
    git_sha = _git_sha()
    protocol_bundle = load_protocol()
    _require(protocol_bundle["git_sha"] == git_sha, "M5 formal Git lineage drift")
    _require(protocol_bundle["payload"]["formal_submission_allowed"] is False, "M5 preflight protocol was mutated to authorize formal work")
    plan_bundle = load_formal_plan(args.formal_plan)
    plan = plan_bundle["payload"]
    _require(plan["preflight_evidence"]["protocol_sha256"] == protocol_bundle["sha256"], "M5 preflight/formal protocol drift")
    readiness_bundle = load_readiness(args.readiness, git_sha=git_sha, plan_sha256=plan_bundle["sha256"])
    authorization_bundle = load_authorization(
        args.authorization,
        git_sha=git_sha,
        plan_sha256=plan_bundle["sha256"],
        readiness_sha256=readiness_bundle["sha256"],
    )
    _require(args.method in plan["matrix"]["methods"] and args.seed in plan["matrix"]["seeds"], "M5 run outside authorized matrix")
    expected_root = formal_run_root(args.method, args.seed)
    output = (args.output_dir or expected_root).expanduser().resolve()
    _require(output == expected_root, "M5 formal output root drift")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "run_report.json"
    if report_path.is_file():
        complete = validate_run_report(json.loads(report_path.read_text(encoding="utf-8")))
        iterations = _completed_iterations(output, method=args.method, seed=args.seed)
        _require(len(iterations) == complete["iteration_count"], "M5 completed run iteration count drift")
        return complete

    base_model = args.base_model.expanduser().resolve()
    initial_adapter = args.initial_adapter.expanduser().resolve()
    compatibility = _minimal_sft_compatibility(protocol=protocol_bundle["payload"], base_model=base_model, adapter=initial_adapter)
    preflight_path = PROJECT_ROOT / plan["preflight_evidence"]["report_path"]
    preflight = _load_hashed(preflight_path, schema="m5_webshop_online_preflight_report_v1")
    _require(preflight["passed"] is True and preflight["content_sha256"] == plan["preflight_evidence"]["report_content_sha256"], "M5 audited online preflight drift")
    task_order = formal_task_order(args.seed)
    invocation = _hashed(
        {
            "schema_version": "m5_webshop_formal_invocation_v1",
            "formal_training": True,
            "git_sha": git_sha,
            "protocol_sha256": protocol_bundle["sha256"],
            "formal_plan_sha256": plan_bundle["sha256"],
            "readiness_sha256": readiness_bundle["sha256"],
            "authorization_sha256": authorization_bundle["sha256"],
            "method": args.method,
            "seed": args.seed,
            "task_order_sha256": sha256_json(task_order),
            "task_count": len(task_order),
            "initial_adapter": str(initial_adapter),
            "initial_adapter_sha256": directory_sha256(initial_adapter),
            "base_model": str(base_model),
            "token_cap": plan["online_budget"]["generated_action_token_cap_per_run"],
            "shared_prefix_turns": plan["online_budget"]["shared_prefix_turns"],
            "sft_compatibility": compatibility,
        }
    )
    invocation_path = output / "invocation.json"
    if invocation_path.is_file():
        _require(json.loads(invocation_path.read_text(encoding="utf-8")) == invocation, "M5 formal invocation drift across recovery")
    else:
        atomic_write_json(invocation_path, invocation)
    _append_allocation(
        output / "allocations.jsonl",
        {
            "schema_version": "m5_webshop_formal_allocation_v1",
            "job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "method": args.method,
            "seed": args.seed,
            "git_sha": git_sha,
        },
    )

    ledger = DurableTokenLedger(output / "generated_turn_ledger.jsonl")
    initial_ledger = ledger.audit()
    budget = AtomicTokenBudget(
        cap=plan["online_budget"]["generated_action_token_cap_per_run"],
        spent=initial_ledger["generated_action_tokens"],
        reservation_size=plan["online_budget"]["maximum_atomic_group_token_reservation"],
    )
    iterations = _completed_iterations(output, method=args.method, seed=args.seed)
    current_adapter = initial_adapter if not iterations else Path(iterations[-1]["output_adapter"])
    current_optimizer = None if not iterations else Path(iterations[-1]["output_optimizer"])
    while budget.cap - budget.spent >= budget.reservation_size:
        iteration_index = len(iterations)
        iteration_root = output / "iterations" / f"i{iteration_index:04d}"
        iteration_root.mkdir(parents=True, exist_ok=True)
        global_start = iteration_index * plan["online_budget"]["tasks_per_iteration"]
        _require(global_start < len(task_order), "M5 formal task roster exhausted before token cap")
        task_ids = list(task_order[global_start : global_start + plan["online_budget"]["tasks_per_iteration"]])
        manifest_path = iteration_root / "iteration_manifest.json"
        if manifest_path.is_file():
            manifest = _load_hashed(manifest_path, schema="m5_webshop_formal_iteration_manifest_v1")
            expected_static = {
                "git_sha": git_sha,
                "protocol_sha256": protocol_bundle["sha256"],
                "method": args.method,
                "seed": args.seed,
                "iteration_index": iteration_index,
                "global_group_start": global_start,
                "task_ids": task_ids,
                "task_ids_sha256": sha256_json(task_ids),
                "input_adapter": str(current_adapter),
                "input_adapter_sha256": directory_sha256(current_adapter),
                "input_optimizer": str(current_optimizer) if current_optimizer else None,
                "input_optimizer_sha256": sha256_file(current_optimizer) if current_optimizer else None,
            }
            _require(
                all(manifest.get(key) == value for key, value in expected_static.items()),
                "M5 formal iteration manifest drift across recovery",
            )
        else:
            manifest = _hashed(
                {
                    "schema_version": "m5_webshop_formal_iteration_manifest_v1",
                    "formal_training": True,
                    "git_sha": git_sha,
                    "protocol_sha256": protocol_bundle["sha256"],
                    "method": args.method,
                    "seed": args.seed,
                    "iteration_index": iteration_index,
                    "global_group_start": global_start,
                    "task_ids": task_ids,
                    "task_ids_sha256": sha256_json(task_ids),
                    "ledger_start_generated_action_tokens": ledger.audit()["generated_action_tokens"],
                    "input_adapter": str(current_adapter),
                    "input_adapter_sha256": directory_sha256(current_adapter),
                    "input_optimizer": str(current_optimizer) if current_optimizer else None,
                    "input_optimizer_sha256": sha256_file(current_optimizer) if current_optimizer else None,
                }
            )
            atomic_write_json(manifest_path, manifest)
        view = build_vllm_adapter_view(
            source_adapter=current_adapter,
            destination=iteration_root / "input_rollout_adapter",
            base_model=base_model,
        )
        lineage = {
            "git_sha": git_sha,
            "protocol_sha256": protocol_bundle["sha256"],
            "method": args.method,
            "adapter_sha256": directory_sha256(current_adapter),
            "rollout_adapter_sha256": view["view_directory_sha256"],
            "adapter_semantic_sha256": view["semantic_tensor_sha256"],
        }
        config = M5VLLMBackendConfig(
            base_model=str(base_model),
            adapter_path=str(current_adapter),
            adapter_sha256=lineage["adapter_sha256"],
            rollout_adapter_path=str(iteration_root / "input_rollout_adapter"),
            rollout_adapter_sha256=lineage["rollout_adapter_sha256"],
            adapter_semantic_sha256=lineage["adapter_semantic_sha256"],
            seed=args.seed,
        )
        engine = await AsyncVLLMGenerationEngine.create(config)
        job_id = os.environ.get("SLURM_JOB_ID", "manual")
        generation_telemetry = output / "telemetry" / f"generation_i{iteration_index:04d}_{job_id}.csv"
        try:
            with TelemetryRecorder(generation_telemetry):
                groups = await _collect_iteration(
                    engine=engine,
                    task_ids=task_ids,
                    global_start=global_start,
                    iteration_index=iteration_index,
                    seed=args.seed,
                    shared_prefix_turns=plan["online_budget"]["shared_prefix_turns"],
                    iteration_root=iteration_root,
                    ledger=ledger,
                    budget=budget,
                    lineage=lineage,
                    base_url=args.base_url,
                )
            await engine.switch_to_learner()
        finally:
            try:
                if engine._phase == "generation":
                    await engine.switch_to_learner()
            except Exception:
                pass
            engine.shutdown()
        ledger_end = ledger.audit()["generated_action_tokens"]
        iteration_tokens = ledger_end - manifest["ledger_start_generated_action_tokens"]
        if not groups:
            _require(budget.cap - budget.spent < budget.reservation_size, "M5 formal iteration produced no committed groups")
            break
        collection = audit_collection(groups, all_generated_action_tokens=iteration_tokens)
        attempts = _attempt_audit(iteration_root / "attempts")
        collection_report = _hashed(
            {
                "schema_version": "m5_webshop_formal_iteration_collection_v1",
                "formal_training": True,
                "git_sha": git_sha,
                "protocol_sha256": protocol_bundle["sha256"],
                "method": args.method,
                "seed": args.seed,
                "iteration_index": iteration_index,
                "generated_action_tokens": iteration_tokens,
                "attempts": attempts,
                "metrics": collection,
                "group_sha256": {group["group_id"]: group["content_sha256"] for group in groups},
            }
        )
        collection_path = iteration_root / "collection_report.json"
        atomic_write_json(collection_path, collection_report)
        learner_telemetry = output / "telemetry" / f"learner_i{iteration_index:04d}_{job_id}.csv"
        with TelemetryRecorder(learner_telemetry):
            learner = train_policy_iteration(
                method=args.method,
                groups=groups,
                iteration_generated_action_tokens=iteration_tokens,
                base_model=base_model,
                initial_adapter=current_adapter,
                input_adapter_semantic_sha256=view["semantic_tensor_sha256"],
                input_optimizer=current_optimizer,
                output_root=iteration_root / "learner",
                protocol=protocol_bundle["payload"],
                git_sha=git_sha,
                protocol_sha256=protocol_bundle["sha256"],
                iteration_index=iteration_index,
                seed=args.seed,
            )
        learner_path = Path(learner["output_adapter"]).parent / "learner_report.json"
        iteration_report = _hashed(
            {
                "schema_version": ITERATION_REPORT_SCHEMA,
                "complete": True,
                "formal_training": True,
                "git_sha": git_sha,
                "protocol_sha256": protocol_bundle["sha256"],
                "method": args.method,
                "seed": args.seed,
                "iteration_index": iteration_index,
                "generated_action_tokens": iteration_tokens,
                "cumulative_generated_action_tokens": ledger_end,
                "group_count": len(groups),
                "collection_report_path": str(collection_path),
                "collection_report_sha256": sha256_file(collection_path),
                "learner_report_path": str(learner_path),
                "learner_report_sha256": sha256_file(learner_path),
                "input_adapter_sha256": lineage["adapter_sha256"],
                "input_optimizer": str(current_optimizer) if current_optimizer else None,
                "input_optimizer_sha256": sha256_file(current_optimizer) if current_optimizer else None,
                "output_adapter": learner["output_adapter"],
                "output_adapter_sha256": learner["output_adapter_sha256"],
                "output_adapter_semantic_sha256": learner["output_adapter_semantic_sha256"],
                "output_optimizer": learner["output_optimizer"],
                "output_optimizer_sha256": learner["output_optimizer_sha256"],
                "iteration_optimizer_updates": learner["iteration_optimizer_updates"],
                "cumulative_optimizer_updates": learner["cumulative_optimizer_updates_after"],
                "effective_optimizer_action_tokens": learner["effective_optimizer_action_tokens"],
                "effective_optimizer_action_token_fraction": learner["effective_optimizer_action_token_fraction"],
            }
        )
        iteration_report_path = iteration_root / "iteration_report.json"
        atomic_write_json(iteration_report_path, iteration_report)
        atomic_write_json(
            output / "latest_checkpoint.json",
            _hashed(
                {
                    "schema_version": "m5_webshop_formal_latest_checkpoint_v1",
                    "method": args.method,
                    "seed": args.seed,
                    "iteration_index": iteration_index,
                    "iteration_report_path": str(iteration_report_path),
                    "iteration_report_sha256": sha256_file(iteration_report_path),
                    "adapter": learner["output_adapter"],
                    "adapter_sha256": learner["output_adapter_sha256"],
                    "optimizer": learner["output_optimizer"],
                    "optimizer_sha256": learner["output_optimizer_sha256"],
                }
            ),
        )
        iterations.append(iteration_report)
        current_adapter = Path(learner["output_adapter"])
        current_optimizer = Path(learner["output_optimizer"])

    _require(iterations, "M5 formal run has no completed iterations")
    ledger_audit = ledger.audit()
    all_groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((output / "iterations").glob("i*/groups/g*.json"))
    ]
    attempt_paths = sorted((output / "iterations").glob("i*/attempts/*.json"))
    valid = total = 0
    for path in attempt_paths:
        payload = _load_hashed(path, schema="m5_webshop_formal_k4_attempt_v1")
        for rollout in payload["rollouts"]:
            total += 1
            valid += rollout["rollout_valid"] is True
    all_attempts = {
        "attempt_file_count": len(attempt_paths),
        "trajectory_attempt_count": total,
        "infrastructure_valid_trajectory_count": valid,
        "infrastructure_valid_fraction": valid / total,
    }
    collection = audit_collection(all_groups, all_generated_action_tokens=ledger_audit["generated_action_tokens"])
    generation_audit = _telemetry_audit(sorted((output / "telemetry").glob("generation_*.csv")))
    learner_audit = _telemetry_audit(sorted((output / "telemetry").glob("learner_*.csv")))
    final_iteration = iterations[-1]
    gates = {
        "token_budget_lower": ledger_audit["generated_action_tokens"] >= plan["online_budget"]["minimum_final_billed_action_tokens"],
        "token_budget_upper": ledger_audit["generated_action_tokens"] <= plan["online_budget"]["generated_action_token_cap_per_run"],
        "infrastructure_valid": all_attempts["infrastructure_valid_fraction"] >= plan["success_contract"]["attempt_infrastructure_valid_fraction_minimum"],
        "committed_token_fraction": collection["committed_group_action_tokens"] / ledger_audit["generated_action_tokens"] >= plan["success_contract"]["committed_token_fraction_minimum"],
        "optimizer_updates": final_iteration["cumulative_optimizer_updates"] >= plan["success_contract"]["cumulative_optimizer_updates_minimum"],
        "all_iteration_parity": all(
            validate_formal_iteration_learner_report(Path(item["learner_report_path"]))["initial_replay_parity"]["passed"]
            for item in iterations
        ),
    }
    # Semantic and directory hashes are different identities; compare the
    # first and final semantic hashes directly for the parameter-update gate.
    initial_view = build_vllm_adapter_view(
        source_adapter=initial_adapter,
        destination=output / "iterations" / "i0000" / "input_rollout_adapter",
        base_model=base_model,
    )
    gates["real_parameter_update"] = final_iteration["output_adapter_semantic_sha256"] != initial_view["semantic_tensor_sha256"]
    unmet = [name for name, passed in gates.items() if not passed]
    _require(not unmet, f"M5 formal run gates failed: {unmet}")
    report = _hashed(
        {
            "schema_version": RUN_REPORT_SCHEMA,
            "complete": True,
            "passed": True,
            "formal_training": True,
            "git_sha": git_sha,
            "protocol_sha256": protocol_bundle["sha256"],
            "formal_plan_sha256": plan_bundle["sha256"],
            "readiness_sha256": readiness_bundle["sha256"],
            "authorization_sha256": authorization_bundle["sha256"],
            "method": args.method,
            "seed": args.seed,
            "iteration_count": len(iterations),
            "group_count": len(all_groups),
            "generated_action_tokens": ledger_audit["generated_action_tokens"],
            "generated_turn_ledger": str(output / "generated_turn_ledger.jsonl"),
            "generated_turn_ledger_sha256": ledger_audit["sha256"],
            "cumulative_optimizer_updates": final_iteration["cumulative_optimizer_updates"],
            "final_adapter": final_iteration["output_adapter"],
            "final_adapter_sha256": final_iteration["output_adapter_sha256"],
            "final_adapter_semantic_sha256": final_iteration["output_adapter_semantic_sha256"],
            "final_optimizer": final_iteration["output_optimizer"],
            "final_optimizer_sha256": final_iteration["output_optimizer_sha256"],
            "attempts": all_attempts,
            "collection_metrics": collection,
            "generation_telemetry": generation_audit,
            "learner_telemetry": learner_audit,
            "gates": gates,
            "unmet_gates": unmet,
        }
    )
    atomic_write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("multi_turn_grpo", "anchor_gigpo"))
    parser.add_argument("--seed", required=True, type=int, choices=(20260801, 20260802, 20260803))
    parser.add_argument("--formal-plan", type=Path, default=PROJECT_ROOT / "data" / "m5_webshop_formal_plan_v1.json")
    parser.add_argument("--readiness", type=Path, default=PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "readiness" / "readiness_manifest_v1.json")
    parser.add_argument("--authorization", type=Path, default=PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "readiness" / "formal_authorization_v1.json")
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--initial-adapter", type=Path, default=PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "preflight" / "sft_gpu" / "training" / "final_adapter_epoch_1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = asyncio.run(run(args))
        print(json.dumps(report, indent=2, sort_keys=True))
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        output = args.output_dir or formal_run_root(args.method, args.seed)
        output.mkdir(parents=True, exist_ok=True)
        failure = _hashed(
            {
                "schema_version": "m5_webshop_formal_failure_v1",
                "complete": True,
                "passed": False,
                "formal_training": True,
                "git_sha": _git_sha(),
                "method": args.method,
                "seed": args.seed,
                "exception_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "traceback": traceback.format_exc()[-16000:],
            }
        )
        atomic_write_json(output / "run_failure.json", failure)
        raise


if __name__ == "__main__":
    main()
