#!/usr/bin/env python3
"""Run or resume one authorized M5 500-task K=4 frozen evaluation identity."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_file, sha256_json
from miniwebwork.long_horizon_rl.vllm_backend import (
    AsyncVLLMGenerationEngine,
    RawAsyncVLLMGenerationEngine,
    RawVLLMBackendConfig,
    RolloutRequestContext,
    ThreadsafeVLLMBackend,
)
from miniwebwork.m5_webshop_protocol import load_protocol
from miniwebwork.model_agent.agent_loop import run_model_episode
from miniwebwork.webshop_rl.agent import QwenWebShopAgent
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment
from miniwebwork.webshop_rl.formal_evaluation import (
    ATTEMPT_SCHEMA,
    IDENTITIES,
    RUN_REPORT_SCHEMA,
    evaluation_run_root,
    frozen_test_task_ids,
    identity_spec,
    load_eval_authorization,
    load_eval_plan,
    summarize_groups,
    validate_eval_report,
    validate_inference_identity,
)
from miniwebwork.webshop_rl.formal_training import self_hash
from miniwebwork.webshop_rl.online_training import (
    M5VLLMBackendConfig,
    build_committed_group,
    trajectory_from_episode,
    validate_committed_group,
)

from m5_webshop_online_preflight import TelemetryRecorder, _assert_clean, _git_sha, _telemetry_audit  # noqa: E402

GROUPS_PER_WAVE = 8
MAX_ATTEMPTS_PER_GROUP = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["content_sha256"] = self_hash(value)
    return value


def _load_hashed(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("schema_version") == schema, f"M5 frozen evaluation schema drift: {path}")
    _require(value.get("content_sha256") == self_hash(value), f"M5 frozen evaluation self-hash drift: {path}")
    return value


def _append_allocation(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    with path.open("ab", buffering=0) as handle:
        handle.write(line)
        os.fsync(handle.fileno())


def _task_metadata(goals_path: Path, expected_sha256: str) -> dict[str, dict[str, Any]]:
    _require(sha256_file(goals_path) == expected_sha256, "M5 frozen evaluation goals bytes drift")
    goals = json.loads(goals_path.read_text(encoding="utf-8"))
    _require(isinstance(goals, list) and len(goals) == 12087, "M5 frozen evaluation goals roster drift")
    output = {}
    for index, goal in enumerate(goals[:500]):
        _require(isinstance(goal, Mapping) and goal.get("goal_index") == index, "M5 frozen test goal index drift")
        category = str(goal.get("category") or "unknown")
        count = 0
        attributes = goal.get("attributes")
        if isinstance(attributes, Mapping):
            count += sum(value not in (None, "", [], {}) for value in attributes.values())
        elif isinstance(attributes, list):
            count += sum(value not in (None, "", [], {}) for value in attributes)
        options = goal.get("goal_options")
        if isinstance(options, Mapping):
            count += sum(value not in (None, "", [], {}) for value in options.values())
        elif isinstance(options, list):
            count += sum(value not in (None, "", [], {}) for value in options)
        price = goal.get("price_upper")
        if isinstance(price, (int, float)) and not isinstance(price, bool) and math.isfinite(float(price)):
            count += 1
        task_id = f"webshop_goal_{index:05d}"
        output[task_id] = {"goal_index": index, "category": category, "constraint_count": count}
    _require(tuple(output) == frozen_test_task_ids(), "M5 frozen evaluation task metadata order drift")
    return output


def _attempt_paths(root: Path, group_id: str) -> list[Path]:
    return sorted(root.glob(f"{group_id}.a*.json"))


async def _attempt_group(
    *,
    engine: AsyncVLLMGenerationEngine | RawAsyncVLLMGenerationEngine,
    loop: asyncio.AbstractEventLoop,
    event_loop_thread_id: int,
    task_id: str,
    task_metadata: Mapping[str, Any],
    group_id: str,
    group_index: int,
    evaluation_seed: int,
    shared_prefix_turns: int,
    groups_root: Path,
    attempts_root: Path,
    lineage: Mapping[str, str],
    base_url: str,
) -> dict[str, Any] | None:
    destination = groups_root / f"{group_id}.json"
    if destination.is_file():
        group = validate_committed_group(json.loads(destination.read_text(encoding="utf-8")))
        _require(group.get("task_metadata") == dict(task_metadata), "M5 frozen evaluation task metadata changed across recovery")
        return group
    prior = _attempt_paths(attempts_root, group_id)
    _require(
        [path.name for path in prior]
        == [f"{group_id}.a{index}.json" for index in range(len(prior))],
        f"M5 frozen evaluation attempt roster drift: {group_id}",
    )
    for path in prior:
        previous = _load_hashed(path, ATTEMPT_SCHEMA)
        _require(previous["identity"] == lineage["identity"] and previous["task_id"] == task_id, f"M5 frozen evaluation prior attempt identity drift: {group_id}")
    _require(len(prior) < MAX_ATTEMPTS_PER_GROUP, f"M5 frozen evaluation exhausted infrastructure retries: {group_id}")
    attempt_index = len(prior)

    async def one_rollout(rollout_index: int) -> dict[str, Any]:
        context = RolloutRequestContext(
            run_seed=evaluation_seed,
            iteration_index=0,
            group_id=group_id,
            attempt_index=attempt_index,
            trajectory_id=f"{group_id}.a{attempt_index}.r{rollout_index}",
            rollout_index=rollout_index,
            shared_prefix_turns=shared_prefix_turns,
            # Infrastructure retries keep the frozen paired sampling seed;
            # request ids remain unique through the real attempt index.
            sampling_attempt_index=0,
        )
        backend = ThreadsafeVLLMBackend(
            engine=engine,
            event_loop=loop,
            context=context,
            timeout_seconds=900,
            event_loop_thread_id=event_loop_thread_id,
        )
        env = WebShopHTTPEnvironment(base_url=base_url, split="test", timeout_seconds=120)
        agent = QwenWebShopAgent(backend)
        try:
            return await asyncio.to_thread(run_model_episode, task_id, env, agent, 18, 15, None, None)
        finally:
            env.close()

    episodes = await asyncio.gather(*(one_rollout(index) for index in range(4)))
    attempt = _hashed(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "formal_evaluation": True,
            "training_updates_allowed": False,
            "git_sha": lineage["git_sha"],
            "protocol_sha256": lineage["protocol_sha256"],
            "identity": lineage["identity"],
            "group_id": group_id,
            "group_index": group_index,
            "task_id": task_id,
            "attempt_index": attempt_index,
            "generated_action_tokens": sum(
                len(turn.get("generated_token_ids", [])) for episode in episodes for turn in episode.get("turns", [])
            ),
            "rollouts": [
                {
                    "rollout_index": index,
                    "rollout_valid": episode.get("rollout_valid"),
                    "success": episode.get("success"),
                    "task_score": episode.get("task_score"),
                    "termination_reason": episode.get("termination_reason"),
                    "model_turns": episode.get("model_turns"),
                    "environment_steps": episode.get("environment_steps"),
                    "error": episode.get("error"),
                }
                for index, episode in enumerate(episodes)
            ],
        }
    )
    atomic_write_json(attempts_root / f"{group_id}.a{attempt_index}.json", attempt)
    if not all(episode.get("rollout_valid") is True for episode in episodes):
        return None
    trajectories = []
    for index, episode in enumerate(episodes):
        trajectory = trajectory_from_episode(
            episode,
            trajectory_id=f"{group_id}.a{attempt_index}.r{index}",
            rollout_index=index,
            adapter_sha256=lineage["adapter_sha256"],
            rollout_adapter_sha256=lineage["rollout_adapter_sha256"],
            adapter_semantic_sha256=lineage["adapter_semantic_sha256"],
        )
        trajectory["elapsed_seconds"] = float(episode.get("elapsed_s", 0.0))
        trajectory["evaluation_turn_summary"] = [
            {
                "turn_index": turn_index,
                "schema_valid": bool(turn.get("schema_valid")),
                "errors": list(turn.get("errors", [])),
                "action_result": turn.get("action_result"),
            }
            for turn_index, turn in enumerate(episode.get("turns", []), start=1)
        ]
        trajectories.append(trajectory)
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
    group["formal_evaluation"] = True
    group["training_updates_allowed"] = False
    group["identity"] = lineage["identity"]
    group["task_metadata"] = dict(task_metadata)
    group["content_sha256"] = self_hash(group)
    group = validate_committed_group(group)
    atomic_write_json(destination, group)
    return group


def _load_groups(root: Path, task_ids: tuple[str, ...], identity: str) -> list[dict[str, Any]]:
    paths = sorted(root.glob("e*.json"))
    groups = [validate_committed_group(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    for path, group in zip(paths, groups):
        _require(group["group_id"] == path.stem, "M5 frozen evaluation group/path drift")
        index = int(path.stem[1:])
        _require(0 <= index < 500 and group["task_id"] == task_ids[index], "M5 frozen evaluation group/task drift")
        _require(group.get("identity") == identity, "M5 frozen evaluation group identity drift")
    _require(len({group["group_id"] for group in groups}) == len(groups), "M5 frozen evaluation duplicate group drift")
    return groups


def _attempt_audit(attempts_root: Path) -> dict[str, Any]:
    paths = sorted(attempts_root.glob("*.json"))
    valid = total = generated = 0
    for path in paths:
        payload = _load_hashed(path, ATTEMPT_SCHEMA)
        generated += int(payload["generated_action_tokens"])
        for item in payload["rollouts"]:
            total += 1
            valid += item["rollout_valid"] is True
    _require(total > 0, "M5 frozen evaluation attempt audit is empty")
    return {
        "attempt_file_count": len(paths),
        "trajectory_attempt_count": total,
        "infrastructure_valid_trajectory_count": valid,
        "infrastructure_valid_fraction": valid / total,
        "all_attempt_generated_action_tokens": generated,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    _assert_clean()
    git_sha = _git_sha()
    _require(git_sha == args.expected_git_sha, "M5 frozen evaluation Git SHA drift")
    protocol = load_protocol()
    _require(protocol["git_sha"] == git_sha, "M5 frozen evaluation consumer Git lineage drift")
    plan_bundle = load_eval_plan(args.eval_plan)
    plan = plan_bundle["payload"]
    _require(protocol["sha256"] == plan["protocol_sha256"], "M5 frozen evaluation protocol drift")
    authorization = load_eval_authorization(
        args.authorization,
        consumer_git_sha=git_sha,
        plan_sha256=plan_bundle["sha256"],
    )
    _require(args.identity in authorization["payload"]["identities"], "M5 evaluation identity is not authorized")
    expected_root = evaluation_run_root(args.identity)
    output = (args.output_dir or expected_root).expanduser().resolve()
    _require(output == expected_root, "M5 frozen evaluation output root drift")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "run_report.json"
    if report_path.is_file():
        return validate_eval_report(json.loads(report_path.read_text(encoding="utf-8")))
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M5 frozen evaluation requires exactly one visible GPU")
    spec = identity_spec(plan, args.identity)
    # Authorization rehashes the full base model once immediately before all
    # submissions. Avoid eight simultaneous multi-gigabyte file rehashes here.
    identity_audit = validate_inference_identity(plan, identity=args.identity, verify_base_model_files=False)
    task_ids = frozen_test_task_ids()
    runtime_root = PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "upstream" / "webshop_full"
    metadata = _task_metadata(runtime_root / "goals.json", plan["test"]["goals_sha256"])
    groups_root = output / "groups"
    attempts_root = output / "attempts"
    groups_root.mkdir(exist_ok=True)
    attempts_root.mkdir(exist_ok=True)
    invocation = _hashed(
        {
            "schema_version": "m5_webshop_frozen_eval_invocation_v1",
            "formal_evaluation": True,
            "training_updates_allowed": False,
            "optimizer_state_loaded": False,
            "consumer_git_sha": git_sha,
            "producer_training_git_sha": plan["producer_training_git_sha"],
            "protocol_sha256": protocol["sha256"],
            "eval_plan_sha256": plan_bundle["sha256"],
            "authorization_sha256": authorization["sha256"],
            "identity": args.identity,
            "identity_spec": spec,
            "identity_audit": identity_audit,
            "task_order_sha256": sha256_json(task_ids),
            "task_count": 500,
            "K": 4,
            "evaluation_seed": plan["test"]["evaluation_seed"],
            "shared_prefix_turns": plan["test"]["shared_prefix_turns"],
            "sampling": plan["test"]["sampling"],
            "goals_sha256": plan["test"]["goals_sha256"],
            "base_model": plan["base_model"],
        }
    )
    invocation_path = output / "invocation.json"
    if invocation_path.is_file():
        _require(json.loads(invocation_path.read_text(encoding="utf-8")) == invocation, "M5 frozen evaluation invocation drift across recovery")
    else:
        atomic_write_json(invocation_path, invocation)
    start_path = output / "evaluation_start.json"
    if start_path.is_file():
        start = _load_hashed(start_path, "m5_webshop_frozen_eval_start_v1")
        _require(start["consumer_git_sha"] == git_sha and start["identity"] == args.identity, "M5 evaluation start identity drift")
    else:
        start = _hashed(
            {
                "schema_version": "m5_webshop_frozen_eval_start_v1",
                "consumer_git_sha": git_sha,
                "identity": args.identity,
                "started_at_unix": time.time(),
            }
        )
        atomic_write_json(start_path, start)
    _append_allocation(
        output / "allocations.jsonl",
        {
            "schema_version": "m5_webshop_frozen_eval_allocation_v1",
            "job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "consumer_git_sha": git_sha,
            "identity": args.identity,
        },
    )
    if spec["kind"] == "raw":
        config = RawVLLMBackendConfig(
            base_model=plan["base_model"]["path"],
            base_model_manifest_sha256=plan["base_model"]["manifest_sha256"],
            base_model_functional_sha256=plan["base_model"]["functional_file_set_sha256"],
            seed=plan["test"]["evaluation_seed"],
        )
        engine = await RawAsyncVLLMGenerationEngine.create(config)
        adapter_sha = plan["base_model"]["manifest_sha256"]
        rollout_sha = semantic_sha = plan["base_model"]["functional_file_set_sha256"]
    else:
        config = M5VLLMBackendConfig(
            base_model=plan["base_model"]["path"],
            adapter_path=str((PROJECT_ROOT / spec["adapter_path"]).resolve()),
            adapter_sha256=spec["adapter_sha256"],
            rollout_adapter_path=str((PROJECT_ROOT / spec["rollout_adapter_path"]).resolve()),
            rollout_adapter_sha256=spec["rollout_adapter_sha256"],
            adapter_semantic_sha256=spec["adapter_semantic_sha256"],
            seed=plan["test"]["evaluation_seed"],
        )
        engine = await AsyncVLLMGenerationEngine.create(config)
        adapter_sha = spec["adapter_sha256"]
        rollout_sha = spec["rollout_adapter_sha256"]
        semantic_sha = spec["adapter_semantic_sha256"]
    lineage = {
        "git_sha": git_sha,
        "protocol_sha256": protocol["sha256"],
        "identity": args.identity,
        "adapter_sha256": adapter_sha,
        "rollout_adapter_sha256": rollout_sha,
        "adapter_semantic_sha256": semantic_sha,
    }
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    telemetry_path = output / "telemetry" / f"generation_{os.environ.get('SLURM_JOB_ID', 'manual')}.csv"
    try:
        with TelemetryRecorder(telemetry_path):
            while True:
                groups = _load_groups(groups_root, task_ids, args.identity)
                completed = {group["group_id"] for group in groups}
                pending = [
                    (index, task_id)
                    for index, task_id in enumerate(task_ids)
                    if f"e{index:04d}" not in completed
                ]
                if not pending:
                    break
                wave = pending[:GROUPS_PER_WAVE]
                await asyncio.gather(
                    *(
                        _attempt_group(
                            engine=engine,
                            loop=loop,
                            event_loop_thread_id=loop_thread,
                            task_id=task_id,
                            task_metadata=metadata[task_id],
                            group_id=f"e{index:04d}",
                            group_index=index,
                            evaluation_seed=plan["test"]["evaluation_seed"],
                            shared_prefix_turns=plan["test"]["shared_prefix_turns"],
                            groups_root=groups_root,
                            attempts_root=attempts_root,
                            lineage=lineage,
                            base_url=args.base_url,
                        )
                        for index, task_id in wave
                    )
                )
    finally:
        engine.shutdown()
    groups = _load_groups(groups_root, task_ids, args.identity)
    _require(len(groups) == 500, "M5 frozen evaluation is missing task groups")
    attempts = _attempt_audit(attempts_root)
    telemetry = _telemetry_audit(sorted((output / "telemetry").glob("generation_*.csv")))
    metrics = summarize_groups(groups)
    gates = {
        "task_count": metrics["task_count"] == plan["success_contract"]["exact_task_count"],
        "trajectory_count": metrics["trajectory_count"] == plan["success_contract"]["exact_trajectory_count"],
        "infrastructure_valid": attempts["infrastructure_valid_fraction"] >= plan["success_contract"]["minimum_infrastructure_valid_attempt_fraction"],
        "gpu_telemetry": telemetry["samples"] > 0,
        "no_training_updates": invocation["training_updates_allowed"] is False and invocation["optimizer_state_loaded"] is False,
    }
    unmet = [name for name, passed in gates.items() if not passed]
    _require(not unmet, f"M5 frozen evaluation gates failed: {unmet}")
    report = _hashed(
        {
            "schema_version": RUN_REPORT_SCHEMA,
            "complete": True,
            "passed": True,
            "formal_evaluation": True,
            "training_updates_allowed": False,
            "optimizer_state_loaded": False,
            "consumer_git_sha": git_sha,
            "producer_training_git_sha": plan["producer_training_git_sha"],
            "protocol_sha256": protocol["sha256"],
            "eval_plan_sha256": plan_bundle["sha256"],
            "authorization_sha256": authorization["sha256"],
            "invocation_path": str(invocation_path),
            "invocation_file_sha256": sha256_file(invocation_path),
            "identity": args.identity,
            "identity_spec": spec,
            "group_content_sha256": {group["group_id"]: group["content_sha256"] for group in groups},
            "attempts": attempts,
            "metrics": metrics,
            "cost": {
                "committed_generated_action_tokens": metrics["generated_action_tokens"],
                "all_attempt_generated_action_tokens": attempts["all_attempt_generated_action_tokens"],
                "environment_steps": metrics["environment_steps"],
                "sum_trajectory_elapsed_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for group in groups for item in group["trajectories"]),
                "end_to_end_wall_seconds": time.time() - float(start["started_at_unix"]),
                "evaluation_start_path": str(start_path),
                "evaluation_start_file_sha256": sha256_file(start_path),
            },
            "gpu_telemetry": telemetry,
            "gates": gates,
            "unmet_gates": unmet,
        }
    )
    atomic_write_json(report_path, report)
    return validate_eval_report(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, choices=IDENTITIES)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--eval-plan", type=Path, default=PROJECT_ROOT / "data" / "m5_webshop_frozen_eval_plan_v1.json")
    parser.add_argument("--authorization", type=Path, default=PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "readiness" / "frozen_eval_authorization_v1.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        print(json.dumps(asyncio.run(run(args)), indent=2, sort_keys=True))
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        output = args.output_dir or evaluation_run_root(args.identity)
        output.mkdir(parents=True, exist_ok=True)
        failure = _hashed(
            {
                "schema_version": "m5_webshop_frozen_eval_failure_v1",
                "complete": True,
                "passed": False,
                "formal_evaluation": True,
                "training_updates_allowed": False,
                "consumer_git_sha": _git_sha(),
                "identity": args.identity,
                "exception_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "traceback": traceback.format_exc()[-16000:],
            }
        )
        atomic_write_json(output / "run_failure.json", failure)
        raise


if __name__ == "__main__":
    main()
