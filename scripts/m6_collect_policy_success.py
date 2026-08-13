#!/usr/bin/env python3
"""Collect or resume M6 Raw success data and shared K4/K8 rollouts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.adapter_view import build_vllm_adapter_view  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    directory_sha256,
    sha256_json,
)
from miniwebwork.long_horizon_rl.model_manifest import validate_base_model_manifest  # noqa: E402
from miniwebwork.long_horizon_rl.vllm_backend import (  # noqa: E402
    AsyncVLLMGenerationEngine,
    RawAsyncVLLMGenerationEngine,
    RawVLLMBackendConfig,
    RolloutRequestContext,
    ThreadsafeVLLMBackend,
)
from miniwebwork.m6_posttraining_protocol import (  # noqa: E402
    load_protocol,
    validate_split_lock,
)
from miniwebwork.m6_pilot import validate_artifact_git_compatibility  # noqa: E402
from miniwebwork.model_agent.agent_loop import run_model_episode  # noqa: E402
from miniwebwork.webshop_rl.agent import QwenWebShopAgent  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import (  # noqa: E402
    build_committed_group,
    trajectory_from_episode,
    validate_committed_group,
)
from miniwebwork.webshop_rl.online_training import MAX_NEW_TOKENS, M5VLLMBackendConfig  # noqa: E402
from miniwebwork.webshop_rl.verifier_td import annotate_episode_with_verifier  # noqa: E402

ATTEMPT_SCHEMA = "m6_rollout_attempt_v1"
ATTEMPT_START_SCHEMA = "m6_rollout_attempt_start_v1"
REPORT_SCHEMA = "m6_rollout_collection_report_v1"
MAX_INFRASTRUCTURE_ATTEMPTS = 4
LINEAGE_FIELDS = (
    "adapter_sha256",
    "rollout_adapter_sha256",
    "adapter_semantic_sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _atomic_episode(path: Path, episode: Mapping[str, Any]) -> None:
    value = dict(episode)
    value["content_sha256"] = _self_hash(value)
    atomic_write_json(path, value)


def _attempt_indices(attempts_root: Path, group_id: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(group_id)}\.a(\d+)(?:\.started)?\.json$")
    indices: set[int] = set()
    for path in attempts_root.glob(f"{group_id}.a*.json"):
        match = pattern.fullmatch(path.name)
        _require(match is not None, f"M6 attempt filename drift: {path.name}")
        value = _load_json(path)
        expected = dict(value)
        observed = expected.pop("content_sha256", None)
        _require(observed == sha256_json(expected), f"M6 attempt artifact self-hash drift: {path.name}")
        _require(value.get("group_id") == group_id, f"M6 attempt artifact group drift: {path.name}")
        _require(int(value.get("attempt_index", -1)) == int(match.group(1)), f"M6 attempt index drift: {path.name}")
        indices.add(int(match.group(1)))
    return sorted(indices)


class DurableTokenLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, payload: Mapping[str, Any]) -> None:
        line = canonical_json_bytes(dict(payload)) + b"\n"
        with self._lock, self.path.open("ab", buffering=0) as handle:
            handle.write(line)
            os.fsync(handle.fileno())

    def audit(self) -> dict[str, Any]:
        rows = []
        if self.path.is_file():
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"M6 token ledger line {line_number} is corrupt") from exc
                _require(isinstance(row, Mapping), "M6 token ledger row is malformed")
                rows.append(row)
        return {
            "record_count": len(rows),
            "generated_action_tokens": sum(int(row["generated_action_tokens"]) for row in rows),
            "sha256": sha256_json(rows) if rows else hashlib.sha256(b"").hexdigest(),
        }


async def _create_identity(
    *,
    base_model: Path,
    adapter: Path | None,
    output: Path,
    seed: int,
) -> tuple[RawAsyncVLLMGenerationEngine | AsyncVLLMGenerationEngine, dict[str, str]]:
    manifest = validate_base_model_manifest(expected_base_model=base_model, verify_files=True)
    if adapter is None:
        engine = await RawAsyncVLLMGenerationEngine.create(
            RawVLLMBackendConfig(
                base_model=str(base_model),
                base_model_manifest_sha256=manifest["sha256"],
                base_model_functional_sha256=manifest["payload"]["functional_file_set_sha256"],
                seed=seed,
            )
        )
        lineage = {
            "adapter_sha256": manifest["sha256"],
            "rollout_adapter_sha256": manifest["payload"]["functional_file_set_sha256"],
            "adapter_semantic_sha256": manifest["payload"]["functional_file_set_sha256"],
        }
    else:
        canonical = adapter.expanduser().resolve()
        view_audit = build_vllm_adapter_view(
            source_adapter=canonical,
            destination=output / "rollout_adapter",
            base_model=base_model,
        )
        lineage = {
            "adapter_sha256": directory_sha256(canonical),
            "rollout_adapter_sha256": view_audit["view_directory_sha256"],
            "adapter_semantic_sha256": view_audit["semantic_tensor_sha256"],
        }
        engine = await AsyncVLLMGenerationEngine.create(
            M5VLLMBackendConfig(
                base_model=str(base_model),
                adapter_path=str(canonical),
                rollout_adapter_path=str(output / "rollout_adapter"),
                seed=seed,
                **lineage,
            )
        )
    return engine, lineage


def _task_roster(path: Path) -> tuple[list[str], str, str, str, str]:
    value = _load_json(path)
    _require(isinstance(value, Mapping), "M6 task roster root is malformed")
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 task roster self-hash drift")
    _require(value.get("schema_version") == "m6_rl_curriculum_v1", "M6 task roster schema drift")
    task_ids = value.get("task_ids")
    _require(
        isinstance(task_ids, list)
        and task_ids
        and len(task_ids) == len(set(task_ids))
        and all(isinstance(item, str) and item for item in task_ids),
        "M6 task roster is empty or duplicated",
    )
    split_hash = value.get("source_split_lock_content_sha256")
    _require(isinstance(split_hash, str) and split_hash, "M6 task roster split binding is missing")
    protocol_hash = value.get("protocol_sha256")
    git_sha = value.get("git_sha")
    _require(isinstance(protocol_hash, str) and protocol_hash, "M6 task roster protocol binding is missing")
    _require(isinstance(git_sha, str) and git_sha, "M6 task roster Git binding is missing")
    return list(task_ids), str(observed), split_hash, protocol_hash, git_sha


def validate_diagnostic_evaluation_contract(args: argparse.Namespace) -> None:
    """Keep phase-one seen-task evaluation outside every training path."""

    _require(args.role == "mini_train" and args.task_roster is not None, "M6 diagnostic evaluation roster drift")
    _require(args.adapter is not None, "M6 diagnostic evaluation requires an SFT or RL adapter")
    _require(args.max_model_turns == 6 and args.max_environment_steps == 6, "M6 seen-task diagnostic horizon drift")
    _require(args.maximum_action_tokens is None, "M6 diagnostic evaluation cannot claim a training token budget")


def _validate_group_run_contract(
    group: Mapping[str, Any],
    *,
    task_id: str,
    group_id: str,
    k: int,
    git_sha: str,
    protocol_sha256: str,
    lineage: Mapping[str, str],
    training_updates_allowed: bool,
) -> dict[str, Any]:
    """Reject a recovered group produced by any other policy or run contract."""

    value = validate_committed_group(group, require_k=k)
    _require(value.get("task_id") == task_id, f"M6 recovered group task drift: {group_id}")
    _require(value.get("group_id") == group_id, f"M6 recovered group id drift: {group_id}")
    _require(value.get("git_sha") == git_sha, f"M6 recovered group Git drift: {group_id}")
    _require(
        value.get("protocol_sha256") == protocol_sha256,
        f"M6 recovered group protocol drift: {group_id}",
    )
    _require(
        value.get("training_updates_allowed") is training_updates_allowed,
        f"M6 recovered group training-purpose drift: {group_id}",
    )
    for field in LINEAGE_FIELDS:
        _require(
            value.get(field) == lineage[field],
            f"M6 recovered group policy lineage drift ({field}): {group_id}",
        )
    return value


async def _one_group(
    *,
    engine: RawAsyncVLLMGenerationEngine | AsyncVLLMGenerationEngine,
    loop: asyncio.AbstractEventLoop,
    loop_thread_id: int,
    task_id: str,
    goal: Mapping[str, Any],
    group_index: int,
    k: int,
    seed: int,
    iteration_index: int,
    groups_root: Path,
    episodes_root: Path,
    attempts_root: Path,
    lineage: Mapping[str, str],
    ledger: DurableTokenLedger,
    protocol_sha256: str,
    git_sha: str,
    base_url: str,
    training_updates_allowed: bool,
    max_model_turns: int,
    max_environment_steps: int,
) -> dict[str, Any] | None:
    group_id = f"g{group_index:04d}"
    destination = groups_root / f"{group_id}.json"
    if destination.is_file():
        return validate_committed_group(_load_json(destination), require_k=k)
    prior = _attempt_indices(attempts_root, group_id)
    attempt_index = max(prior, default=-1) + 1
    _require(attempt_index < MAX_INFRASTRUCTURE_ATTEMPTS, f"M6 group exhausted infrastructure attempts: {group_id}")
    attempt_start = {
        "schema_version": ATTEMPT_START_SCHEMA,
        "task_id": task_id,
        "group_id": group_id,
        "attempt_index": attempt_index,
        "K": k,
        "git_sha": git_sha,
        "protocol_sha256": protocol_sha256,
        "training_updates_allowed": training_updates_allowed,
        "policy_lineage": dict(lineage),
    }
    attempt_start["content_sha256"] = _self_hash(attempt_start)
    atomic_write_json(attempts_root / f"{group_id}.a{attempt_index}.started.json", attempt_start)

    async def one(rollout_index: int) -> dict[str, Any]:
        context = RolloutRequestContext(
            run_seed=seed,
            iteration_index=iteration_index,
            group_id=group_id,
            attempt_index=attempt_index,
            trajectory_id=f"{group_id}.a{attempt_index}.r{rollout_index}",
            rollout_index=rollout_index,
            shared_prefix_turns=0,
            sampling_attempt_index=0,
        )
        backend = ThreadsafeVLLMBackend(
            engine=engine,
            event_loop=loop,
            context=context,
            timeout_seconds=900,
            event_loop_thread_id=loop_thread_id,
        )
        environment = WebShopHTTPEnvironment(base_url=base_url, split="train", timeout_seconds=120)
        agent = QwenWebShopAgent(backend)
        def charge(turn: Mapping[str, Any]) -> None:
            generated_ids = list(turn.get("generated_token_ids", []))
            ledger.append(
                {
                    "schema_version": "m6_generated_turn_ledger_v1",
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
            episode = await asyncio.to_thread(
                run_model_episode,
                task_id,
                environment,
                agent,
                max_model_turns,
                max_environment_steps,
                charge,
                None,
            )
            if episode.get("rollout_valid") is True:
                episode = annotate_episode_with_verifier(episode, goal=goal)
            return episode
        finally:
            environment.close()

    episodes = await asyncio.gather(*(one(index) for index in range(k)))
    attempt_episode_root = attempts_root / "episodes"
    attempt_episode_root.mkdir(exist_ok=True)
    for rollout_index, episode in enumerate(episodes):
        _atomic_episode(
            attempt_episode_root / f"{group_id}.a{attempt_index}.r{rollout_index}.json",
            episode,
        )
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "task_id": task_id,
        "group_id": group_id,
        "attempt_index": attempt_index,
        "K": k,
        "rollout_valid": [episode.get("rollout_valid") for episode in episodes],
        "task_scores": [episode.get("task_score") for episode in episodes],
        "generated_action_tokens": sum(
            len(turn.get("generated_token_ids", []))
            for episode in episodes
            for turn in episode.get("turns", [])
        ),
    }
    attempt["content_sha256"] = _self_hash(attempt)
    atomic_write_json(attempts_root / f"{group_id}.a{attempt_index}.json", attempt)
    if not all(episode.get("rollout_valid") is True for episode in episodes):
        return None
    trajectories = []
    for rollout_index, episode in enumerate(episodes):
        trajectory_id = f"{group_id}.a{attempt_index}.r{rollout_index}"
        episode["trajectory_id"] = trajectory_id
        _atomic_episode(episodes_root / f"{trajectory_id}.json", episode)
        trajectories.append(
            trajectory_from_episode(
                episode,
                trajectory_id=trajectory_id,
                rollout_index=rollout_index,
                **lineage,
            )
        )
    group = build_committed_group(
        task_id=task_id,
        group_id=group_id,
        attempt_index=attempt_index,
        trajectories=trajectories,
        expected_k=k,
        git_sha=git_sha,
        protocol_sha256=protocol_sha256,
        training_updates_allowed=training_updates_allowed,
        **lineage,
    )
    atomic_write_json(destination, group)
    return group


async def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_bundle = load_protocol()
    protocol = protocol_bundle["payload"]
    git_sha = _git_sha()
    _require(git_sha == protocol_bundle["git_sha"], "M6 rollout Git/protocol drift")
    split_lock = validate_split_lock(_load_json(args.split_lock))
    _require(split_lock["protocol_sha256"] == protocol_bundle["sha256"], "M6 rollout split/protocol drift")
    task_ids = list(split_lock["roles"][args.role]["task_ids"])
    roster_sha256 = None
    roster_producer_git_sha = None
    if args.task_roster is not None:
        selected, roster_sha256, roster_split_sha256, roster_protocol_sha256, roster_git_sha = _task_roster(args.task_roster)
        _require(
            roster_split_sha256 == split_lock["content_sha256"],
            "M6 task roster was frozen from a different split lock",
        )
        _require(set(selected) <= set(task_ids), "M6 task roster escapes the frozen role")
        _require(roster_protocol_sha256 == protocol_bundle["sha256"], "M6 task roster protocol drift")
        roster_producer_git_sha = validate_artifact_git_compatibility(
            artifact_name="RL curriculum",
            producer_git_sha=roster_git_sha,
            consumer_git_sha=git_sha,
            explicitly_authorized_producer_git_sha=args.task_roster_producer_git_sha,
        )
        task_ids = selected
    else:
        _require(
            args.task_roster_producer_git_sha is None,
            "M6 task-roster producer Git was provided without a task roster",
        )
    _require(args.task_offset >= 0, "M6 task offset must be non-negative")
    task_ids = task_ids[args.task_offset :]
    if args.maximum_tasks is not None:
        _require(args.maximum_tasks > 0, "M6 maximum tasks must be positive")
        task_ids = task_ids[: args.maximum_tasks]
    _require(task_ids, "M6 selected task roster is empty")
    expected_k = int(protocol["mini"]["evaluation_K"] if args.mode == "evaluation" else protocol["rl"]["group_size"])
    _require(args.k == expected_k, "M6 mode/K contract drift")
    training_updates_allowed = args.mode == "rl_collection"
    if args.mode == "raw_collection":
        _require(args.role == "mini_train", "M6 Raw collection must use mini_train")
        _require(args.adapter is None and args.task_roster is None and args.task_offset == 0, "M6 Raw collection policy/roster drift")
        _require(len(task_ids) == int(protocol["split"]["mini_train_tasks"]), "M6 Raw collection task count drift")
    elif args.mode == "evaluation":
        _require(args.role in {"mini_dev", "formal_dev"}, "M6 evaluation role drift")
        _require(args.task_roster is None and args.task_offset == 0, "M6 evaluation roster drift")
        expected_tasks = int(protocol["split"][f"{args.role}_tasks"])
        _require(len(task_ids) == expected_tasks, "M6 evaluation task count drift")
    elif args.mode == "diagnostic_evaluation":
        validate_diagnostic_evaluation_contract(args)
    else:
        _require(args.role == "mini_train" and args.task_roster is not None, "M6 RL collection curriculum drift")
    _require(not training_updates_allowed or args.adapter is not None, "M6 RL collection requires an adapter")
    if training_updates_allowed:
        _require(args.maximum_tasks == int(protocol["mini"]["rl_collection_groups_per_iteration"]), "M6 RL group budget drift")
        _require(args.max_model_turns == int(protocol["mini"]["rl_collection_max_model_turns"]), "M6 RL model-turn cap drift")
        _require(args.max_environment_steps == int(protocol["mini"]["rl_collection_max_environment_steps"]), "M6 RL environment-step cap drift")
        _require(args.maximum_action_tokens is not None and args.maximum_action_tokens > 0, "M6 RL action-token cap missing")
        _require(
            args.maximum_action_tokens <= int(protocol["mini"]["rl_collection_action_token_cap_per_iteration"]),
            "M6 RL iteration action-token cap drift",
        )
    else:
        _require(args.maximum_action_tokens is None, "M6 non-RL rollout cannot claim an RL token cap")
    output = args.output_dir.expanduser().resolve()
    groups_root, episodes_root, attempts_root = output / "groups", output / "episodes", output / "attempts"
    for path in (groups_root, episodes_root, attempts_root):
        path.mkdir(parents=True, exist_ok=True)
    ledger = DurableTokenLedger(output / "generated_turn_ledger.jsonl")
    goals = _load_json(args.goals)
    _require(sha256_json(goals) == split_lock["goals_canonical_sha256"], "M6 rollout goals/split drift")
    goal_map = {f"webshop_goal_{int(item['goal_index']):05d}": item for item in goals}
    _require(all(task_id in goal_map for task_id in task_ids), "M6 rollout goal roster is incomplete")
    # Reconstruct already charged budget before allocating GPU memory.  A
    # timeout after generation but before group commit must not receive a new
    # budget simply because this process restarted.
    if args.maximum_action_tokens is not None:
        _require(
            ledger.audit()["generated_action_tokens"] <= args.maximum_action_tokens,
            "M6 persisted token ledger already exceeds the action-token cap",
        )
    engine, lineage = await _create_identity(
        base_model=args.base_model.expanduser().resolve(),
        adapter=args.adapter,
        output=output,
        seed=args.seed,
    )
    loop = asyncio.get_running_loop()
    loop_thread_id = threading.get_ident()
    try:
        invocation = {
            "schema_version": "m6_rollout_invocation_v1",
            "development_only": True,
            "formal_training": False,
            "mode": args.mode,
            "role": args.role,
            "K": args.k,
            "task_count": len(task_ids),
            "task_order_sha256": sha256_json(task_ids),
            "task_roster_content_sha256": roster_sha256,
            "task_roster_producer_git_sha": roster_producer_git_sha,
            "task_roster_consumer_git_sha": git_sha if roster_sha256 is not None else None,
            "task_offset": args.task_offset,
            "seed": args.seed,
            "iteration_index": args.iteration_index,
            "max_model_turns": args.max_model_turns,
            "max_environment_steps": args.max_environment_steps,
            "maximum_action_tokens": args.maximum_action_tokens,
            "training_updates_allowed": training_updates_allowed,
            "protocol_sha256": protocol_bundle["sha256"],
            "git_sha": git_sha,
            "split_lock_content_sha256": split_lock["content_sha256"],
            "base_model": str(args.base_model.expanduser().resolve()),
            "adapter": str(args.adapter.expanduser().resolve()) if args.adapter else None,
            "policy_lineage": dict(lineage),
        }
        invocation["content_sha256"] = _self_hash(invocation)
        invocation_path = output / "invocation.json"
        if invocation_path.is_file():
            _require(_load_json(invocation_path) == invocation, "M6 rollout invocation changed across recovery")
        else:
            atomic_write_json(invocation_path, invocation)

        # A 24-hour successor may only reuse groups from this exact policy,
        # split, task order, protocol and purpose.  This prevents a partially
        # completed directory from silently mixing two adapter generations.
        for index, task_id in enumerate(task_ids):
            existing = groups_root / f"g{index:04d}.json"
            if existing.is_file():
                _validate_group_run_contract(
                    _load_json(existing),
                    task_id=task_id,
                    group_id=f"g{index:04d}",
                    k=args.k,
                    git_sha=git_sha,
                    protocol_sha256=protocol_bundle["sha256"],
                    lineage=lineage,
                    training_updates_allowed=training_updates_allowed,
                )
        while True:
            pending = [
                (index, task_id)
                for index, task_id in enumerate(task_ids)
                if not (groups_root / f"g{index:04d}.json").is_file()
            ]
            if not pending:
                break
            wave = pending[: args.concurrent_groups]
            if args.maximum_action_tokens is not None:
                # Reserve the mathematical maximum before starting a complete
                # K8 attempt.  This counts every generated token, including
                # infrastructure-invalid attempts, and therefore cannot
                # overshoot the frozen cap after generation has already run.
                worst_case_attempt = len(wave) * args.k * args.max_model_turns * MAX_NEW_TOKENS
                _require(
                    ledger.audit()["generated_action_tokens"] + worst_case_attempt
                    <= args.maximum_action_tokens,
                    "M6 action-token budget cannot reserve another complete rollout attempt",
                )
            results = await asyncio.gather(
                *(
                    _one_group(
                        engine=engine,
                        loop=loop,
                        loop_thread_id=loop_thread_id,
                        task_id=task_id,
                        goal=goal_map[task_id],
                        group_index=index,
                        k=args.k,
                        seed=args.seed,
                        iteration_index=args.iteration_index,
                        groups_root=groups_root,
                        episodes_root=episodes_root,
                        attempts_root=attempts_root,
                        lineage=lineage,
                        ledger=ledger,
                        protocol_sha256=protocol_bundle["sha256"],
                        git_sha=git_sha,
                        base_url=args.base_url,
                        training_updates_allowed=training_updates_allowed,
                        max_model_turns=args.max_model_turns,
                        max_environment_steps=args.max_environment_steps,
                    )
                    for index, task_id in wave
                )
            )
            # An invalid infrastructure attempt is retried on the next wave;
            # algorithm failures are valid completed groups and never retried.
            _require(any(item is not None for item in results) or all(
                max(_attempt_indices(attempts_root, f"g{index:04d}"), default=-1) + 1
                < MAX_INFRASTRUCTURE_ATTEMPTS
                for index, _ in wave
            ), "M6 rollout wave exhausted infrastructure retries")
    finally:
        engine.shutdown()
    groups = [
        _validate_group_run_contract(
            _load_json(groups_root / f"g{index:04d}.json"),
            task_id=task_id,
            group_id=f"g{index:04d}",
            k=args.k,
            git_sha=git_sha,
            protocol_sha256=protocol_bundle["sha256"],
            lineage=lineage,
            training_updates_allowed=training_updates_allowed,
        )
        for index, task_id in enumerate(task_ids)
    ]
    trajectories = [trajectory for group in groups for trajectory in group["trajectories"]]
    ledger_audit = ledger.audit()
    _require(
        ledger_audit["generated_action_tokens"] >= sum(group["generated_action_tokens"] for group in groups),
        "M6 token ledger excludes committed group tokens",
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "complete": True,
        "development_only": True,
        "mode": args.mode,
        "role": args.role,
        "task_count": len(groups),
        "trajectory_count": len(trajectories),
        "K": args.k,
        "strict_success_task_count": len({group["task_id"] for group in groups if any(item["success"] for item in group["trajectories"])}),
        "strict_success_trajectory_count": sum(item["success"] for item in trajectories),
        "mixed_strict_reward_group_count": sum(
            len({item["binary_reward"] for item in group["trajectories"]}) > 1 for group in groups
        ),
        "generated_action_tokens": sum(group["generated_action_tokens"] for group in groups),
        "all_attempt_generated_action_tokens": ledger_audit["generated_action_tokens"],
        "maximum_action_tokens": args.maximum_action_tokens,
        "training_updates_allowed": training_updates_allowed,
        "action_token_budget_respected": (
            args.maximum_action_tokens is None
            or ledger_audit["generated_action_tokens"] <= args.maximum_action_tokens
        ),
        "token_ledger": ledger_audit,
        "git_sha": git_sha,
        "protocol_sha256": protocol_bundle["sha256"],
        "split_lock_content_sha256": split_lock["content_sha256"],
        "task_order_sha256": sha256_json(task_ids),
        "task_roster_content_sha256": roster_sha256,
        "task_roster_producer_git_sha": roster_producer_git_sha,
        "task_roster_consumer_git_sha": git_sha if roster_sha256 is not None else None,
        "invocation_content_sha256": invocation["content_sha256"],
        "policy_lineage": dict(lineage),
        "group_content_sha256": [group["content_sha256"] for group in groups],
        "elapsed_seconds": time.time() - args.started_at,
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(output / "collection_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw_collection", "evaluation", "diagnostic_evaluation", "rl_collection"), required=True)
    parser.add_argument("--role", choices=("mini_train", "mini_dev", "formal_dev", "train"), required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--k", type=int, choices=(4, 8), required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--iteration-index", type=int, default=0)
    parser.add_argument("--maximum-tasks", type=int)
    parser.add_argument("--task-roster", type=Path)
    parser.add_argument("--task-roster-producer-git-sha")
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--max-model-turns", type=int, default=18)
    parser.add_argument("--max-environment-steps", type=int, default=15)
    parser.add_argument("--maximum-action-tokens", type=int)
    parser.add_argument("--concurrent-groups", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    args = parser.parse_args()
    _require(1 <= args.concurrent_groups <= 4, "M6 group concurrency must be in [1,4]")
    _require(args.max_model_turns > 0 and args.max_environment_steps > 0, "M6 turn caps must be positive")
    args.started_at = time.time()
    return args


def main() -> None:
    report = asyncio.run(run(parse_args()))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
