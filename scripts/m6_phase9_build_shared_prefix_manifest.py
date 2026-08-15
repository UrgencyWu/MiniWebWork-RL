#!/usr/bin/env python3
"""Freeze an eight-task, public shared-prefix suffix-smoke manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m6_phase4_build_data_rosters import _bucket  # noqa: E402
from m6_phase7_terminal_contrast_diagnostic import (  # noqa: E402
    FORBIDDEN_PATH_PARTS,
    _load_hashed,
)
from m6_phase8_prefix_reset_feasibility import _strict_prefix_candidate  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402

TASK_COUNT = 8
K = 4
SELECTION_SEED = 20260833
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _safe_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    lowered = {part.casefold() for part in resolved.parts}
    _require(not (lowered & FORBIDDEN_PATH_PARTS), f"Phase9 {label} touches promotion/holdout")
    return resolved


def _git_sha() -> str:
    value = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase9 repository Git SHA drift")
    return value


def _load_sft_task_ids(paths: Sequence[Path]) -> set[str]:
    output: set[str] = set()
    for raw in paths:
        path = _safe_path(raw, label="SFT input")
        _require(path.is_file(), f"Phase9 SFT input missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            _require(isinstance(task_id, str) and task_id, "Phase9 SFT row lacks task_id")
            output.add(task_id)
    return output


def _rank(task_id: str) -> bytes:
    return hashlib.sha256(f"m6-phase9-shared-prefix-v1|{SELECTION_SEED}|{task_id}".encode()).digest()


def _select_stratified(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    goal_map: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Take one deterministic task per bucket before taking a second."""

    _require(len(candidates) >= TASK_COUNT, "Phase9 has fewer than eight branchable tasks")
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for task_id in candidates:
        _require(task_id in goal_map, f"Phase9 goal missing: {task_id}")
        by_bucket[_bucket(goal_map[task_id])].append(task_id)
    for values in by_bucket.values():
        values.sort(key=lambda task_id: (_rank(task_id), task_id))
    selected: list[str] = []
    depth = 0
    while len(selected) < TASK_COUNT:
        wave = [
            values[depth]
            for _, values in sorted(by_bucket.items())
            if depth < len(values)
        ]
        wave.sort(key=lambda task_id: (_rank(task_id), task_id))
        _require(bool(wave), "Phase9 stratified selection exhausted unexpectedly")
        selected.extend(wave[: TASK_COUNT - len(selected)])
        depth += 1
    _require(len(selected) == TASK_COUNT and len(set(selected)) == TASK_COUNT, "Phase9 selection drift")
    return selected


def _prefix_binding(
    *,
    root: Path,
    report: Mapping[str, Any],
    group: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    rollout_index: int,
    prefix_turn_count: int,
    public_option_action_count: int,
) -> dict[str, Any]:
    turns = trajectory["turns"][:prefix_turn_count]
    commands = [str((turn.get("action") or {}).get("command", "")).strip() for turn in turns]
    _require(all(commands), "Phase9 source prefix contains an empty action")
    token_evidence = [
        {
            "prompt_token_ids": list(turn["prompt_token_ids"]),
            "generated_token_ids": list(turn["generated_token_ids"]),
            "behavior_logprobs": list(turn["behavior_logprobs"]),
            "sampling_logprobs": list(turn["sampling_logprobs"]),
        }
        for turn in turns
    ]
    return {
        "task_id": str(group["task_id"]),
        "source_collection_root": str(root),
        "source_collection_report_content_sha256": report["content_sha256"],
        "source_group_id": str(group["group_id"]),
        "source_group_content_sha256": group["content_sha256"],
        "source_rollout_index": rollout_index,
        "prefix_turn_count": prefix_turn_count,
        "prefix_action_sequence_sha256": sha256_json(commands),
        "prefix_token_evidence_sha256": sha256_json(token_evidence),
        "public_option_action_count": public_option_action_count,
        "source_policy_lineage": dict(report["policy_lineage"]),
        "source_git_sha": str(report["git_sha"]),
    }


def _load_candidates(
    roots: Sequence[Path],
    *,
    allowed_task_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    best: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    common_lineage: dict[str, str] | None = None
    for raw_root in roots:
        root = _safe_path(raw_root, label="prescan input")
        report = _load_hashed(root / "collection_report.json")
        _require(
            report.get("complete") is True
            and report.get("development_only") is True
            and report.get("mode") == "phase4_data_synthesis"
            and report.get("K") == K
            and report.get("training_updates_allowed") is False,
            f"Phase9 source collection contract drift: {root}",
        )
        lineage = dict(report["policy_lineage"])
        if common_lineage is None:
            common_lineage = lineage
        _require(lineage == common_lineage, "Phase9 source SFT policy lineage drift")
        paths = sorted((root / "groups").glob("g*.json"))
        _require(len(paths) == report["task_count"], f"Phase9 source group count drift: {root}")
        groups = [validate_committed_group(_load_hashed(path), require_k=K) for path in paths]
        _require(
            report["group_content_sha256"] == [group["content_sha256"] for group in groups],
            f"Phase9 source group/report binding drift: {root}",
        )
        bindings.append({
            "collection_root": str(root),
            "collection_report_content_sha256": report["content_sha256"],
            "policy_lineage": lineage,
        })
        for group in groups:
            task_id = str(group["task_id"])
            if task_id not in allowed_task_ids:
                continue
            for rollout_index, trajectory in enumerate(group["trajectories"]):
                candidate = _strict_prefix_candidate(trajectory)
                if candidate is None or candidate["public_option_action_count"] < 2:
                    continue
                prefix_turn_count = int(candidate["prefix_action_count"])
                binding = _prefix_binding(
                    root=root,
                    report=report,
                    group=group,
                    trajectory=trajectory,
                    rollout_index=rollout_index,
                    prefix_turn_count=prefix_turn_count,
                    public_option_action_count=int(candidate["public_option_action_count"]),
                )
                previous = best.get(task_id)
                key = (
                    prefix_turn_count,
                    -int(candidate["public_option_action_count"]),
                    binding["prefix_action_sequence_sha256"],
                    binding["source_group_content_sha256"],
                    rollout_index,
                )
                if previous is None or key < tuple(previous["selection_key"]):
                    binding["selection_key"] = list(key)
                    best[task_id] = binding
    _require(common_lineage is not None, "Phase9 source collection set is empty")
    for value in best.values():
        value.pop("selection_key", None)
    return best, bindings, common_lineage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prescan-root", type=Path, action="append", required=True)
    parser.add_argument("--phase8-report", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--sft-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--roster-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    git_sha = _git_sha()
    _require(args.producer_git_sha == git_sha, "Phase9 producer Git drift")
    protocol = load_protocol()
    _require(protocol["git_sha"] == git_sha, "Phase9 protocol/repository Git drift")
    phase8 = _load_hashed(_safe_path(args.phase8_report, label="Phase8 report"))
    _require(
        phase8.get("schema_version") == "m6_phase8_prefix_reset_feasibility_v1"
        and phase8.get("decision", {}).get("prefix_reset_feasible") is True
        and phase8.get("training_performed") is False
        and phase8.get("target_asin_or_hidden_answer_used") is False,
        "Phase9 Phase8 prerequisite drift",
    )
    branchable = set(phase8["branchable_task_ids"])
    _require(len(branchable) == phase8["branchable_task_count"], "Phase9 Phase8 branchable roster drift")

    split = validate_split_lock(json.loads(_safe_path(args.split_lock, label="split lock").read_text()))
    _require(split["protocol_sha256"] == protocol["sha256"], "Phase9 split/protocol drift")
    goals = json.loads(_safe_path(args.goals, label="goals").read_text())
    _require(isinstance(goals, list) and sha256_json(goals) == split["goals_canonical_sha256"], "Phase9 goals drift")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    sft_ids = _load_sft_task_ids(args.sft_jsonl)
    _require(branchable <= set(split["roles"]["train"]["task_ids"]), "Phase9 branchable task escaped train role")
    _require(not (branchable & sft_ids), "Phase9 branchable task overlaps SFT")

    candidates, source_bindings, policy_lineage = _load_candidates(args.prescan_root, allowed_task_ids=branchable)
    _require(set(candidates) == branchable, "Phase9 could not reproduce the Phase8 branchable set")
    selected = _select_stratified(candidates, goal_map=goal_map)
    roster = {
        "schema_version": "m6_rl_curriculum_v1",
        "study_id": protocol["payload"]["study_id"],
        "development_only": True,
        "formal_training": False,
        "purpose": "phase9_shared_public_prefix_k4_suffix_smoke",
        "selection": "phase8_branchable_bucket_then_seeded_hash_v1",
        "selection_seed": SELECTION_SEED,
        "source_role": "train",
        "source_split_lock_content_sha256": split["content_sha256"],
        "protocol_sha256": protocol["sha256"],
        "git_sha": git_sha,
        "source_phase8_report_content_sha256": phase8["content_sha256"],
        "sft_overlap_count": 0,
        "nontrain_role_overlap_count": 0,
        "task_count": TASK_COUNT,
        "task_ids": selected,
        "task_order_sha256": sha256_json(selected),
        "selected_bucket_counts": dict(sorted(Counter(_bucket(goal_map[item]) for item in selected).items())),
        "smoke_contract": {
            "training_performed": False,
            "optimizer_steps": 0,
            "policy_identity": "frozen_sft_adapter",
            "K": K,
            "max_model_turns": 18,
            "max_environment_steps": 15,
            "rollout_seed": SELECTION_SEED,
            "shared_prefix_per_task": True,
            "future_policy_loss_scope": "suffix_tokens_only",
        },
    }
    roster["content_sha256"] = _self_hash(roster)
    manifest = {
        "schema_version": "m6_phase9_shared_prefix_manifest_v1",
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "public_fields_only": True,
        "target_asin_or_hidden_answer_used": False,
        "post_action_internal_state_used": False,
        "producer_git_sha": git_sha,
        "protocol_sha256": protocol["sha256"],
        "split_lock_content_sha256": split["content_sha256"],
        "phase8_report_content_sha256": phase8["content_sha256"],
        "task_roster_content_sha256": roster["content_sha256"],
        "task_order_sha256": roster["task_order_sha256"],
        "selection_seed": SELECTION_SEED,
        "K": K,
        "task_count": TASK_COUNT,
        "policy_lineage": policy_lineage,
        "source_bindings": source_bindings,
        "tasks": [candidates[task_id] for task_id in selected],
        "future_policy_loss_contract": {
            "prefix_tokens_eligible": False,
            "suffix_starts_after_shared_prefix_turns": True,
        },
        "frozen_gates": {
            "minimum_exact_replay_tasks": 6,
            "minimum_mixed_same_item_tasks": 4,
            "minimum_option_contrast_tasks": 3,
        },
    }
    manifest["content_sha256"] = _self_hash(manifest)
    roster_output = _safe_path(args.roster_output, label="roster output")
    manifest_output = _safe_path(args.manifest_output, label="manifest output")
    _require(not roster_output.exists() and not manifest_output.exists(), "Phase9 output already exists")
    roster_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(roster_output, roster)
    atomic_write_json(manifest_output, manifest)
    print(json.dumps({"roster": roster, "manifest": manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
