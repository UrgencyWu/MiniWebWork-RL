#!/usr/bin/env python3
"""Select 40 SFT-disjoint mixed tasks for the minimal Phase4 online run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402

TARGET_TASKS = 40
SELECTION_SEED = 20260828


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _constraint_count(goal: Mapping[str, Any]) -> int:
    attributes = goal.get("instruction_attributes") or goal.get("attributes") or []
    options = goal.get("goal_options") or []
    return len(attributes) + len(options) + int(isinstance(goal.get("price_upper"), (int, float)))


def _bucket(goal: Mapping[str, Any]) -> str:
    category = str(goal.get("category") or "unknown").strip().casefold()
    constraints = _constraint_count(goal)
    return f"{category}|constraints_{min(constraints, 6)}{'_plus' if constraints >= 6 else ''}"


def _rank(task_id: str) -> bytes:
    return hashlib.sha256(f"m6-phase4-online-v1|{SELECTION_SEED}|{task_id}".encode()).digest()


def _sft_task_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.add(str(json.loads(line)["task_id"]))
    return result


def _select(candidates: list[str], goal_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    by_bucket: dict[str, list[str]] = {}
    for task_id in candidates:
        by_bucket.setdefault(_bucket(goal_map[task_id]), []).append(task_id)
    exact = {bucket: TARGET_TASKS * len(values) / len(candidates) for bucket, values in by_bucket.items()}
    quotas = {bucket: math.floor(value) for bucket, value in exact.items()}
    for bucket in sorted(by_bucket, key=lambda key: (-(exact[key] - quotas[key]), key))[: TARGET_TASKS - sum(quotas.values())]:
        quotas[bucket] += 1
    selected = [
        task_id
        for bucket, values in by_bucket.items()
        for task_id in sorted(values, key=lambda item: (_rank(item), item))[: quotas[bucket]]
    ]
    selected.sort(key=lambda item: (_rank(item), item))
    _require(len(selected) == TARGET_TASKS and len(set(selected)) == TARGET_TASKS, "Phase4 online roster size drift")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, action="append", required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--sft-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol()
    split = validate_split_lock(json.loads(args.split_lock.read_text(encoding="utf-8")))
    _require(split["protocol_sha256"] == protocol["sha256"], "Phase4 online split/protocol drift")
    goals = json.loads(args.goals.read_text(encoding="utf-8"))
    _require(sha256_json(goals) == split["goals_canonical_sha256"], "Phase4 online goals/split drift")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    train_ids = set(split["roles"]["train"]["task_ids"])
    sft_ids = _sft_task_ids(args.sft_jsonl)

    mixed: dict[str, dict[str, Any]] = {}
    source_reports: list[str] = []
    for root in args.collection_root:
        resolved = root.expanduser().resolve()
        report = json.loads((resolved / "collection_report.json").read_text(encoding="utf-8"))
        _require(report.get("complete") is True and report.get("K") == 4, "Phase4 prescan report drift")
        source_reports.append(str(resolved / "collection_report.json"))
        for path in sorted((resolved / "groups").glob("g*.json")):
            group = validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=4)
            rewards = {float(item["binary_reward"]) for item in group["trajectories"]}
            if len(rewards) == 2:
                mixed[group["task_id"]] = group
    candidates = sorted(mixed)
    _require(len(candidates) >= 64, "Phase4 online roster requires at least 64 mixed candidates")
    _require(set(candidates) <= train_ids, "Phase4 mixed candidate escaped train role")
    _require(not (set(candidates) & sft_ids), "Phase4 mixed candidate overlaps SFT")
    _require(all(task_id in goal_map for task_id in candidates), "Phase4 mixed candidate goal missing")
    selected = _select(candidates, goal_map)
    selected_counts = Counter(_bucket(goal_map[task_id]) for task_id in selected)
    candidate_counts = Counter(_bucket(goal_map[task_id]) for task_id in candidates)
    value = {
        "schema_version": "m6_rl_curriculum_v1",
        "study_id": protocol["payload"]["study_id"],
        "development_only": True,
        "formal_training": False,
        "purpose": "phase4_minimal_online_trajectory_grpo",
        "selection": "mixed_current_sft_policy_category_constraint_proportional_hash_v1",
        "selection_seed": SELECTION_SEED,
        "source_split_lock_content_sha256": split["content_sha256"],
        "protocol_sha256": protocol["sha256"],
        "git_sha": protocol["git_sha"],
        "source_role": "train",
        "candidate_mixed_task_count": len(candidates),
        "sft_overlap_count": 0,
        "task_count": len(selected),
        "task_ids": selected,
        "candidate_bucket_counts": dict(sorted(candidate_counts.items())),
        "selected_bucket_counts": dict(sorted(selected_counts.items())),
        "task_order_sha256": sha256_json(selected),
        "source_collection_reports": source_reports,
    }
    value["content_sha256"] = sha256_json(value)
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase4 online roster output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

