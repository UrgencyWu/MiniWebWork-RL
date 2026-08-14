#!/usr/bin/env python3
"""Freeze a 64-task SFT-disjoint train-role roster for the Phase2 horizon test."""

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

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _constraint_count(goal: Mapping[str, Any]) -> int:
    attributes = goal.get("instruction_attributes") or goal.get("attributes") or []
    options = goal.get("goal_options") or []
    return len(attributes) + len(options) + int(isinstance(goal.get("price_upper"), (int, float)))


def _bucket(goal: Mapping[str, Any]) -> str:
    category = str(goal.get("category") or "unknown").strip().casefold()
    count = _constraint_count(goal)
    return f"{category}|constraints_{min(count, 6)}{'_plus' if count >= 6 else ''}"


def _rank(seed: int, task_id: str) -> bytes:
    return hashlib.sha256(f"m6-phase2-horizon-v1|{seed}|{task_id}".encode()).digest()


def _proportional_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    _require(population >= total > 0, "M6 Phase2 horizon quota population drift")
    exact = {key: total * count / population for key, count in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    _require(sum(quotas.values()) == total, "M6 Phase2 horizon quota total drift")
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    _require(args.task_count == 64 and args.seed == 20260821, "M6 Phase2 horizon roster contract drift")
    protocol = load_protocol()
    split = validate_split_lock(json.loads(args.split_lock.read_text(encoding="utf-8")))
    _require(split["protocol_sha256"] == protocol["sha256"], "M6 Phase2 horizon split/protocol drift")
    train_ids = list(split["roles"]["train"]["task_ids"])
    exposed_role_ids = {
        role: set(value["task_ids"])
        for role, value in split["roles"].items()
        if role != "train"
    }
    exposed_ids = set().union(*exposed_role_ids.values())
    train_set = set(train_ids)
    overlap_counts = {
        role: len(train_set & task_ids)
        for role, task_ids in exposed_role_ids.items()
    }
    excluded_overlap_ids = train_set & exposed_ids
    eligible_ids = [task_id for task_id in train_ids if task_id not in exposed_ids]
    _require(
        len(eligible_ids) == len(train_ids) - len(excluded_overlap_ids),
        "M6 Phase2 horizon eligible-task set difference drift",
    )
    _require(len(eligible_ids) >= args.task_count, "M6 Phase2 horizon eligible population too small")
    goals = json.loads(args.goals.read_text(encoding="utf-8"))
    _require(sha256_json(goals) == split["goals_canonical_sha256"], "M6 Phase2 horizon goals/split drift")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    _require(all(task_id in goal_map for task_id in eligible_ids), "M6 Phase2 horizon goal map incomplete")
    bucket_rows: dict[str, list[str]] = {}
    for task_id in eligible_ids:
        bucket_rows.setdefault(_bucket(goal_map[task_id]), []).append(task_id)
    counts = {key: len(value) for key, value in bucket_rows.items()}
    quotas = _proportional_quotas(counts, args.task_count)
    selected = []
    for key, task_ids in bucket_rows.items():
        ranked = sorted(task_ids, key=lambda task_id: (_rank(args.seed, task_id), task_id))
        selected.extend(ranked[: quotas[key]])
    selected.sort(key=lambda task_id: (_rank(args.seed + 1, task_id), task_id))
    _require(len(selected) == args.task_count and len(set(selected)) == args.task_count, "M6 Phase2 horizon selection drift")
    _require(not (set(selected) & exposed_ids), "M6 Phase2 horizon selected task overlaps exposed roles")
    selected_buckets = Counter(_bucket(goal_map[task_id]) for task_id in selected)
    report = {
        "schema_version": "m6_rl_curriculum_v1",
        "study_id": protocol["payload"]["study_id"],
        "development_only": True,
        "formal_training": False,
        "purpose": "phase2_paired_horizon_diagnostic_only",
        "selection": "train_role_minus_exposed_category_constraint_proportional_hash_v1",
        "selection_seed": args.seed,
        "source_split_lock_content_sha256": split["content_sha256"],
        "protocol_sha256": protocol["sha256"],
        "git_sha": protocol["git_sha"],
        "source_role": "train",
        "source_role_task_count": len(train_ids),
        "excluded_roles": sorted(exposed_role_ids),
        "source_role_overlap_counts": dict(sorted(overlap_counts.items())),
        "excluded_overlap_task_count": len(excluded_overlap_ids),
        "eligible_task_count": len(eligible_ids),
        "task_count": len(selected),
        "task_ids": selected,
        "population_bucket_counts": dict(sorted(counts.items())),
        "selected_bucket_counts": dict(sorted(selected_buckets.items())),
        "task_order_sha256": sha256_json(selected),
    }
    report["content_sha256"] = sha256_json(report)
    publish_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
