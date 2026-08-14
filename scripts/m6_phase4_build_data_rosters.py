#!/usr/bin/env python3
"""Freeze two disjoint 64-task rosters for SFT-complementary RL data synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402

PARTITION_TASK_COUNT = 64
PARTITION_SEEDS = {"a": 20260824, "b": 20260825}
EXPECTED_SFT_TASK_COUNT = 156
MAXIMUM_BUCKET_SHARE_GAP = 0.05


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


def _rank(namespace: str, seed: int, task_id: str) -> bytes:
    return hashlib.sha256(f"{namespace}|{seed}|{task_id}".encode()).digest()


def _proportional_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    _require(population >= total > 0, "M6 Phase4 roster quota population drift")
    exact = {key: total * count / population for key, count in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    _require(sum(quotas.values()) == total, "M6 Phase4 roster quota total drift")
    _require(all(quotas[key] <= counts[key] for key in counts), "M6 Phase4 roster quota exceeds bucket")
    return quotas


def _select_partition(
    eligible_ids: Sequence[str],
    goal_map: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    count: int = PARTITION_TASK_COUNT,
) -> tuple[list[str], dict[str, int], dict[str, int], float]:
    rows: dict[str, list[str]] = {}
    for task_id in eligible_ids:
        rows.setdefault(_bucket(goal_map[task_id]), []).append(task_id)
    population_counts = {key: len(value) for key, value in rows.items()}
    quotas = _proportional_quotas(population_counts, count)
    selected: list[str] = []
    for bucket, task_ids in rows.items():
        ranked = sorted(task_ids, key=lambda task_id: (_rank("m6-phase4-data-v1", seed, task_id), task_id))
        selected.extend(ranked[: quotas[bucket]])
    selected.sort(key=lambda task_id: (_rank("m6-phase4-order-v1", seed, task_id), task_id))
    selected_counts = Counter(_bucket(goal_map[task_id]) for task_id in selected)
    population_total = len(eligible_ids)
    maximum_gap = max(
        abs(selected_counts.get(bucket, 0) / count - population_counts[bucket] / population_total)
        for bucket in population_counts
    )
    _require(len(selected) == count and len(set(selected)) == count, "M6 Phase4 partition selection drift")
    _require(maximum_gap <= MAXIMUM_BUCKET_SHARE_GAP, "M6 Phase4 partition bucket share gap exceeded")
    return selected, population_counts, dict(selected_counts), maximum_gap


def _load_roster(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), f"M6 Phase4 exclude roster self-hash drift: {path}")
    task_ids = value.get("task_ids")
    _require(isinstance(task_ids, list) and task_ids, f"M6 Phase4 exclude roster is empty: {path}")
    return value


def _load_jsonl_task_ids(paths: Sequence[Path]) -> set[str]:
    task_ids: set[str] = set()
    for path in paths:
        _require(path.is_file(), f"M6 Phase4 SFT JSONL is missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            _require(isinstance(task_id, str) and task_id, "M6 Phase4 SFT row lacks task_id")
            task_ids.add(task_id)
    return task_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--sft-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--exclude-roster", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol()
    split = validate_split_lock(json.loads(args.split_lock.read_text(encoding="utf-8")))
    _require(split["protocol_sha256"] == protocol["sha256"], "M6 Phase4 split/protocol drift")
    goals = json.loads(args.goals.read_text(encoding="utf-8"))
    _require(sha256_json(goals) == split["goals_canonical_sha256"], "M6 Phase4 goals/split drift")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}

    train_ids = list(split["roles"]["train"]["task_ids"])
    nontrain_role_ids = {
        role: set(value["task_ids"])
        for role, value in split["roles"].items()
        if role != "train"
    }
    nontrain_ids = set().union(*nontrain_role_ids.values())
    sft_ids = _load_jsonl_task_ids(args.sft_jsonl)
    _require(len(sft_ids) == EXPECTED_SFT_TASK_COUNT, "M6 Phase4 SFT task count drift")
    _require(sft_ids <= nontrain_ids, "M6 Phase4 SFT tasks escaped frozen non-train roles")
    prior_rosters = [_load_roster(path) for path in args.exclude_roster]
    prior_ids = {task_id for roster in prior_rosters for task_id in roster["task_ids"]}
    _require(not (prior_ids & sft_ids), "M6 Phase4 prior roster overlaps SFT")

    excluded = nontrain_ids | prior_ids | sft_ids
    eligible = [task_id for task_id in train_ids if task_id not in excluded]
    _require(len(eligible) >= 2 * PARTITION_TASK_COUNT, "M6 Phase4 eligible population too small")
    _require(all(task_id in goal_map for task_id in eligible), "M6 Phase4 goal map incomplete")

    selected_by_partition: dict[str, list[str]] = {}
    partition_reports: dict[str, dict[str, Any]] = {}
    remaining = list(eligible)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_paths = [output_dir / f"roster_{name}.json" for name in PARTITION_SEEDS] + [output_dir / "manifest.json"]
    _require(not any(path.exists() for path in target_paths), "M6 Phase4 roster output already exists")

    for name, seed in PARTITION_SEEDS.items():
        selected, population_counts, selected_counts, maximum_gap = _select_partition(
            remaining,
            goal_map,
            seed=seed,
        )
        selected_by_partition[name] = selected
        remaining = [task_id for task_id in remaining if task_id not in set(selected)]
        roster = {
            "schema_version": "m6_rl_curriculum_v1",
            "study_id": protocol["payload"]["study_id"],
            "development_only": True,
            "formal_training": False,
            "purpose": "phase4_sft_disjoint_online_data_synthesis",
            "selection": "train_minus_nontrain_sft_and_prior_category_constraint_proportional_hash_v1",
            "partition": name,
            "selection_seed": seed,
            "source_split_lock_content_sha256": split["content_sha256"],
            "protocol_sha256": protocol["sha256"],
            "git_sha": protocol["git_sha"],
            "source_role": "train",
            "source_role_task_count": len(train_ids),
            "excluded_roles": sorted(nontrain_role_ids),
            "sft_task_count": len(sft_ids),
            "prior_excluded_task_count": len(prior_ids),
            "eligible_task_count_before_partition": sum(population_counts.values()),
            "task_count": len(selected),
            "task_ids": selected,
            "population_bucket_counts": dict(sorted(population_counts.items())),
            "selected_bucket_counts": dict(sorted(selected_counts.items())),
            "maximum_bucket_share_gap": maximum_gap,
            "task_order_sha256": sha256_json(selected),
        }
        roster["content_sha256"] = sha256_json(roster)
        publish_immutable_json(output_dir / f"roster_{name}.json", roster)
        partition_reports[name] = {
            "selection_seed": seed,
            "task_count": len(selected),
            "task_order_sha256": roster["task_order_sha256"],
            "roster_content_sha256": roster["content_sha256"],
            "maximum_bucket_share_gap": maximum_gap,
            "file": f"roster_{name}.json",
        }

    union = set().union(*(set(value) for value in selected_by_partition.values()))
    _require(len(union) == 2 * PARTITION_TASK_COUNT, "M6 Phase4 partitions overlap")
    _require(not (union & nontrain_ids), "M6 Phase4 roster overlaps a frozen non-train role")
    _require(not (union & sft_ids), "M6 Phase4 roster overlaps SFT")
    _require(not (union & prior_ids), "M6 Phase4 roster overlaps prior P1 data")
    manifest = {
        "schema_version": "m6_phase4_data_roster_manifest_v1",
        "study_id": protocol["payload"]["study_id"],
        "development_only": True,
        "formal_training": False,
        "purpose": "high_coverage_strong_contrast_sft_complementary_online_synthesis",
        "source_split_lock_content_sha256": split["content_sha256"],
        "protocol_sha256": protocol["sha256"],
        "git_sha": protocol["git_sha"],
        "sft_task_count": len(sft_ids),
        "sft_task_order_sha256": sha256_json(sorted(sft_ids)),
        "prior_roster_content_sha256": [roster["content_sha256"] for roster in prior_rosters],
        "combined_task_count": len(union),
        "partitions": partition_reports,
        "partition_overlap_count": 0,
        "sft_overlap_count": 0,
        "nontrain_role_overlap_count": 0,
        "prior_roster_overlap_count": 0,
        "maximum_allowed_bucket_share_gap": MAXIMUM_BUCKET_SHARE_GAP,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    publish_immutable_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
