#!/usr/bin/env python3
"""Build the frozen 96-task Phase10-D same-corpus paired evaluation roster.

Fresh development-only tasks are drawn from the base train role after burning
every versioned historical exposure union, the complete Phase10-C split, the
SFT D4 and D35 corpora and every already-viewed evaluation roster.  Selection
is stratified by the public category/constraint proxy so the SFT4(D4) vs
SFT35(D4) scale screen cannot be confounded by task mix.  K4, the 18/15
horizon, the task order and the rollout seed are frozen in the roster itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from miniwebwork.m5_webshop_protocol import normalized_instruction, task_id_for_goal_index  # noqa: E402
from miniwebwork.m6_phase10c_data import (  # noqa: E402
    validate_phase10b_exposure_union,
    validate_phase10c_exposure_union,
    validate_phase10c_split,
)
from miniwebwork.m6_posttraining_protocol import (  # noqa: E402
    load_protocol,
    validate_exposure_registry,
    validate_split_lock,
)

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TASK_ID_RE = re.compile(r"^webshop_goal_\d{5}$")
GOAL_COUNT = 12087
TASK_COUNT = 96
SELECTION_SEED = 20260866
ROLLOUT_SEED = 20260867
CONSTRAINT_BUCKET_CAP = 4
SELECTION_NAMESPACE = "m6-phase10d-same-corpus-eval-v1"
PURPOSE = "phase10d_same_corpus_scale_paired_evaluation"
EVALUATION_STAGE = "same_corpus"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _git_sha() -> str:
    value = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase10-D evaluation roster Git drift")
    return value


def _goal_index(task_id: str) -> int:
    _require(TASK_ID_RE.fullmatch(task_id) is not None, "Phase10-D task identity drift")
    return int(task_id.rsplit("_", 1)[1])


def _validate_phase10_exposure_union(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the phase10 exposure union fields this roster consumes."""

    value = dict(payload)
    _require(value.get("schema_version") == "m6_phase10_exposure_union_v1",
             "Phase10-D input Phase10 exposure schema drift")
    _require(value.get("complete") is True and value.get("development_only") is True,
             "Phase10-D input Phase10 exposure is incomplete")
    _require(value.get("promotion_and_holdout_read") is False,
             "Phase10-D input Phase10 exposure opened promotion/holdout")
    rows = value.get("exposures")
    _require(isinstance(rows, list) and rows, "Phase10-D input Phase10 exposure rows missing")
    _require(value.get("exposed_task_count") == len(rows), "Phase10-D input Phase10 exposure count drift")
    _require(value.get("exposed_task_id_sha256") == sha256_json([row["task_id"] for row in rows]),
             "Phase10-D input Phase10 exposure aggregate hash drift")
    _require(value.get("content_sha256") == _self_hash(value),
             "Phase10-D input Phase10 exposure self-hash drift")
    return value


def _validate_exposure_rows(payload: Mapping[str, Any], label: str) -> set[str]:
    rows = payload.get("exposures")
    _require(isinstance(rows, list) and rows, f"Phase10-D {label} rows missing")
    task_ids = set()
    for row in rows:
        task_id = str(row.get("task_id", ""))
        _require(TASK_ID_RE.fullmatch(task_id) is not None, f"Phase10-D {label} task identity drift")
        _require(0 <= _goal_index(task_id) < GOAL_COUNT, f"Phase10-D {label} goal index drift")
        task_ids.add(task_id)
    return task_ids


def _corpus_task_ids(path: Path) -> set[str]:
    resolved = path.expanduser().resolve()
    task_ids: set[str] = set()
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        _require(isinstance(row, Mapping), f"Phase10-D corpus row is malformed: {path}:{line_number}")
        task_id = str(row.get("task_id", ""))
        _require(TASK_ID_RE.fullmatch(task_id) is not None,
                 f"Phase10-D corpus task identity drift: {path}:{line_number}")
        _require(0 <= _goal_index(task_id) < GOAL_COUNT, f"Phase10-D corpus goal index drift: {path}")
        task_ids.add(task_id)
    _require(bool(task_ids), f"Phase10-D corpus contains no tasks: {path}")
    return task_ids


def _constraint_count(goal: Mapping[str, Any]) -> int:
    attributes = goal.get("instruction_attributes") or goal.get("attributes") or []
    options = goal.get("goal_options") or []
    return len(attributes) + len(options) + int(isinstance(goal.get("price_upper"), (int, float)))


def _allocate(total: int, pools: Mapping[str, int]) -> dict[str, int]:
    """Proportional allocation with largest remainders and name tie-breaks."""

    _require(bool(pools) and total >= len(pools), "Phase10-D allocation is degenerate")
    scale = total / sum(pools.values())
    exact = {key: value * scale for key, value in pools.items()}
    allocated = {key: math.floor(exact[key]) for key in pools}
    remainder = total - sum(allocated.values())
    for key in sorted(pools, key=lambda item: (-(exact[item] - allocated[item]), item))[:remainder]:
        allocated[key] += 1
    return allocated


def _cell_rank(task_id: str) -> bytes:
    return hashlib.sha256(
        f"{SELECTION_NAMESPACE}|{SELECTION_SEED}|{task_id}".encode("utf-8")
    ).digest()


def build_roster(
    *,
    goals: Sequence[Mapping[str, Any]],
    base_split: Mapping[str, Any],
    exclusions: Sequence[Mapping[str, Any]],
    protocol_sha256: str,
    git_sha: str,
) -> dict[str, Any]:
    """Select 96 fresh train tasks stratified by public category/constraint proxy."""

    _require(len(goals) == GOAL_COUNT, "Phase10-D canonical goal count drift")
    for index, goal in enumerate(goals):
        _require(isinstance(goal, Mapping) and goal.get("goal_index") == index,
                 f"Phase10-D goal row drift: {index}")
    _require(sha256_json(list(goals)) == base_split["goals_canonical_sha256"],
             "Phase10-D goals/base split drift")
    train = list(base_split["roles"]["train"]["task_ids"])
    _require(len(train) == len(set(train)), "Phase10-D base train role has duplicates")
    train_set = set(train)

    excluded: dict[str, set[str]] = {}
    bindings: list[dict[str, Any]] = []
    for item in exclusions:
        name = str(item["name"])
        _require(name in {"m5_goal_exposure_registry_v1",
                          "phase10_exposure_union",
                          "phase10b_exposure_union",
                          "phase10c_exposure_union",
                          "phase10c_split_all_roles",
                          "sft_d4_corpus",
                          "sft_d35_corpus"}, f"Phase10-D unknown exclusion: {name}")
        _require(name not in excluded, f"Phase10-D duplicate exclusion: {name}")
        task_ids = set(item["task_ids"])
        _require(bool(task_ids), f"Phase10-D exclusion is empty: {name}")
        excluded[name] = task_ids
        bindings.append({
            "name": name,
            "paths": [str(Path(path).expanduser().resolve()) for path in item["paths"]],
            "file_sha256": [str(value) for value in item["file_sha256"]],
            "task_count": len(task_ids),
        })
    excluded_union: set[str] = set()
    for task_ids in excluded.values():
        excluded_union.update(task_ids)

    goal_map = {task_id_for_goal_index(index): goal for index, goal in enumerate(goals)}
    excluded_instructions = {
        normalized_instruction(str(goal_map[task_id]["instruction"]))
        for task_id in excluded_union
    }
    fresh = [
        task_id for task_id in train
        if task_id not in excluded_union
        and normalized_instruction(str(goal_map[task_id]["instruction"])) not in excluded_instructions
    ]
    _require(len(fresh) >= TASK_COUNT, "Phase10-D fresh train population is too small")

    cells: dict[str, list[str]] = {}
    for task_id in fresh:
        goal = goal_map[task_id]
        category = str(goal.get("category", ""))
        _require(category, f"Phase10-D goal has no public category: {task_id}")
        bucket = min(_constraint_count(goal), CONSTRAINT_BUCKET_CAP)
        cells.setdefault(f"{category}|{bucket}", []).append(task_id)
    category_pools = Counter()
    for key, tasks in cells.items():
        category_pools[key.rsplit("|", 1)[0]] += len(tasks)
    category_alloc = _allocate(TASK_COUNT, dict(category_pools))

    selected: list[str] = []
    stratification: dict[str, dict[str, Any]] = {}
    used_instructions: set[str] = set()
    for category in sorted(category_alloc):
        bucket_pools = Counter()
        for key, tasks in cells.items():
            if key.rsplit("|", 1)[0] == category:
                bucket_pools[int(key.rsplit("|", 1)[1])] += len(tasks)
        bucket_alloc = _allocate(category_alloc[category], {str(b): c for b, c in bucket_pools.items()})
        for bucket in sorted(bucket_alloc, key=int):
            key = f"{category}|{bucket}"
            taken: list[str] = []
            for task_id in sorted(cells[key], key=lambda item: (_cell_rank(item), item)):
                instruction = normalized_instruction(str(goal_map[task_id]["instruction"]))
                if instruction in used_instructions:
                    continue
                used_instructions.add(instruction)
                taken.append(task_id)
                if len(taken) == bucket_alloc[bucket]:
                    break
            _require(len(taken) == bucket_alloc[bucket],
                     f"Phase10-D cell population too small: {key}")
            selected.extend(taken)
            stratification[key] = {
                "category": category,
                "constraint_bucket": int(bucket),
                "pool_task_count": len(cells[key]),
                "selected_task_count": len(taken),
            }
    _require(len(selected) == len(set(selected)) == TASK_COUNT, "Phase10-D selection count drift")
    _require(set(selected).isdisjoint(excluded_union), "Phase10-D selected an excluded task")
    _require(set(selected) <= train_set, "Phase10-D selection escaped train role")

    value = {
        "schema_version": "m6_rl_curriculum_v1",
        "development_only": True,
        "formal_training": False,
        "purpose": PURPOSE,
        "selection": "phase10d_same_corpus_fresh_stratified_v1",
        "selection_seed": SELECTION_SEED,
        "selection_namespace": SELECTION_NAMESPACE,
        "source_role": "train",
        "source_split_lock_content_sha256": base_split["content_sha256"],
        "protocol_sha256": protocol_sha256,
        "git_sha": git_sha,
        "phase10c_evaluation_stage": EVALUATION_STAGE,
        "constraint_proxy": {
            "public_category": "goals.category",
            "public_constraint_count": (
                "len(instruction_attributes|attributes) + len(goal_options) + price_upper_present"
            ),
            "constraint_bucket_cap": CONSTRAINT_BUCKET_CAP,
        },
        "exclusion_bindings": bindings,
        "exclusion_union_task_count": len(excluded_union),
        "excluded_from_train_task_count": len(excluded_union & train_set),
        "fresh_train_task_count": len(fresh),
        "stratification": stratification,
        "category_task_counts": {
            category: count for category, count in sorted(category_alloc.items())
        },
        "task_count": TASK_COUNT,
        "task_ids": selected,
        "task_order_sha256": sha256_json(selected),
        "K": 4,
        "max_model_turns": 18,
        "max_environment_steps": 15,
        "rollout_seed": ROLLOUT_SEED,
        "training_performed": False,
        "optimizer_steps": 0,
        "checks": {
            "selected_has_excluded_overlap": 0,
            "selected_instruction_overlap_with_excluded": 0,
            "selected_instruction_unique": True,
            "selected_subset_of_train": True,
            "promotion_and_holdout_read": False,
        },
    }
    value["content_sha256"] = _self_hash(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--m5-exposure-registry", type=Path, required=True)
    parser.add_argument("--phase10-exposure-union", type=Path, required=True)
    parser.add_argument("--phase10b-exposure-union", type=Path, required=True)
    parser.add_argument("--phase10c-exposure-union", type=Path, required=True)
    parser.add_argument("--phase10c-split", type=Path, required=True)
    parser.add_argument("--d4-train", type=Path, required=True)
    parser.add_argument("--d4-dev", type=Path, required=True)
    parser.add_argument("--d35-train", type=Path, required=True)
    parser.add_argument("--d35-dev", type=Path, required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    current_git = _git_sha()
    _require(args.producer_git_sha == current_git, "Phase10-D roster producer Git drift")
    protocol = load_protocol()

    def read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))

    base = validate_split_lock(read_json(args.base_split))
    _require(base["protocol_sha256"] == protocol["sha256"], "Phase10-D roster protocol drift")
    goals = read_json(args.goals)
    phase10c_split = validate_phase10c_split(read_json(args.phase10c_split))
    phase10c_roles: set[str] = set()
    for role in phase10c_split["roles"].values():
        phase10c_roles.update(role["task_ids"])

    exclusions = [
        {
            "name": "m5_goal_exposure_registry_v1",
            "paths": [args.m5_exposure_registry],
            "file_sha256": [sha256_file(args.m5_exposure_registry.expanduser().resolve())],
            "task_ids": _validate_exposure_rows(
                validate_exposure_registry(read_json(args.m5_exposure_registry)),
                "M5 exposure registry"),
        },
        {
            "name": "phase10_exposure_union",
            "paths": [args.phase10_exposure_union],
            "file_sha256": [sha256_file(args.phase10_exposure_union.expanduser().resolve())],
            "task_ids": _validate_exposure_rows(
                _validate_phase10_exposure_union(read_json(args.phase10_exposure_union)),
                "Phase10 exposure union"),
        },
        {
            "name": "phase10b_exposure_union",
            "paths": [args.phase10b_exposure_union],
            "file_sha256": [sha256_file(args.phase10b_exposure_union.expanduser().resolve())],
            "task_ids": _validate_exposure_rows(
                validate_phase10b_exposure_union(read_json(args.phase10b_exposure_union)),
                "Phase10-B exposure union"),
        },
        {
            "name": "phase10c_exposure_union",
            "paths": [args.phase10c_exposure_union],
            "file_sha256": [sha256_file(args.phase10c_exposure_union.expanduser().resolve())],
            "task_ids": _validate_exposure_rows(
                validate_phase10c_exposure_union(read_json(args.phase10c_exposure_union)),
                "Phase10-C exposure union"),
        },
        {
            "name": "phase10c_split_all_roles",
            "paths": [args.phase10c_split],
            "file_sha256": [sha256_file(args.phase10c_split.expanduser().resolve())],
            "task_ids": phase10c_roles,
        },
        {
            "name": "sft_d4_corpus",
            "paths": [args.d4_train, args.d4_dev],
            "file_sha256": [
                sha256_file(args.d4_train.expanduser().resolve()),
                sha256_file(args.d4_dev.expanduser().resolve()),
            ],
            "task_ids": _corpus_task_ids(args.d4_train) | _corpus_task_ids(args.d4_dev),
        },
        {
            "name": "sft_d35_corpus",
            "paths": [args.d35_train, args.d35_dev],
            "file_sha256": [
                sha256_file(args.d35_train.expanduser().resolve()),
                sha256_file(args.d35_dev.expanduser().resolve()),
            ],
            "task_ids": _corpus_task_ids(args.d35_train) | _corpus_task_ids(args.d35_dev),
        },
    ]
    roster = build_roster(
        goals=goals,
        base_split=base,
        exclusions=exclusions,
        protocol_sha256=protocol["sha256"],
        git_sha=current_git,
    )
    root = args.output_root.expanduser().resolve()
    destination = root / f"{EVALUATION_STAGE}.json"
    _require(not destination.exists(), f"Phase10-D evaluation roster exists: {EVALUATION_STAGE}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, roster)
    print(json.dumps({
        "path": str(destination),
        "task_count": roster["task_count"],
        "fresh_train_task_count": roster["fresh_train_task_count"],
        "exclusion_union_task_count": roster["exclusion_union_task_count"],
        "category_task_counts": roster["category_task_counts"],
        "task_order_sha256": roster["task_order_sha256"],
        "content_sha256": roster["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
