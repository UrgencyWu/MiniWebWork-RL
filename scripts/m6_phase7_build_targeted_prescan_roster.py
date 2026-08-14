#!/usr/bin/env python3
"""Freeze 32 SFT-disjoint tasks for a targeted terminal-contrast prescan."""

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
    _failure_class,
    _load_groups,
    _load_hashed,
    _prebuy_public_choice,
)
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402

TARGET_TASKS = 32
SELECTION_SEED = 20260832
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    lowered = {part.casefold() for part in resolved.parts}
    _require(not (lowered & FORBIDDEN_PATH_PARTS), f"Phase7 {label} touches promotion/holdout")
    return resolved


def _git_sha() -> str:
    value = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase7 repository Git SHA drift")
    return value


def _load_jsonl_task_ids(paths: Sequence[Path]) -> set[str]:
    result: set[str] = set()
    for raw_path in paths:
        path = _safe_path(raw_path, label="SFT input")
        _require(path.is_file(), f"Phase7 SFT JSONL missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            _require(isinstance(task_id, str) and task_id, "Phase7 SFT row lacks task_id")
            result.add(task_id)
    _require(result, "Phase7 SFT task set is empty")
    return result


def _goal_has_options(goal: Mapping[str, Any]) -> bool:
    options = goal.get("goal_options")
    return isinstance(options, (list, tuple, set, dict)) and bool(options)


def _candidate_metrics(groups: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], set[str]]:
    """Count strict/partial pairs and identify tasks already having same-item pairs."""

    pair_counts: dict[str, int] = defaultdict(int)
    same_item_tasks: set[str] = set()
    for group in groups:
        task_id = str(group["task_id"])
        strict: list[dict[str, Any]] = []
        partial: list[dict[str, Any]] = []
        for trajectory in group["trajectories"]:
            failure_class = _failure_class(trajectory)
            choice = _prebuy_public_choice(trajectory)
            if choice is None:
                continue
            if failure_class == "strict_success":
                strict.append(choice)
            elif failure_class == "partial_purchase":
                partial.append(choice)
        if not strict or not partial:
            continue
        pair_counts[task_id] += len(strict) * len(partial)
        if any(left["asin"] == right["asin"] for left in strict for right in partial):
            same_item_tasks.add(task_id)
    return dict(pair_counts), same_item_tasks


def _rank(task_id: str) -> bytes:
    return hashlib.sha256(f"m6-phase7-targeted-prescan-v1|{SELECTION_SEED}|{task_id}".encode()).digest()


def _select_candidates(
    pair_counts: Mapping[str, int],
    same_item_tasks: set[str],
    goal_map: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    candidates = sorted(set(pair_counts) - same_item_tasks)
    _require(len(candidates) >= TARGET_TASKS, "Phase7 targeted prescan has fewer than 32 eligible tasks")
    _require(all(task_id in goal_map for task_id in candidates), "Phase7 candidate goal missing")
    ordered = sorted(
        candidates,
        key=lambda task_id: (
            0 if _goal_has_options(goal_map[task_id]) else 1,
            -int(pair_counts[task_id]),
            _rank(task_id),
            task_id,
        ),
    )
    selected = ordered[:TARGET_TASKS]
    _require(len(selected) == TARGET_TASKS and len(set(selected)) == TARGET_TASKS, "Phase7 roster size drift")
    return selected, candidates


def _binding_key(binding: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(Path(str(binding["collection_root"])).expanduser().resolve()),
        str(binding["collection_report_content_sha256"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prescan-root", type=Path, action="append", required=True)
    parser.add_argument("--diagnostic-report", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--sft-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    current_git = _git_sha()
    _require(args.producer_git_sha == current_git, "Phase7 roster producer Git drift")
    protocol = load_protocol()
    _require(protocol["git_sha"] == current_git, "Phase7 protocol/repository Git drift")

    diagnostic_path = _safe_path(args.diagnostic_report, label="diagnostic report")
    diagnostic = _load_hashed(diagnostic_path)
    _require(
        diagnostic.get("schema_version") == "m6_phase7_terminal_contrast_diagnostic_v2"
        and diagnostic.get("complete") is True
        and diagnostic.get("development_only") is True
        and diagnostic.get("training_performed") is False
        and diagnostic.get("target_asin_or_hidden_answer_used") is False,
        "Phase7 diagnostic contract drift",
    )
    _require(
        diagnostic.get("decision", {}).get("next_action") == "collect_targeted_prescan_before_training",
        "Phase7 targeted prescan was not selected by the frozen diagnostic",
    )

    split_path = _safe_path(args.split_lock, label="split lock")
    split = validate_split_lock(json.loads(split_path.read_text(encoding="utf-8")))
    _require(split["protocol_sha256"] == protocol["sha256"], "Phase7 split/protocol drift")
    goals_path = _safe_path(args.goals, label="goals")
    goals = json.loads(goals_path.read_text(encoding="utf-8"))
    _require(isinstance(goals, list), "Phase7 goals payload must be a list")
    _require(sha256_json(goals) == split["goals_canonical_sha256"], "Phase7 goals/split drift")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    train_ids = set(split["roles"]["train"]["task_ids"])
    sft_ids = _load_jsonl_task_ids(args.sft_jsonl)

    roots = [_safe_path(path, label="prescan input") for path in args.prescan_root]
    groups, bindings = _load_groups(roots, expected_mode="phase4_data_synthesis")
    expected_bindings = {_binding_key(item) for item in diagnostic["prescan_bindings"]}
    actual_bindings = {_binding_key(item) for item in bindings}
    _require(actual_bindings == expected_bindings, "Phase7 diagnostic/prescan input binding drift")

    pair_counts, same_item_tasks = _candidate_metrics(groups)
    prescan_summary = diagnostic["prescan"]
    _require(
        len(pair_counts) == prescan_summary["strict_partial_task_count"],
        "Phase7 strict-partial task count drift",
    )
    _require(
        same_item_tasks == set(prescan_summary["same_item_task_ids"]),
        "Phase7 same-item exclusion drift",
    )
    _require(set(pair_counts) <= train_ids, "Phase7 candidate escaped frozen train role")
    _require(not (set(pair_counts) & sft_ids), "Phase7 candidate overlaps SFT")

    selected, candidates = _select_candidates(pair_counts, same_item_tasks, goal_map)
    selected_counts = Counter(_bucket(goal_map[task_id]) for task_id in selected)
    candidate_counts = Counter(_bucket(goal_map[task_id]) for task_id in candidates)
    selected_option_count = sum(_goal_has_options(goal_map[task_id]) for task_id in selected)
    report_hashes = [binding["collection_report_content_sha256"] for binding in bindings]
    value = {
        "schema_version": "m6_rl_curriculum_v1",
        "study_id": protocol["payload"]["study_id"],
        "development_only": True,
        "formal_training": False,
        "purpose": "phase7_targeted_same_item_terminal_contrast_prescan",
        "selection": "strict_partial_without_same_item_option_first_pair_count_then_hash_v1",
        "selection_seed": SELECTION_SEED,
        "source_role": "train",
        "source_split_lock_content_sha256": split["content_sha256"],
        "protocol_sha256": protocol["sha256"],
        "git_sha": current_git,
        "source_diagnostic_content_sha256": diagnostic["content_sha256"],
        "source_collection_report_content_sha256": report_hashes,
        "source_strict_partial_task_count": len(pair_counts),
        "excluded_existing_same_item_task_count": len(same_item_tasks),
        "candidate_task_count": len(candidates),
        "candidate_option_task_count": sum(_goal_has_options(goal_map[task_id]) for task_id in candidates),
        "selected_option_task_count": selected_option_count,
        "sft_overlap_count": 0,
        "nontrain_role_overlap_count": 0,
        "task_count": len(selected),
        "task_ids": selected,
        "candidate_bucket_counts": dict(sorted(candidate_counts.items())),
        "selected_bucket_counts": dict(sorted(selected_counts.items())),
        "strict_partial_pair_counts": {task_id: pair_counts[task_id] for task_id in selected},
        "task_order_sha256": sha256_json(selected),
        "targeted_prescan_contract": {
            "training_performed": False,
            "optimizer_steps": 0,
            "policy_identity": "frozen_sft_adapter",
            "K": 4,
            "max_model_turns": 18,
            "max_environment_steps": 15,
            "rollout_seed": SELECTION_SEED,
            "future_rl_must_recollect_on_policy": True,
        },
    }
    value["content_sha256"] = sha256_json(value)
    output = _safe_path(args.output, label="roster output")
    _require(not output.exists(), "Phase7 targeted roster output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
