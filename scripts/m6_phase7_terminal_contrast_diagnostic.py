#!/usr/bin/env python3
"""Measure whether Phase4 RL data contains useful terminal success/failure contrasts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m6_phase4_build_data_rosters import _bucket  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402

K = 4
ONLINE_SAME_ITEM_MIXED_SHARE = 0.50
ONLINE_OPTION_STRONG_SHARE = 0.50
MINIMUM_REBUILD_SAME_ITEM_TASKS = 20
MINIMUM_REBUILD_OPTION_TASKS = 12
FORBIDDEN_PATH_PARTS = {"promotion", "holdout"}
PUBLIC_ASIN_ACTION_RE = re.compile(r"^click\[(B[0-9A-Z]{9})\]$", re.IGNORECASE)
SELECTED_OPTION_RE = re.compile(
    r"^\s*-\s*([^\n:]+?)\s*\(selected:\s*([^\)]+)\)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("content_sha256") == _self_hash(value), f"Phase7 self-hash drift: {path}")
    return value


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _selected_options_from_visible_text(value: Any) -> tuple[tuple[str, str], ...]:
    output = []
    for match in SELECTED_OPTION_RE.finditer(str(value or "")):
        name = _normalized_text(match.group(1))
        selected = _normalized_text(match.group(2))
        if name and selected and selected != "not selected":
            output.append((name, selected))
    return tuple(sorted(set(output)))


def _prebuy_public_choice(trajectory: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read only the policy-visible state immediately before public Buy Now."""

    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        return None
    buy_index = None
    for index in range(len(turns) - 1, -1, -1):
        turn = turns[index]
        action = turn.get("action")
        command = _normalized_text(action.get("command")) if isinstance(action, Mapping) else ""
        if command == "click[buy now]":
            buy_index = index
            break
    if buy_index is None:
        return None
    observation = turns[buy_index].get("observation")
    if not isinstance(observation, Mapping):
        return None
    asin = ""
    for turn in reversed(turns[:buy_index]):
        action = turn.get("action")
        command = str(action.get("command", "")).strip() if isinstance(action, Mapping) else ""
        match = PUBLIC_ASIN_ACTION_RE.fullmatch(command)
        if match is not None:
            asin = match.group(1).upper()
            break
    if not asin:
        return None
    return {
        "asin": asin,
        "selected_options": _selected_options_from_visible_text(observation.get("visible_text")),
    }


def _failure_class(trajectory: Mapping[str, Any]) -> str:
    if bool(trajectory.get("success")):
        return "strict_success"
    termination = str(trajectory.get("termination_reason", ""))
    if termination == "purchase":
        return "partial_purchase" if float(trajectory.get("task_score", 0.0)) > 0.0 else "zero_match_purchase"
    if termination in {"max_model_turns", "max_environment_steps"}:
        return "horizon_exhaustion"
    return "schema_or_action_failure"


def _load_groups(roots: Sequence[Path], *, expected_mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        lowered_parts = {part.casefold() for part in root.parts}
        _require(not (lowered_parts & FORBIDDEN_PATH_PARTS), "Phase7 input touches promotion/holdout")
        report = _load_hashed(root / "collection_report.json")
        _require(
            report.get("complete") is True
            and report.get("development_only") is True
            and report.get("mode") == expected_mode
            and report.get("K") == K,
            f"Phase7 collection contract drift: {root}",
        )
        paths = sorted((root / "groups").glob("g*.json"))
        _require(len(paths) == report.get("task_count"), f"Phase7 group count drift: {root}")
        loaded = [_load_hashed(path) for path in paths]
        _require(
            report.get("group_content_sha256") == [group["content_sha256"] for group in loaded],
            f"Phase7 group/report binding drift: {root}",
        )
        for group in loaded:
            trajectories = group.get("trajectories")
            _require(
                group.get("complete") is True
                and group.get("K") == K
                and isinstance(trajectories, list)
                and len(trajectories) == K,
                f"Phase7 malformed K4 group: {root}",
            )
        groups.extend(loaded)
        bindings.append({
            "collection_root": str(root),
            "collection_report_content_sha256": report["content_sha256"],
            "task_count": report["task_count"],
            "trajectory_count": report["trajectory_count"],
            "policy_lineage": report["policy_lineage"],
        })
    return groups, bindings


def _analyze_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    goal_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    unique_tasks: set[str] = set()
    mixed_tasks: set[str] = set()
    strict_partial_tasks: set[str] = set()
    same_item_tasks: set[str] = set()
    option_contrast_tasks: set[str] = set()
    strict_partial_pairs = same_item_pairs = option_contrast_pairs = 0
    missing_purchase_state = 0

    for group in groups:
        task_id = str(group["task_id"])
        _require(task_id in goal_map, f"Phase7 task missing from goals: {task_id}")
        unique_tasks.add(task_id)
        rows = []
        for trajectory in group["trajectories"]:
            failure_class = _failure_class(trajectory)
            failure_counts[failure_class] += 1
            choice = _prebuy_public_choice(trajectory)
            if failure_class in {"strict_success", "partial_purchase", "zero_match_purchase"} and choice is None:
                missing_purchase_state += 1
            rows.append((trajectory, failure_class, choice))
        success_count = sum(failure_class == "strict_success" for _, failure_class, _ in rows)
        if 0 < success_count < K:
            mixed_tasks.add(task_id)
        strict = [choice for _, failure_class, choice in rows if failure_class == "strict_success" and choice]
        partial = [choice for _, failure_class, choice in rows if failure_class == "partial_purchase" and choice]
        if strict and partial:
            strict_partial_tasks.add(task_id)
        for preferred in strict:
            for rejected in partial:
                strict_partial_pairs += 1
                if preferred["asin"] != rejected["asin"]:
                    continue
                same_item_pairs += 1
                same_item_tasks.add(task_id)
                if preferred["selected_options"] != rejected["selected_options"]:
                    option_contrast_pairs += 1
                    option_contrast_tasks.add(task_id)

    def bucket_counts(task_ids: set[str]) -> dict[str, int]:
        return dict(sorted(Counter(_bucket(goal_map[task_id]) for task_id in task_ids).items()))

    return {
        "group_count": len(groups),
        "trajectory_count": len(groups) * K,
        "unique_task_count": len(unique_tasks),
        "mixed_task_count": len(mixed_tasks),
        "strict_partial_pair_count": strict_partial_pairs,
        "same_item_strict_partial_pair_count": same_item_pairs,
        "option_contrast_pair_count": option_contrast_pairs,
        "strict_partial_task_count": len(strict_partial_tasks),
        "same_item_strict_partial_task_count": len(same_item_tasks),
        "option_contrast_task_count": len(option_contrast_tasks),
        "missing_public_purchase_state_count": missing_purchase_state,
        "failure_counts": dict(sorted(failure_counts.items())),
        "mixed_task_buckets": bucket_counts(mixed_tasks),
        "same_item_task_buckets": bucket_counts(same_item_tasks),
        "option_contrast_task_buckets": bucket_counts(option_contrast_tasks),
        "same_item_task_ids": sorted(same_item_tasks),
        "option_contrast_task_ids": sorted(option_contrast_tasks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, action="append", required=True)
    parser.add_argument("--prescan-root", type=Path, action="append", required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    goals = json.loads(args.goals.expanduser().resolve().read_text(encoding="utf-8"))
    _require(isinstance(goals, list), "Phase7 goals payload must be a list")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    online_groups, online_bindings = _load_groups(args.online_root, expected_mode="phase4_online_rl_collection")
    prescan_groups, prescan_bindings = _load_groups(args.prescan_root, expected_mode="phase4_data_synthesis")
    online = _analyze_groups(online_groups, goal_map=goal_map)
    prescan = _analyze_groups(prescan_groups, goal_map=goal_map)

    online_same_share = (
        online["same_item_strict_partial_task_count"] / online["mixed_task_count"]
        if online["mixed_task_count"] else 0.0
    )
    online_option_share = (
        online["option_contrast_task_count"] / online["same_item_strict_partial_task_count"]
        if online["same_item_strict_partial_task_count"] else 0.0
    )
    online_coverage_adequate = (
        online_same_share >= ONLINE_SAME_ITEM_MIXED_SHARE
        and online_option_share >= ONLINE_OPTION_STRONG_SHARE
    )
    rebuild_feasible = (
        prescan["same_item_strict_partial_task_count"] >= MINIMUM_REBUILD_SAME_ITEM_TASKS
        and prescan["option_contrast_task_count"] >= MINIMUM_REBUILD_OPTION_TASKS
    )
    if online_coverage_adequate:
        next_action = "stop_data_selection_hypothesis"
    elif rebuild_feasible:
        next_action = "build_20_task_strong_contrast_roster"
    else:
        next_action = "collect_targeted_prescan_before_training"
    report = {
        "schema_version": "m6_phase7_terminal_contrast_diagnostic_v2",
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "public_prebuy_state_only": True,
        "target_asin_or_hidden_answer_used": False,
        "public_item_identity_source": "last_prebuy_policy_action_matching_public_asin",
        "public_option_identity_source": "selected_markers_in_policy_visible_text",
        "online_bindings": online_bindings,
        "prescan_bindings": prescan_bindings,
        "online": online,
        "prescan": prescan,
        "frozen_thresholds": {
            "online_same_item_mixed_share": ONLINE_SAME_ITEM_MIXED_SHARE,
            "online_option_strong_share": ONLINE_OPTION_STRONG_SHARE,
            "minimum_rebuild_same_item_tasks": MINIMUM_REBUILD_SAME_ITEM_TASKS,
            "minimum_rebuild_option_tasks": MINIMUM_REBUILD_OPTION_TASKS,
        },
        "decision": {
            "online_same_item_mixed_share": online_same_share,
            "online_option_strong_share": online_option_share,
            "online_contrast_coverage_adequate": online_coverage_adequate,
            "online_contrast_quality_gap_confirmed": not online_coverage_adequate,
            "prescan_20_task_rebuild_feasible": rebuild_feasible,
            "next_action": next_action,
        },
    }
    report["content_sha256"] = _self_hash(report)
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase7 output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
