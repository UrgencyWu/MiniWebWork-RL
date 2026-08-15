#!/usr/bin/env python3
"""Check whether public successful prefixes can support shared-item suffix rollouts."""

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
from m6_phase7_terminal_contrast_diagnostic import (  # noqa: E402
    FORBIDDEN_PATH_PARTS,
    PUBLIC_ASIN_ACTION_RE,
    _load_groups,
)
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402

MINIMUM_REPLAYABLE_TASKS = 20
MINIMUM_BRANCHABLE_TASKS = 12
PRODUCT_PAGE_TYPES = {"item", "product"}
PUBLIC_CLICK_RE = re.compile(r"^click\[(.*)\]$", re.IGNORECASE)
NAVIGATION_LABELS = {
    "description",
    "features",
    "reviews",
    "back to search",
    "buy now",
    "previous",
    "prev",
    "next",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _goal_has_options(goal: Mapping[str, Any]) -> bool:
    options = goal.get("goal_options")
    return isinstance(options, (list, tuple, set, dict)) and bool(options)


def _is_navigation_label(label: str) -> bool:
    normalized = _normalized(label)
    if normalized in NAVIGATION_LABELS:
        return True
    if normalized.isdigit():
        return True
    return any(
        normalized.startswith(prefix)
        for prefix in ("page ", "next ", "prev ", "previous ", "< prev", "next >")
    )


def _option_actions(available_actions: Any) -> tuple[str, ...]:
    if not isinstance(available_actions, list):
        return ()
    output: set[str] = set()
    for raw in available_actions:
        command = str(raw or "").strip()
        match = PUBLIC_CLICK_RE.fullmatch(command)
        if match is None or PUBLIC_ASIN_ACTION_RE.fullmatch(command) is not None:
            continue
        if _is_navigation_label(match.group(1)):
            continue
        output.add(command)
    return tuple(sorted(output, key=lambda value: (_normalized(value), value)))


def _strict_prefix_candidate(trajectory: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a public, successful prefix ending at an ASIN click."""

    if not bool(trajectory.get("success")) or trajectory.get("termination_reason") != "purchase":
        return None
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        return None
    buy_index = next(
        (
            index
            for index in range(len(turns) - 1, -1, -1)
            if _normalized((turns[index].get("action") or {}).get("command")) == "click[buy now]"
        ),
        None,
    )
    if buy_index is None:
        return None
    asin_index = next(
        (
            index
            for index in range(buy_index - 1, -1, -1)
            if PUBLIC_ASIN_ACTION_RE.fullmatch(str((turns[index].get("action") or {}).get("command", "")).strip())
        ),
        None,
    )
    if asin_index is None or asin_index + 1 >= len(turns):
        return None
    prefix_turns = turns[: asin_index + 1]
    if not all((turn.get("action_result") or {}).get("success") is True for turn in prefix_turns):
        return None
    commands = [str((turn.get("action") or {}).get("command", "")).strip() for turn in prefix_turns]
    if not all(commands):
        return None
    product_observation = turns[asin_index + 1].get("observation")
    if not isinstance(product_observation, Mapping):
        return None
    if _normalized(product_observation.get("page_type")) not in PRODUCT_PAGE_TYPES:
        return None
    available_actions = product_observation.get("available_actions")
    if not isinstance(available_actions, list) or not available_actions:
        return None
    branches = _option_actions(available_actions)
    return {
        "prefix_action_count": len(commands),
        "prefix_action_sequence_sha256": sha256_json(commands),
        "public_option_action_count": len(branches),
        "public_option_action_set_sha256": sha256_json(list(branches)),
    }


def _best_task_candidates(
    groups: Sequence[Mapping[str, Any]],
    *,
    goal_map: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str], Counter[str]]:
    best: dict[str, dict[str, Any]] = {}
    strict_tasks: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    for group in groups:
        task_id = str(group["task_id"])
        _require(task_id in goal_map, f"Phase8 task missing from goals: {task_id}")
        strict = [trajectory for trajectory in group["trajectories"] if bool(trajectory.get("success"))]
        if not strict:
            continue
        strict_tasks.add(task_id)
        if not _goal_has_options(goal_map[task_id]):
            rejection_counts["goal_without_options"] += 1
            continue
        task_candidates = [candidate for trajectory in strict if (candidate := _strict_prefix_candidate(trajectory))]
        if not task_candidates:
            rejection_counts["no_public_successful_product_prefix"] += 1
            continue
        candidate = min(
            task_candidates,
            key=lambda value: (
                value["prefix_action_count"],
                -value["public_option_action_count"],
                value["prefix_action_sequence_sha256"],
            ),
        )
        previous = best.get(task_id)
        if previous is None or (
            candidate["prefix_action_count"],
            -candidate["public_option_action_count"],
            candidate["prefix_action_sequence_sha256"],
        ) < (
            previous["prefix_action_count"],
            -previous["public_option_action_count"],
            previous["prefix_action_sequence_sha256"],
        ):
            best[task_id] = candidate
    return best, strict_tasks, rejection_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prescan-root", type=Path, action="append", required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in [*args.prescan_root, args.goals, args.output]:
        lowered = {part.casefold() for part in path.expanduser().resolve().parts}
        _require(not (lowered & FORBIDDEN_PATH_PARTS), "Phase8 input/output touches promotion/holdout")
    goals = json.loads(args.goals.expanduser().resolve().read_text(encoding="utf-8"))
    _require(isinstance(goals, list), "Phase8 goals payload must be a list")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    groups, bindings = _load_groups(args.prescan_root, expected_mode="phase4_data_synthesis")
    best, strict_tasks, rejection_counts = _best_task_candidates(groups, goal_map=goal_map)
    replayable = set(best)
    branchable = {
        task_id for task_id, candidate in best.items() if candidate["public_option_action_count"] >= 2
    }

    def buckets(task_ids: set[str]) -> dict[str, int]:
        return dict(sorted(Counter(_bucket(goal_map[task_id]) for task_id in task_ids).items()))

    passed = len(replayable) >= MINIMUM_REPLAYABLE_TASKS and len(branchable) >= MINIMUM_BRANCHABLE_TASKS
    prefixes = [dict(task_id=task_id, **best[task_id]) for task_id in sorted(best)]
    report = {
        "schema_version": "m6_phase8_prefix_reset_feasibility_v1",
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "public_fields_only": True,
        "target_asin_or_hidden_answer_used": False,
        "prefix_ends_at_last_public_asin_click": True,
        "post_action_internal_state_used": False,
        "source_bindings": bindings,
        "group_count": len(groups),
        "trajectory_count": sum(len(group["trajectories"]) for group in groups),
        "unique_strict_task_count": len(strict_tasks),
        "replayable_task_count": len(replayable),
        "branchable_task_count": len(branchable),
        "replayable_task_ids": sorted(replayable),
        "branchable_task_ids": sorted(branchable),
        "replayable_task_buckets": buckets(replayable),
        "branchable_task_buckets": buckets(branchable),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "public_prefixes": prefixes,
        "frozen_thresholds": {
            "minimum_replayable_tasks": MINIMUM_REPLAYABLE_TASKS,
            "minimum_branchable_tasks": MINIMUM_BRANCHABLE_TASKS,
            "minimum_public_option_actions_per_branchable_task": 2,
        },
        "decision": {
            "prefix_reset_feasible": passed,
            "next_action": (
                "design_shared_prefix_k4_suffix_rollout" if passed else "stop_prefix_reset_direction"
            ),
        },
    }
    report["content_sha256"] = _self_hash(report)
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase8 output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
