"""Frozen multi-turn GRPO and GiGPO-style step-aware credit assignment."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..m4_long_horizon_protocol import (
    CREDIT_ADVANTAGE_EPSILON as ADVANTAGE_EPSILON,
    CREDIT_ANCHOR_VISIT_POLICY as ANCHOR_VISIT_POLICY,
    CREDIT_FORMULA_VERSION,
    CREDIT_MICRO_RETURN_GAMMA as MICRO_RETURN_GAMMA,
    CREDIT_MICRO_WEIGHT as MICRO_ADVANTAGE_WEIGHT,
    GROUP_SIZE,
)
from .contracts import validate_committed_group

BASELINE_METHOD = "multi_turn_grpo"
STEP_AWARE_METHOD = "step_aware_gpo"


def standardized_advantages(
    values: Sequence[float],
    *,
    epsilon: float = ADVANTAGE_EPSILON,
) -> tuple[float, ...]:
    """Population-standardize one comparison group, or return exact zeros."""

    if not values:
        return ()
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("advantage epsilon must be finite and positive")
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("advantage values must be finite")
    mean = sum(normalized) / len(normalized)
    variance = sum((value - mean) ** 2 for value in normalized) / len(normalized)
    standard_deviation = math.sqrt(variance)
    if len(normalized) < 2 or standard_deviation <= epsilon:
        return tuple(0.0 for _ in normalized)
    return tuple((value - mean) / (standard_deviation + epsilon) for value in normalized)


def _first_anchor_visits(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[int, int, float]]]:
    """Collect at most one discounted-return sample per trajectory/anchor."""

    anchors: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for trajectory_index, trajectory in enumerate(trajectories):
        turns = trajectory["turns"]
        reward = float(trajectory["reward"])
        seen: set[str] = set()
        for turn_index, turn in enumerate(turns):
            anchor = turn["anchor_signature"]
            if anchor in seen:
                continue
            seen.add(anchor)
            remaining_turns = len(turns) - turn_index - 1
            discounted_return = reward * (MICRO_RETURN_GAMMA ** remaining_turns)
            anchors[anchor].append((trajectory_index, turn_index, discounted_return))
    return anchors


def assign_group_credit(
    group: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    """Assign one scalar advantage to every generated browser-Agent turn.

    The baseline broadcasts the episode-relative terminal advantage to each
    turn.  The main method adds a standardized discounted-return comparison
    only at the first visit to a public anchor that occurs in at least two
    distinct trajectories and has non-zero return variance.  Every other turn
    falls back exactly to the macro advantage; no process reward is invented.
    """

    if method not in (BASELINE_METHOD, STEP_AWARE_METHOD):
        raise ValueError(f"unsupported focused credit method: {method}")
    validated = validate_committed_group(group)
    trajectories = validated["trajectories"]
    if len(trajectories) != GROUP_SIZE:
        raise ValueError("credit assignment requires exactly K trajectories")
    rewards = [float(trajectory["reward"]) for trajectory in trajectories]
    macro = standardized_advantages(rewards)
    turn_credit = [
        [
            {
                "turn_index": turn_index + 1,
                "anchor_signature": turn["anchor_signature"],
                "macro_advantage": macro[trajectory_index],
                "micro_advantage": 0.0,
                "turn_advantage": macro[trajectory_index],
                "micro_status": "baseline" if method == BASELINE_METHOD else "no_shared_signal",
            }
            for turn_index, turn in enumerate(trajectory["turns"])
        ]
        for trajectory_index, trajectory in enumerate(trajectories)
    ]

    shared_anchor_count = 0
    informative_anchor_count = 0
    first_visit_count = 0
    if method == STEP_AWARE_METHOD:
        anchors = _first_anchor_visits(trajectories)
        for anchor, visits in sorted(anchors.items()):
            if len(visits) < 2:
                continue
            shared_anchor_count += 1
            advantages = standardized_advantages([visit[2] for visit in visits])
            informative = any(abs(value) > 0 for value in advantages)
            if informative:
                informative_anchor_count += 1
            for (trajectory_index, turn_index, _), micro in zip(visits, advantages):
                item = turn_credit[trajectory_index][turn_index]
                item["micro_advantage"] = micro
                item["turn_advantage"] = macro[trajectory_index] + MICRO_ADVANTAGE_WEIGHT * micro
                item["micro_status"] = "informative" if informative else "zero_variance"
                first_visit_count += 1

    flat_count = sum(len(items) for items in turn_credit)
    informative_turn_count = sum(
        abs(item["micro_advantage"]) > 0
        for items in turn_credit
        for item in items
    )
    return {
        "schema_version": "m4_long_horizon_credit_assignment_v1",
        "formula_version": CREDIT_FORMULA_VERSION,
        "method": method,
        "group_id": validated["group_id"],
        "K": GROUP_SIZE,
        "reward_contract": {"success": 1.0, "valid_failure": 0.0, "infrastructure_failure": None},
        "epsilon": ADVANTAGE_EPSILON,
        "gamma": MICRO_RETURN_GAMMA,
        "omega": MICRO_ADVANTAGE_WEIGHT,
        "anchor_visit_policy": ANCHOR_VISIT_POLICY,
        "rewards": rewards,
        "macro_advantages": list(macro),
        "turn_credit": turn_credit,
        "metrics": {
            "turn_count": flat_count,
            "shared_anchor_count": shared_anchor_count,
            "informative_anchor_count": informative_anchor_count,
            "first_visit_anchor_assignment_count": first_visit_count,
            "informative_micro_turn_count": informative_turn_count,
            "informative_micro_turn_fraction": informative_turn_count / flat_count if flat_count else 0.0,
        },
        "normalization_contract": "token mean within turn; turn mean within trajectory; trajectory mean within K group",
    }
