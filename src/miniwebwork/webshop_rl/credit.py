"""Frozen public-state credit contract for the M5 WebShop study.

The state used for GiGPO-style grouping is deliberately not the full model
prompt.  WebShop is an MDP: rollouts that reach the same task-scoped public
state can be compared even when their earlier action histories differ.  Exact
prompt tokens remain separately hashed so behavior-policy evidence is still
auditable without destroying state convergence.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..long_horizon_rl.contracts import sha256_json, token_ids_sha256

BASELINE_METHOD = "multi_turn_grpo"
ANCHOR_METHOD = "anchor_gigpo"
CREDIT_FORMULA_VERSION = "webshop_public_state_macro_micro_v1"
GROUP_SIZE = 4
ADVANTAGE_EPSILON = 1e-6
MICRO_RETURN_GAMMA = 0.95
MICRO_ADVANTAGE_WEIGHT = 1.0
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_credit_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact task-scoped MDP state visible before an action.

    Policy-only context (step number and recent history), runtime identity,
    target metadata, verifier output, and unknown fields are excluded.  The
    action list is retained because it defines which transitions are publicly
    executable from this state.
    """

    _require(isinstance(observation, Mapping), "WebShop observation must be a mapping")
    actions = observation.get("available_actions")
    _require(isinstance(actions, (list, tuple)), "WebShop public action list is missing")
    normalized_actions: list[str] = []
    for action in actions:
        _require(isinstance(action, str) and action, "WebShop public action is malformed")
        _require(action not in normalized_actions, "WebShop public action list contains a duplicate")
        normalized_actions.append(action)
    required_strings = {
        field: observation.get(field)
        for field in ("schema_version", "task_id", "instruction", "page_type", "visible_text")
    }
    for field, value in required_strings.items():
        _require(isinstance(value, str) and value, f"WebShop credit state lacks {field}")
    visible_text = required_strings["visible_text"]
    _require(
        isinstance(visible_text, str) and len(visible_text) <= 8000,
        "WebShop credit state exceeds the prompt-visible text bound",
    )
    return {
        **required_strings,
        "text_truncated": bool(observation.get("text_truncated", False)),
        "available_actions": normalized_actions,
        "terminal": bool(observation.get("terminal", False)),
    }


def public_state_anchor_signature(observation: Mapping[str, Any]) -> str:
    """Hash only the exact public state used for within-task credit groups."""

    return sha256_json(
        {
            "anchor_schema": "m5_webshop_public_state_anchor_v1",
            "public_state": canonical_credit_state(observation),
        }
    )


def policy_context_signature(prompt_token_ids: Sequence[int]) -> str:
    """Hash the exact policy input separately from the credit anchor."""

    return token_ids_sha256(prompt_token_ids)


def standardized_advantages(
    values: Sequence[float],
    *,
    epsilon: float = ADVANTAGE_EPSILON,
) -> tuple[float, ...]:
    """Population-standardize one K-group, returning exact zero on no signal."""

    _require(math.isfinite(epsilon) and epsilon > 0, "credit epsilon must be finite and positive")
    normalized = tuple(float(value) for value in values)
    _require(all(math.isfinite(value) for value in normalized), "credit values must be finite")
    if not normalized:
        return ()
    mean = sum(normalized) / len(normalized)
    variance = sum((value - mean) ** 2 for value in normalized) / len(normalized)
    standard_deviation = math.sqrt(variance)
    if len(normalized) < 2 or standard_deviation <= epsilon:
        return tuple(0.0 for _ in normalized)
    return tuple((value - mean) / (standard_deviation + epsilon) for value in normalized)


def _validate_trajectories(trajectories: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    _require(len(trajectories) == GROUP_SIZE, "M5 credit assignment requires exactly K=4 trajectories")
    task_ids: set[str] = set()
    validated: list[Mapping[str, Any]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        _require(isinstance(trajectory, Mapping), f"trajectory {trajectory_index} is malformed")
        task_id = trajectory.get("task_id")
        _require(isinstance(task_id, str) and task_id, f"trajectory {trajectory_index} lacks task_id")
        task_ids.add(task_id)
        reward = trajectory.get("reward")
        _require(
            isinstance(reward, (int, float)) and not isinstance(reward, bool) and float(reward) in (0.0, 1.0),
            f"trajectory {trajectory_index} reward is not binary",
        )
        turns = trajectory.get("turns")
        _require(isinstance(turns, list) and turns, f"trajectory {trajectory_index} has no turns")
        for turn_index, turn in enumerate(turns):
            _require(isinstance(turn, Mapping), f"trajectory {trajectory_index} turn {turn_index} is malformed")
            anchor = turn.get("public_state_anchor_sha256")
            _require(
                isinstance(anchor, str) and SHA256_RE.fullmatch(anchor) is not None,
                f"trajectory {trajectory_index} turn {turn_index} lacks a public-state anchor",
            )
        validated.append(trajectory)
    _require(len(task_ids) == 1, "M5 credit group crosses task identities")
    return tuple(validated)


def _first_anchor_visits(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[int, int, float]]]:
    """Collect one discounted-return sample per trajectory and public state."""

    anchors: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for trajectory_index, trajectory in enumerate(trajectories):
        turns = trajectory["turns"]
        terminal_reward = float(trajectory["reward"])
        seen: set[str] = set()
        for turn_index, turn in enumerate(turns):
            anchor = str(turn["public_state_anchor_sha256"])
            if anchor in seen:
                continue
            seen.add(anchor)
            remaining_actions = len(turns) - turn_index - 1
            discounted_return = terminal_reward * (MICRO_RETURN_GAMMA**remaining_actions)
            anchors[anchor].append((trajectory_index, turn_index, discounted_return))
    return anchors


def assign_group_credit(
    trajectories: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    """Assign macro GRPO and optional first-visit public-state micro credit."""

    _require(method in {BASELINE_METHOD, ANCHOR_METHOD}, "unsupported M5 online method")
    validated = _validate_trajectories(trajectories)
    rewards = tuple(float(trajectory["reward"]) for trajectory in validated)
    macro = standardized_advantages(rewards)
    turn_credit = [
        [
            {
                "turn_index": turn_index + 1,
                "public_state_anchor_sha256": turn["public_state_anchor_sha256"],
                "macro_advantage": macro[trajectory_index],
                "micro_advantage": 0.0,
                "turn_advantage": macro[trajectory_index],
                "micro_status": "baseline" if method == BASELINE_METHOD else "no_shared_signal",
            }
            for turn_index, turn in enumerate(trajectory["turns"])
        ]
        for trajectory_index, trajectory in enumerate(validated)
    ]

    shared_anchor_count = 0
    informative_anchor_count = 0
    non_initial_shared_anchor_count = 0
    non_initial_informative_anchor_count = 0
    first_visit_assignment_count = 0
    if method == ANCHOR_METHOD:
        for anchor, visits in sorted(_first_anchor_visits(validated).items()):
            distinct_trajectories = {trajectory_index for trajectory_index, _, _ in visits}
            if len(distinct_trajectories) < 2:
                continue
            shared_anchor_count += 1
            non_initial = sum(turn_index > 0 for _, turn_index, _ in visits) >= 2
            non_initial_shared_anchor_count += int(non_initial)
            micro_values = standardized_advantages([discounted_return for _, _, discounted_return in visits])
            informative = any(abs(value) > 0 for value in micro_values)
            informative_anchor_count += int(informative)
            non_initial_informative_anchor_count += int(non_initial and informative)
            for (trajectory_index, turn_index, _), micro in zip(visits, micro_values):
                item = turn_credit[trajectory_index][turn_index]
                item["micro_advantage"] = micro
                item["turn_advantage"] = macro[trajectory_index] + MICRO_ADVANTAGE_WEIGHT * micro
                item["micro_status"] = "informative" if informative else "zero_variance"
                first_visit_assignment_count += 1

    flat = [item for trajectory_credit in turn_credit for item in trajectory_credit]
    informative_turn_count = sum(abs(float(item["micro_advantage"])) > 0 for item in flat)
    non_initial_informative_turn_count = sum(
        turn_index > 0 and abs(float(item["micro_advantage"])) > 0
        for trajectory_credit in turn_credit
        for turn_index, item in enumerate(trajectory_credit)
    )
    return {
        "schema_version": "m5_webshop_credit_assignment_v1",
        "formula_version": CREDIT_FORMULA_VERSION,
        "method": method,
        "task_id": validated[0]["task_id"],
        "K": GROUP_SIZE,
        "epsilon": ADVANTAGE_EPSILON,
        "gamma": MICRO_RETURN_GAMMA,
        "omega": MICRO_ADVANTAGE_WEIGHT,
        "anchor_visit_policy": "first_visit_per_trajectory",
        "rewards": list(rewards),
        "macro_advantages": list(macro),
        "turn_credit": turn_credit,
        "metrics": {
            "turn_count": len(flat),
            "shared_anchor_count": shared_anchor_count,
            "informative_anchor_count": informative_anchor_count,
            "non_initial_shared_anchor_count": non_initial_shared_anchor_count,
            "non_initial_informative_anchor_count": non_initial_informative_anchor_count,
            "first_visit_anchor_assignment_count": first_visit_assignment_count,
            "informative_micro_turn_count": informative_turn_count,
            "informative_micro_turn_fraction": informative_turn_count / len(flat),
            "non_initial_informative_micro_turn_count": non_initial_informative_turn_count,
            "non_initial_informative_micro_turn_fraction": non_initial_informative_turn_count / len(flat),
        },
        "normalization_contract": "population standardization within the same task K4 group",
    }
