"""Dependency-light registry for the five preregistered M4 methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AlgorithmId = Literal["sft", "rsft", "rloo", "grpo", "gspo"]
OFFLINE_ALGORITHMS = frozenset({"sft", "rsft"})
ONLINE_ALGORITHMS = frozenset({"rloo", "grpo", "gspo"})
ALL_ALGORITHMS = OFFLINE_ALGORITHMS | ONLINE_ALGORITHMS


@dataclass(frozen=True)
class AlgorithmSpec:
    algorithm_id: AlgorithmId
    regime: Literal["offline", "online"]
    advantage_estimator: str
    likelihood_ratio: str
    requires_strict_on_policy_rollouts: bool


ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "sft": AlgorithmSpec("sft", "offline", "none", "none", False),
    "rsft": AlgorithmSpec("rsft", "offline", "verified_best_of_n", "none", False),
    "rloo": AlgorithmSpec("rloo", "online", "leave_one_out", "per_action_token", True),
    "grpo": AlgorithmSpec("grpo", "online", "group_normalized", "per_action_token", True),
    "gspo": AlgorithmSpec("gspo", "online", "group_normalized", "per_trajectory_sequence", True),
}


def get_algorithm_spec(algorithm_id: str) -> AlgorithmSpec:
    """Return a declared M4 method or reject aliases/undeclared variants."""
    try:
        return ALGORITHM_SPECS[algorithm_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported M4 algorithm {algorithm_id!r}; expected one of {sorted(ALL_ALGORITHMS)}"
        ) from exc
