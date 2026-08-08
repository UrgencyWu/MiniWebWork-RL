"""Deterministic cold-coverage then uncertainty-weighted task sampler."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..m4_long_horizon_protocol import (
    TASK_SAMPLER_MINIMUM_WEIGHT as MINIMUM_UNCERTAINTY_WEIGHT,
    TASK_SAMPLER_VERSION as SAMPLER_VERSION,
)


@dataclass(frozen=True, order=True)
class TaskDescriptor:
    task_id: str
    task_family: str
    horizon_stratum: str

    def validate(self) -> None:
        if not self.task_id or not self.task_family:
            raise ValueError("task descriptor requires task id and family")
        if self.horizon_stratum not in {"basic", "medium", "long"}:
            raise ValueError(f"invalid horizon stratum: {self.horizon_stratum}")


@dataclass
class TaskSignal:
    committed_groups: int = 0
    infra_invalid_attempts: int = 0
    valid_trajectories: int = 0
    successes: int = 0

    def validate(self) -> None:
        for field in ("committed_groups", "infra_invalid_attempts", "valid_trajectories", "successes"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid sampler signal field: {field}")
        if self.successes > self.valid_trajectories:
            raise ValueError("task successes exceed valid trajectories")

    @property
    def posterior_success_probability(self) -> float:
        self.validate()
        return (self.successes + 1.0) / (self.valid_trajectories + 2.0)

    @property
    def uncertainty_weight(self) -> float:
        probability = self.posterior_success_probability
        return max(MINIMUM_UNCERTAINTY_WEIGHT, 4.0 * probability * (1.0 - probability))


def _stable_uniform(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 1.0) / (2**64 + 1.0)


class DeterministicSignalSampler:
    """Balance task families while preserving cold coverage and sparse signal.

    Selection never reads dev/test outcomes.  Until every task in a family has
    one committed group, unseen tasks sort ahead of seen tasks.  Thereafter an
    exact Beta(1,1) posterior uncertainty weight strictly prioritizes tasks
    near the success/failure boundary.  A SHA-derived key deterministically
    breaks ties for study seed and iteration.
    """

    def __init__(
        self,
        tasks: Sequence[TaskDescriptor],
        *,
        study_seed: int,
        signals: Mapping[str, TaskSignal | Mapping[str, int]] | None = None,
    ):
        if not isinstance(study_seed, int) or isinstance(study_seed, bool) or study_seed < 0:
            raise ValueError("study seed must be a non-negative integer")
        ordered = sorted(tasks)
        if not ordered or len({task.task_id for task in ordered}) != len(ordered):
            raise ValueError("sampler requires a non-empty unique task roster")
        for task in ordered:
            task.validate()
        self.tasks = tuple(ordered)
        self.study_seed = study_seed
        self.signals = {task.task_id: TaskSignal() for task in ordered}
        for task_id, raw in (signals or {}).items():
            if task_id not in self.signals:
                raise ValueError(f"sampler signal references unknown task: {task_id}")
            signal = raw if isinstance(raw, TaskSignal) else TaskSignal(**dict(raw))
            signal.validate()
            self.signals[task_id] = signal

    def record_committed_group(self, task_id: str, rewards: Sequence[float]) -> None:
        if task_id not in self.signals:
            raise ValueError(f"unknown sampler task: {task_id}")
        normalized = [float(reward) for reward in rewards]
        if not normalized or any(reward not in (0.0, 1.0) for reward in normalized):
            raise ValueError("committed sampler rewards must be a non-empty binary sequence")
        signal = self.signals[task_id]
        signal.committed_groups += 1
        signal.valid_trajectories += len(normalized)
        signal.successes += sum(reward == 1.0 for reward in normalized)

    def record_infra_invalid_attempt(self, task_id: str) -> None:
        if task_id not in self.signals:
            raise ValueError(f"unknown sampler task: {task_id}")
        self.signals[task_id].infra_invalid_attempts += 1

    @staticmethod
    def _family_quotas(families: Sequence[str], limit: int) -> dict[str, int]:
        base, remainder = divmod(limit, len(families))
        return {
            family: base + int(index < remainder)
            for index, family in enumerate(families)
        }

    def _selection_key(self, task: TaskDescriptor, iteration_index: int) -> tuple[int, float, float, str]:
        signal = self.signals[task.task_id]
        cold_rank = 0 if signal.committed_groups == 0 else 1
        tie_break = _stable_uniform(SAMPLER_VERSION, self.study_seed, iteration_index, task.task_id)
        return cold_rank, -signal.uncertainty_weight, tie_break, task.task_id

    def select(self, *, iteration_index: int, limit: int) -> tuple[TaskDescriptor, ...]:
        if not isinstance(iteration_index, int) or isinstance(iteration_index, bool) or iteration_index < 0:
            raise ValueError("iteration index must be a non-negative integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 < limit <= len(self.tasks):
            raise ValueError("sampler limit must be within the task roster")
        by_family: dict[str, list[TaskDescriptor]] = defaultdict(list)
        for task in self.tasks:
            by_family[task.task_family].append(task)
        families = sorted(by_family)
        quotas = self._family_quotas(families, limit)
        selected: list[TaskDescriptor] = []
        selected_ids: set[str] = set()
        for family in families:
            ranked = sorted(
                by_family[family],
                key=lambda task: self._selection_key(task, iteration_index),
            )
            for task in ranked[: quotas[family]]:
                selected.append(task)
                selected_ids.add(task.task_id)
        if len(selected) < limit:
            remaining = sorted(
                (task for task in self.tasks if task.task_id not in selected_ids),
                key=lambda task: self._selection_key(task, iteration_index),
            )
            selected.extend(remaining[: limit - len(selected)])
        if len(selected) != limit or len({task.task_id for task in selected}) != limit:
            raise RuntimeError("sampler failed to produce the requested unique task count")
        return tuple(selected)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "m4_long_horizon_sampler_state_v1",
            "sampler_version": SAMPLER_VERSION,
            "study_seed": self.study_seed,
            "task_count": len(self.tasks),
            "signals": {
                task_id: {
                    "committed_groups": signal.committed_groups,
                    "infra_invalid_attempts": signal.infra_invalid_attempts,
                    "valid_trajectories": signal.valid_trajectories,
                    "successes": signal.successes,
                    "posterior_success_probability": signal.posterior_success_probability,
                    "uncertainty_weight": signal.uncertainty_weight,
                }
                for task_id, signal in sorted(self.signals.items())
            },
        }
