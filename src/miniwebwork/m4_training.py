"""Algorithm-neutral M4 training-data planning with strict audit gates.

This module is deliberately CPU-only.  GPU scripts consume its immutable plan
instead of deciding ad hoc which rollout groups or RSFT traces to train on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .m4_algorithms import get_algorithm_spec
from .rl.batch import ReplayGroup, build_replay_group


def _record_action_tokens(record: Any) -> int:
    return sum(len(step.generated_token_ids) for step in record.steps)


@dataclass(frozen=True)
class OnlineUpdateGroupPlan:
    task_id: str
    included: bool
    reason: str
    action_token_count: int
    trajectory_count: int
    replay_group: ReplayGroup | None = None

    def audit_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Replay evidence is already retained in the source artifact.  Avoid
        # duplicating token IDs and prompts inside every plan/report.
        result["replay_group"] = (
            {
                "task_id": self.replay_group.task_id,
                "policy": self.replay_group.policy,
                "temperature": self.replay_group.temperature,
                "top_p": self.replay_group.top_p,
                "top_k": self.replay_group.top_k,
                "max_raw_sampling_logprob_abs_diff": self.replay_group.max_raw_sampling_logprob_abs_diff,
                "logprob_match_tolerance": self.replay_group.logprob_match_tolerance,
                "trajectory_count": len(self.replay_group.trajectories),
            }
            if self.replay_group is not None
            else None
        )
        return result


@dataclass(frozen=True)
class OnlineUpdatePlan:
    algorithm_id: str
    action_token_budget: int
    selected_action_tokens: int
    selected_groups: int
    skipped_groups: int
    groups: tuple[OnlineUpdateGroupPlan, ...]

    def audit_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "action_token_budget": self.action_token_budget,
            "selected_action_tokens": self.selected_action_tokens,
            "selected_groups": self.selected_groups,
            "skipped_groups": self.skipped_groups,
            "groups": [group.audit_dict() for group in self.groups],
        }


def build_online_update_plan(
    records: list[Any],
    algorithm_id: str,
    *,
    action_token_budget: int,
    logprob_match_tolerance: float = 0.05,
) -> OnlineUpdatePlan:
    """Select valid same-task online updates in deterministic task-ID order.

    Groups without mixed reward, strict log-probability compatibility, or a
    full valid replay are recorded as skipped.  They are not silently dropped,
    and they never become a fake zero-reward update.  The token cap is applied
    to selected, differentiable action tokens only.
    """
    algorithm = get_algorithm_spec(algorithm_id)
    if algorithm.regime != "online":
        raise ValueError(f"{algorithm_id} is offline and cannot build an online update plan")
    if action_token_budget <= 0:
        raise ValueError("action_token_budget must be positive")
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(record.task_id, []).append(record)

    total_tokens = 0
    plans: list[OnlineUpdateGroupPlan] = []
    for task_id in sorted(grouped):
        task_records = grouped[task_id]
        action_tokens = sum(_record_action_tokens(record) for record in task_records)
        try:
            replay_group = build_replay_group(
                task_records,
                logprob_match_tolerance=logprob_match_tolerance,
            )
            # Materialize the selected algorithm's estimator now. This catches
            # unknown IDs and records that cannot support the declared method.
            replay_group.advantages_for(algorithm_id)
        except ValueError as exc:
            plans.append(
                OnlineUpdateGroupPlan(
                    task_id=task_id,
                    included=False,
                    reason=f"ineligible:{exc}",
                    action_token_count=action_tokens,
                    trajectory_count=len(task_records),
                )
            )
            continue
        selected_tokens = sum(
            trajectory.action_token_count for trajectory in replay_group.trajectories
        )
        if total_tokens + selected_tokens > action_token_budget:
            plans.append(
                OnlineUpdateGroupPlan(
                    task_id=task_id,
                    included=False,
                    reason="budget_exhausted",
                    action_token_count=selected_tokens,
                    trajectory_count=len(replay_group.trajectories),
                    replay_group=replay_group,
                )
            )
            continue
        total_tokens += selected_tokens
        plans.append(
            OnlineUpdateGroupPlan(
                task_id=task_id,
                included=True,
                reason="selected",
                action_token_count=selected_tokens,
                trajectory_count=len(replay_group.trajectories),
                replay_group=replay_group,
            )
        )
    return OnlineUpdatePlan(
        algorithm_id=algorithm_id,
        action_token_budget=action_token_budget,
        selected_action_tokens=total_tokens,
        selected_groups=sum(plan.included for plan in plans),
        skipped_groups=sum(not plan.included for plan in plans),
        groups=tuple(plans),
    )


@dataclass(frozen=True)
class RSFTSelection:
    task_id: str
    selected_episode_id: str | None
    selected_rollout_index: int | None
    candidate_count: int
    valid_success_count: int
    reason: str


def select_rsft_rollouts(records: list[Any]) -> tuple[dict[str, Any], ...]:
    """Select one verified train rollout per task with deterministic Best-of-N.

    All valid successes have the same terminal reward in RLVR.  The declared
    tie-break is fewer action tokens, then rollout index, then episode ID.  A
    task without a verifier-successful candidate remains represented in the
    audit output, but contributes no RSFT training example.
    """
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(record.task_id, []).append(record)
    selections: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        candidates = grouped[task_id]
        successes = [
            record
            for record in candidates
            if record.rollout_valid and record.success and float(record.reward) == 1.0
        ]
        if not successes:
            selection = RSFTSelection(
                task_id=task_id,
                selected_episode_id=None,
                selected_rollout_index=None,
                candidate_count=len(candidates),
                valid_success_count=0,
                reason="no_verified_success",
            )
            selections.append({"selection": asdict(selection), "record": None})
            continue
        winner = min(
            successes,
            key=lambda record: (
                _record_action_tokens(record),
                record.rollout_index,
                record.episode_id,
            ),
        )
        selection = RSFTSelection(
            task_id=task_id,
            selected_episode_id=winner.episode_id,
            selected_rollout_index=winner.rollout_index,
            candidate_count=len(candidates),
            valid_success_count=len(successes),
            reason="selected_verified_best_of_n",
        )
        selections.append({"selection": asdict(selection), "record": winner})
    return tuple(selections)
