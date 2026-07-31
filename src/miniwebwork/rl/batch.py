"""Convert validated multi-turn rollout groups into policy-update inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..rollout import RolloutRecord, RolloutStep, summarize_group
from .objective import group_relative_advantages


@dataclass(frozen=True)
class TurnReplay:
    """Exact conditional-generation evidence for one browser turn."""

    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    old_policy_logprobs: tuple[float, ...]
    sampling_logprobs: tuple[float, ...]

    @classmethod
    def from_step(cls, step: RolloutStep) -> "TurnReplay":
        step.validate()
        if not step.generated_token_ids:
            raise ValueError(f"turn {step.turn} contains no generated completion token")
        if not step.prompt_token_ids:
            raise ValueError(f"turn {step.turn} contains no prompt token evidence")
        if len(step.token_logprobs) != len(step.generated_token_ids):
            raise ValueError(f"turn {step.turn} raw policy log-probabilities are incomplete")
        if len(step.sampling_logprobs) != len(step.generated_token_ids):
            raise ValueError(f"turn {step.turn} sampling log-probabilities are incomplete")
        return cls(
            prompt_token_ids=tuple(step.prompt_token_ids),
            completion_token_ids=tuple(step.generated_token_ids),
            old_policy_logprobs=tuple(float(value) for value in step.token_logprobs),
            sampling_logprobs=tuple(float(value) for value in step.sampling_logprobs),
        )


@dataclass(frozen=True)
class TrajectoryReplay:
    """One trajectory and the group-relative advantage applied to its actions."""

    task_id: str
    episode_id: str
    rollout_seed: int
    reward: float
    advantage: float
    turns: tuple[TurnReplay, ...]

    @property
    def action_token_count(self) -> int:
        return sum(len(turn.completion_token_ids) for turn in self.turns)

    @property
    def old_policy_logprobs(self) -> tuple[float, ...]:
        return tuple(
            value
            for turn in self.turns
            for value in turn.old_policy_logprobs
        )


@dataclass(frozen=True)
class ReplayGroup:
    """Same-task, same-policy trajectories ready for a strict update."""

    task_id: str
    policy: str
    temperature: float
    top_p: float
    top_k: int
    trajectories: tuple[TrajectoryReplay, ...]

    @property
    def advantages(self) -> torch.Tensor:
        return torch.tensor(
            [trajectory.advantage for trajectory in self.trajectories],
            dtype=torch.float32,
        )


def build_replay_group(
    records: list[RolloutRecord],
    *,
    update_distribution_compatible: bool,
) -> ReplayGroup:
    """Validate a rollout group and attach group-relative advantages.

    Diagnostic truncated groups must pass
    ``update_distribution_compatible=False`` and are rejected here even when
    they contain mixed rewards. Every generated turn in an update batch must
    retain both raw-policy and actual sampling-distribution probabilities.
    """
    summary = summarize_group(
        records,
        requested_k=len(records),
        update_distribution_compatible=update_distribution_compatible,
    )
    if not summary.has_learning_signal:
        raise ValueError("rollout group has no mixed-reward learning signal")
    if not summary.valid_for_grpo_update:
        raise ValueError("rollout group distribution is not compatible with update")

    valid_records = [record for record in records if record.rollout_valid]
    rewards = torch.tensor(
        [float(record.reward) for record in valid_records],
        dtype=torch.float32,
    )
    advantages = group_relative_advantages(rewards)

    trajectories: list[TrajectoryReplay] = []
    for record, advantage in zip(valid_records, advantages.tolist()):
        record.validate()
        turns = tuple(
            TurnReplay.from_step(step)
            for step in record.steps
            if step.generated_token_ids
        )
        if not turns:
            raise ValueError(
                f"trajectory {record.episode_id} contains no generated action token"
            )
        trajectories.append(
            TrajectoryReplay(
                task_id=record.task_id,
                episode_id=record.episode_id,
                rollout_seed=record.rollout_seed,
                reward=float(record.reward),
                advantage=float(advantage),
                turns=turns,
            )
        )

    return ReplayGroup(
        task_id=summary.task_id,
        policy=summary.policy,
        temperature=summary.temperature,
        top_p=summary.top_p,
        top_k=summary.top_k,
        trajectories=tuple(trajectories),
    )


def pad_trajectory_logprobs(
    sequences: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad rank-1 trajectory log-probability sequences and return a mask."""
    if not sequences:
        raise ValueError("cannot pad an empty sequence list")
    if any(sequence.ndim != 1 for sequence in sequences):
        raise ValueError("every log-probability sequence must be rank 1")
    if any(sequence.numel() == 0 for sequence in sequences):
        raise ValueError("every trajectory must contain at least one action token")
    if any(not torch.isfinite(sequence).all() for sequence in sequences):
        raise ValueError("log-probability sequence contains NaN or Inf")

    device = sequences[0].device
    dtype = sequences[0].dtype
    if any(sequence.device != device for sequence in sequences):
        raise ValueError("all sequences must be on the same device")
    if any(sequence.dtype != dtype for sequence in sequences):
        raise ValueError("all sequences must have the same dtype")

    maximum = max(int(sequence.numel()) for sequence in sequences)
    padded = torch.zeros((len(sequences), maximum), dtype=dtype, device=device)
    mask = torch.zeros((len(sequences), maximum), dtype=torch.bool, device=device)
    for row, sequence in enumerate(sequences):
        length = int(sequence.numel())
        padded[row, :length] = sequence
        mask[row, :length] = True
    return padded, mask


def old_policy_batch(group: ReplayGroup) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded raw old-policy log-probabilities and action-token mask."""
    sequences = [
        torch.tensor(trajectory.old_policy_logprobs, dtype=torch.float32)
        for trajectory in group.trajectories
    ]
    return pad_trajectory_logprobs(sequences)
