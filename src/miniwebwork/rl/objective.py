"""Loss functions for grouped multi-turn browser trajectories.

A browser trajectory contains multiple model turns.  Each turn has its own
prompt and generated JSON action, so the model forward pass is performed per
turn.  The resulting action-token log-probabilities are concatenated per
trajectory, padded, and supplied to this module.  One terminal group-relative
advantage is broadcast across all action tokens in the same trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class NoRewardVarianceError(ValueError):
    """Raised when a GRPO group contains no relative learning signal."""


@dataclass
class PolicyLossResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor
    clip_fraction: torch.Tensor
    approximate_kl: torch.Tensor
    mean_ratio: torch.Tensor
    valid_token_count: int
    trajectory_count: int


def group_relative_advantages(
    rewards: torch.Tensor,
    epsilon: float = 1e-8,
    *,
    require_variance: bool = True,
) -> torch.Tensor:
    """Normalize one reward per trajectory within a same-task rollout group.

    Parameters
    ----------
    rewards:
        Rank-1 finite tensor containing rewards from valid policy rollouts.
        Infrastructure failures must already have been filtered out.
    epsilon:
        Numerical stabilizer.
    require_variance:
        When true, reject groups whose population standard deviation is zero.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be rank 1, got {tuple(rewards.shape)}")
    if rewards.numel() < 2:
        raise ValueError("GRPO requires at least two valid trajectories per group")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards contain NaN or Inf")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    rewards = rewards.float()
    standard_deviation = rewards.std(unbiased=False)
    if require_variance and standard_deviation <= epsilon:
        raise NoRewardVarianceError("rollout group has zero reward variance")
    return (rewards - rewards.mean()) / (standard_deviation + epsilon)


def _validate_policy_tensors(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    reference_logprobs: torch.Tensor | None,
) -> None:
    if current_logprobs.ndim != 2:
        raise ValueError("current_logprobs must have shape [trajectories, tokens]")
    if old_logprobs.shape != current_logprobs.shape:
        raise ValueError("old_logprobs shape must match current_logprobs")
    if token_mask.shape != current_logprobs.shape:
        raise ValueError("token_mask shape must match logprobs")
    if advantages.ndim != 1 or advantages.shape[0] != current_logprobs.shape[0]:
        raise ValueError("advantages must have one value per trajectory")
    if reference_logprobs is not None and reference_logprobs.shape != current_logprobs.shape:
        raise ValueError("reference_logprobs shape must match current_logprobs")
    if current_logprobs.shape[0] < 2:
        raise ValueError("policy loss requires at least two trajectories")
    if token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean")
    if not token_mask.any():
        raise ValueError("token_mask contains no trainable action token")

    for name, tensor in (
        ("current_logprobs", current_logprobs),
        ("old_logprobs", old_logprobs),
        ("advantages", advantages),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if reference_logprobs is not None and not torch.isfinite(reference_logprobs).all():
        raise ValueError("reference_logprobs contains NaN or Inf")

    token_counts = token_mask.sum(dim=1)
    if (token_counts == 0).any():
        raise ValueError("every trajectory must contain at least one action token")


def clipped_trajectory_policy_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    reference_logprobs: torch.Tensor | None = None,
    kl_beta: float = 0.0,
) -> PolicyLossResult:
    """Compute a trajectory-normalized PPO/GRPO-style token objective.

    Each row represents all model-generated action tokens from one browser
    trajectory, concatenated across turns.  The terminal trajectory advantage
    is applied to every action token in that row.  Per-token objectives are
    averaged *within each trajectory*, then trajectories are averaged equally;
    long trajectories therefore do not dominate solely because they contain
    more action tokens.

    ``old_logprobs`` must describe the frozen behavior/policy distribution used
    for the batch.  For the first strictly on-policy pilot, use a fixed
    generation distribution and recompute current log-probabilities under the
    same temperature convention.  Do not mix raw model log-probabilities with
    post-top-p sampling log-probabilities.
    """
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if kl_beta < 0:
        raise ValueError("kl_beta must be non-negative")

    _validate_policy_tensors(
        current_logprobs,
        old_logprobs,
        advantages,
        token_mask,
        reference_logprobs,
    )

    mask = token_mask.to(dtype=current_logprobs.dtype)
    detached_old = old_logprobs.detach()
    detached_advantages = advantages.detach().unsqueeze(1)
    log_ratio = current_logprobs - detached_old
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)

    unclipped_objective = ratio * detached_advantages
    clipped_objective = clipped_ratio * detached_advantages
    token_objective = torch.minimum(unclipped_objective, clipped_objective)

    tokens_per_trajectory = mask.sum(dim=1)
    trajectory_policy_loss = -(
        (token_objective * mask).sum(dim=1) / tokens_per_trajectory
    )
    policy_loss = trajectory_policy_loss.mean()

    if reference_logprobs is not None and kl_beta > 0:
        # Non-negative k3 estimator for KL(current || reference):
        # exp(ref-current) - (ref-current) - 1.
        reference_log_ratio = reference_logprobs.detach() - current_logprobs
        token_kl = torch.exp(reference_log_ratio) - reference_log_ratio - 1.0
        trajectory_kl = (token_kl * mask).sum(dim=1) / tokens_per_trajectory
        reference_kl = trajectory_kl.mean()
    else:
        reference_kl = current_logprobs.new_zeros(())

    loss = policy_loss + kl_beta * reference_kl
    clipped = (ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)
    valid_token_count = int(token_mask.sum().item())
    clip_fraction = (clipped.to(mask.dtype) * mask).sum() / mask.sum()
    # Standard non-negative approximate KL between old and current policy.
    approximate_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / mask.sum()
    mean_ratio = (ratio * mask).sum() / mask.sum()

    for name, value in (
        ("loss", loss),
        ("policy_loss", policy_loss),
        ("reference_kl", reference_kl),
        ("clip_fraction", clip_fraction),
        ("approximate_kl", approximate_kl),
        ("mean_ratio", mean_ratio),
    ):
        if not torch.isfinite(value):
            raise FloatingPointError(f"{name} is NaN or Inf")

    return PolicyLossResult(
        loss=loss,
        policy_loss=policy_loss,
        reference_kl=reference_kl,
        clip_fraction=clip_fraction,
        approximate_kl=approximate_kl,
        mean_ratio=mean_ratio,
        valid_token_count=valid_token_count,
        trajectory_count=current_logprobs.shape[0],
    )
