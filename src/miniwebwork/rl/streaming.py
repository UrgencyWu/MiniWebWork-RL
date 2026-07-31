"""Memory-bounded loss for streaming multi-turn trajectory updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StreamingTrajectoryLoss:
    loss: torch.Tensor
    clip_fraction: torch.Tensor
    approximate_kl: torch.Tensor
    mean_ratio: torch.Tensor
    token_count: int


def clipped_single_trajectory_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: torch.Tensor | float,
    *,
    clip_epsilon: float = 0.2,
) -> StreamingTrajectoryLoss:
    """Compute one trajectory's token-mean clipped policy loss.

    The caller divides ``loss`` by the number of trajectories before backward
    so sequential gradient accumulation is exactly the equal-trajectory mean
    used by ``clipped_trajectory_policy_loss``. This permits one trajectory's
    per-turn forward graphs to be released before the next trajectory.
    """
    if current_logprobs.ndim != 1 or old_logprobs.ndim != 1:
        raise ValueError("trajectory logprobs must be rank 1")
    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError("current and old trajectory logprobs must have equal shape")
    if current_logprobs.numel() == 0:
        raise ValueError("trajectory must contain at least one action token")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if not torch.isfinite(current_logprobs).all():
        raise ValueError("current_logprobs contains NaN or Inf")
    if not torch.isfinite(old_logprobs).all():
        raise ValueError("old_logprobs contains NaN or Inf")

    advantage_tensor = torch.as_tensor(
        advantage,
        dtype=current_logprobs.dtype,
        device=current_logprobs.device,
    )
    if advantage_tensor.ndim != 0 or not torch.isfinite(advantage_tensor):
        raise ValueError("advantage must be one finite scalar")

    log_ratio = current_logprobs - old_logprobs.detach()
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    objective = torch.minimum(
        ratio * advantage_tensor.detach(),
        clipped_ratio * advantage_tensor.detach(),
    )
    loss = -objective.mean()
    clipped = (ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)
    clip_fraction = clipped.float().mean()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    mean_ratio = ratio.mean()

    for name, value in (
        ("loss", loss),
        ("clip_fraction", clip_fraction),
        ("approximate_kl", approximate_kl),
        ("mean_ratio", mean_ratio),
    ):
        if not torch.isfinite(value):
            raise FloatingPointError(f"{name} is NaN or Inf")

    return StreamingTrajectoryLoss(
        loss=loss,
        clip_fraction=clip_fraction,
        approximate_kl=approximate_kl,
        mean_ratio=mean_ratio,
        token_count=int(current_logprobs.numel()),
    )
