"""Memory-bounded, method-neutral M4 online policy update primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .batch import ReplayGroup, TurnReplay
from .streaming import clipped_single_trajectory_loss, sequence_single_trajectory_loss

TurnLogprobFn = Callable[[TurnReplay], torch.Tensor]


@dataclass
class M4GroupLoss:
    loss: torch.Tensor
    action_token_count: int
    trajectory_count: int
    clip_fraction: float
    approximate_kl: float
    mean_ratio: float


def m4_group_policy_loss(
    replay_group: ReplayGroup,
    algorithm_id: str,
    turn_logprobs: TurnLogprobFn,
    *,
    clip_epsilon: float = 0.2,
) -> M4GroupLoss:
    """Build one equal-trajectory M4 loss from real per-turn forwards.

    RLOO and GRPO use tokenwise ratios but normalize each trajectory before the
    group mean.  GSPO concatenates only the action-token scores from the
    trajectory's existing turn forwards and uses one sequence ratio.  No
    prompt token, padding token, or fabricated cross-turn completion enters
    either calculation.
    """
    # `advantages_for` validates the declared algorithm and selects its
    # registered estimator from the terminal rewards in this strict group.
    advantages = replay_group.advantages_for(algorithm_id)
    trajectories = replay_group.trajectories
    if not trajectories:
        raise ValueError("replay group contains no trajectories")

    total_loss: torch.Tensor | None = None
    weighted_clip = weighted_kl = weighted_ratio = 0.0
    total_tokens = 0
    for trajectory, advantage in zip(trajectories, advantages):
        trajectory_tokens = trajectory.action_token_count
        if trajectory_tokens <= 0:
            raise ValueError("trajectory contains no action token")
        if algorithm_id == "gspo":
            currents: list[torch.Tensor] = []
            olds: list[torch.Tensor] = []
            for turn in trajectory.turns:
                current = turn_logprobs(turn)
                old = torch.tensor(
                    turn.old_policy_logprobs,
                    dtype=current.dtype,
                    device=current.device,
                )
                if current.ndim != 1 or current.shape != old.shape:
                    raise ValueError("turn logprob callback returned an incompatible tensor")
                currents.append(current)
                olds.append(old)
            segment = sequence_single_trajectory_loss(
                torch.cat(currents), torch.cat(olds), advantage, clip_epsilon=clip_epsilon
            )
            trajectory_loss = segment.loss
            weight = 1.0 / len(trajectories)
            weighted_clip += float(segment.clip_fraction.detach().cpu()) / len(trajectories)
            weighted_kl += float(segment.approximate_kl.detach().cpu()) / len(trajectories)
            weighted_ratio += float(segment.mean_ratio.detach().cpu()) / len(trajectories)
        else:
            trajectory_loss: torch.Tensor | None = None
            for turn in trajectory.turns:
                current = turn_logprobs(turn)
                old = torch.tensor(
                    turn.old_policy_logprobs,
                    dtype=current.dtype,
                    device=current.device,
                )
                if current.ndim != 1 or current.shape != old.shape:
                    raise ValueError("turn logprob callback returned an incompatible tensor")
                segment = clipped_single_trajectory_loss(
                    current, old, advantage, clip_epsilon=clip_epsilon
                )
                turn_weight = segment.token_count / trajectory_tokens
                trajectory_loss = (
                    segment.loss * turn_weight
                    if trajectory_loss is None
                    else trajectory_loss + segment.loss * turn_weight
                )
                weighted_clip += float(segment.clip_fraction.detach().cpu()) * turn_weight / len(trajectories)
                weighted_kl += float(segment.approximate_kl.detach().cpu()) * turn_weight / len(trajectories)
                weighted_ratio += float(segment.mean_ratio.detach().cpu()) * turn_weight / len(trajectories)
            if trajectory_loss is None:
                raise ValueError("trajectory contains no turn evidence")
            weight = 1.0 / len(trajectories)
        total_loss = trajectory_loss * weight if total_loss is None else total_loss + trajectory_loss * weight
        total_tokens += trajectory_tokens
    if total_loss is None or not torch.isfinite(total_loss):
        raise FloatingPointError("M4 group policy loss is NaN or Inf")
    return M4GroupLoss(
        loss=total_loss,
        action_token_count=total_tokens,
        trajectory_count=len(trajectories),
        clip_fraction=weighted_clip,
        approximate_kl=weighted_kl,
        mean_ratio=weighted_ratio,
    )


def apply_m4_online_update(
    replay_groups: Iterable[ReplayGroup],
    algorithm_id: str,
    parameters: Iterable[torch.nn.Parameter],
    turn_logprobs: TurnLogprobFn,
    *,
    learning_rate: float,
    clip_epsilon: float = 0.2,
    gradient_clip: float = 1.0,
) -> dict[str, float | int | str]:
    """Apply one optimizer step over an already-audited collection batch."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    parameters = list(parameters)
    if not parameters:
        raise ValueError("M4 online update requires at least one trainable parameter")
    groups = list(replay_groups)
    if not groups:
        raise ValueError("M4 online update requires at least one selected group")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in parameters]
    metrics: list[M4GroupLoss] = []
    for group in groups:
        result = m4_group_policy_loss(
            group, algorithm_id, turn_logprobs, clip_epsilon=clip_epsilon
        )
        (result.loss / len(groups)).backward()
        metrics.append(result)
    nonzero_gradients = sum(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in parameters
    )
    if nonzero_gradients == 0:
        raise RuntimeError("M4 online update has no non-zero gradient")
    pre_clip_norm = torch.sqrt(sum(
        parameter.grad.detach().float().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ))
    clipped_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    changed = sum(
        not torch.equal(parameter.detach(), before_value)
        for parameter, before_value in zip(parameters, before)
    )
    if changed == 0:
        raise RuntimeError("M4 optimizer step did not change any trainable parameter")
    return {
        "algorithm_id": algorithm_id,
        "optimizer_updates": 1,
        "selected_groups": len(groups),
        "action_token_count": sum(result.action_token_count for result in metrics),
        "mean_policy_loss": sum(float(result.loss.detach().cpu()) for result in metrics) / len(metrics),
        "mean_clip_fraction": sum(result.clip_fraction for result in metrics) / len(metrics),
        "mean_approximate_kl": sum(result.approximate_kl for result in metrics) / len(metrics),
        "mean_ratio": sum(result.mean_ratio for result in metrics) / len(metrics),
        "nonzero_gradient_parameter_tensors": nonzero_gradients,
        "changed_parameter_tensors": changed,
        "pre_clip_gradient_norm": float(pre_clip_norm.detach().cpu()),
        "clip_grad_norm_return": float(clipped_norm.detach().cpu()),
    }
