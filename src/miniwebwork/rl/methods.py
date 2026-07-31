"""M4 algorithm registry with one auditable policy-loss interface."""

from __future__ import annotations

import torch

from ..m4_algorithms import AlgorithmSpec, get_algorithm_spec
from .objective import (
    PolicyLossResult,
    clipped_trajectory_policy_loss,
    group_relative_advantages,
    leave_one_out_advantages,
    sequence_clipped_trajectory_policy_loss,
)

def online_advantages(algorithm_id: str, rewards: torch.Tensor) -> torch.Tensor:
    """Build method-specific advantages from a strict same-task rollout group."""
    spec = get_algorithm_spec(algorithm_id)
    if spec.regime != "online":
        raise ValueError(f"{algorithm_id} is offline and has no rollout advantage")
    if algorithm_id == "rloo":
        return leave_one_out_advantages(rewards)
    return group_relative_advantages(rewards)


def online_policy_loss(
    algorithm_id: str,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    reference_logprobs: torch.Tensor | None = None,
    kl_beta: float = 0.0,
) -> PolicyLossResult:
    """Apply the registered M4 online method to action-token evidence only."""
    spec = get_algorithm_spec(algorithm_id)
    if spec.regime != "online":
        raise ValueError(f"{algorithm_id} is offline and has no policy-ratio loss")
    loss_fn = (
        sequence_clipped_trajectory_policy_loss
        if algorithm_id == "gspo"
        else clipped_trajectory_policy_loss
    )
    return loss_fn(
        current_logprobs,
        old_logprobs,
        advantages,
        token_mask,
        clip_epsilon=clip_epsilon,
        reference_logprobs=reference_logprobs,
        kl_beta=kl_beta,
    )
