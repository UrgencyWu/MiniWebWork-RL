import pytest
import torch

from miniwebwork.rl.objective import (
    NoRewardVarianceError,
    clipped_trajectory_policy_loss,
    group_relative_advantages,
)


def test_group_relative_advantages_for_binary_rewards():
    advantages = group_relative_advantages(torch.tensor([0.0, 1.0]))

    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0]), atol=1e-6)
    assert abs(float(advantages.mean())) < 1e-6


def test_group_relative_advantages_reject_zero_variance():
    with pytest.raises(NoRewardVarianceError, match="zero reward variance"):
        group_relative_advantages(torch.tensor([0.0, 0.0, 0.0]))


def test_on_policy_balanced_group_has_zero_initial_policy_loss():
    old = torch.tensor(
        [
            [-0.2, -0.4, 0.0],
            [-0.1, -0.3, -0.5],
        ],
        requires_grad=False,
    )
    current = old.clone().requires_grad_(True)
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    advantages = torch.tensor([-1.0, 1.0])

    result = clipped_trajectory_policy_loss(
        current,
        old,
        advantages,
        mask,
    )

    assert abs(float(result.policy_loss.detach())) < 1e-6
    assert torch.allclose(result.mean_ratio.detach(), torch.tensor(1.0))
    assert torch.allclose(result.clip_fraction.detach(), torch.tensor(0.0))
    result.loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_positive_advantage_ratio_is_clipped():
    old = torch.zeros((2, 2))
    current = torch.tensor(
        [
            [torch.log(torch.tensor(2.0)), torch.log(torch.tensor(2.0))],
            [0.0, 0.0],
        ],
        requires_grad=True,
    )
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.ones((2, 2), dtype=torch.bool)

    result = clipped_trajectory_policy_loss(
        current,
        old,
        advantages,
        mask,
        clip_epsilon=0.2,
    )

    # The first trajectory's positive-advantage ratio is capped at 1.2.
    # The second trajectory contributes -1.0, so the averaged objective is
    # (1.2 - 1.0) / 2 and loss is -0.1.
    assert torch.allclose(result.policy_loss.detach(), torch.tensor(-0.1), atol=1e-6)
    assert result.clip_fraction > 0


def test_trajectory_normalization_prevents_long_trace_domination():
    old = torch.zeros((2, 4))
    current = old.clone().requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, True],
        ]
    )

    result = clipped_trajectory_policy_loss(current, old, advantages, mask)

    # Equal trajectory weighting gives zero despite unequal token counts.
    assert abs(float(result.policy_loss.detach())) < 1e-6
    assert result.valid_token_count == 5
    assert result.trajectory_count == 2


def test_reference_kl_is_nonnegative():
    current = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], requires_grad=True)
    old = current.detach().clone()
    reference = torch.tensor([[-0.2, -0.3], [-0.5, -0.2]])
    advantages = torch.tensor([-1.0, 1.0])
    mask = torch.ones((2, 2), dtype=torch.bool)

    result = clipped_trajectory_policy_loss(
        current,
        old,
        advantages,
        mask,
        reference_logprobs=reference,
        kl_beta=0.05,
    )

    assert result.reference_kl >= 0
    assert result.loss >= result.policy_loss


def test_objective_rejects_trajectory_without_action_tokens():
    current = torch.zeros((2, 2))
    old = torch.zeros((2, 2))
    advantages = torch.tensor([-1.0, 1.0])
    mask = torch.tensor([[True, False], [False, False]])

    with pytest.raises(ValueError, match="every trajectory"):
        clipped_trajectory_policy_loss(current, old, advantages, mask)


def test_objective_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="old_logprobs shape"):
        clipped_trajectory_policy_loss(
            torch.zeros((2, 2)),
            torch.zeros((2, 3)),
            torch.tensor([-1.0, 1.0]),
            torch.ones((2, 2), dtype=torch.bool),
        )
