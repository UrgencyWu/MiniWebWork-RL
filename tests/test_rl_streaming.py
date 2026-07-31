import pytest
import torch

from miniwebwork.rl.objective import clipped_trajectory_policy_loss
from miniwebwork.rl.streaming import clipped_single_trajectory_loss


def test_streaming_losses_equal_batched_equal_trajectory_objective():
    current = torch.tensor(
        [
            [-0.10, -0.30, 0.0],
            [-0.20, -0.40, -0.60],
        ],
        dtype=torch.float32,
    )
    old = torch.tensor(
        [
            [-0.12, -0.25, 0.0],
            [-0.22, -0.35, -0.55],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    advantages = torch.tensor([-1.0, 1.0])

    batched = clipped_trajectory_policy_loss(
        current,
        old,
        advantages,
        mask,
        clip_epsilon=0.2,
    )
    first = clipped_single_trajectory_loss(
        current[0, mask[0]],
        old[0, mask[0]],
        advantages[0],
        clip_epsilon=0.2,
    )
    second = clipped_single_trajectory_loss(
        current[1, mask[1]],
        old[1, mask[1]],
        advantages[1],
        clip_epsilon=0.2,
    )

    streaming_mean = (first.loss + second.loss) / 2
    assert torch.allclose(streaming_mean, batched.policy_loss, atol=1e-7)


def test_streaming_loss_supports_gradient_accumulation():
    current_a = torch.tensor([-0.1, -0.2], requires_grad=True)
    current_b = torch.tensor([-0.3], requires_grad=True)

    loss_a = clipped_single_trajectory_loss(
        current_a,
        torch.tensor([-0.1, -0.2]),
        -1.0,
    ).loss
    loss_b = clipped_single_trajectory_loss(
        current_b,
        torch.tensor([-0.3]),
        1.0,
    ).loss
    (loss_a / 2).backward()
    (loss_b / 2).backward()

    assert current_a.grad is not None
    assert current_b.grad is not None
    assert torch.isfinite(current_a.grad).all()
    assert torch.isfinite(current_b.grad).all()


def test_streaming_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="equal shape"):
        clipped_single_trajectory_loss(
            torch.tensor([-0.1]),
            torch.tensor([-0.1, -0.2]),
            1.0,
        )


def test_streaming_loss_rejects_nonfinite_advantage():
    with pytest.raises(ValueError, match="finite scalar"):
        clipped_single_trajectory_loss(
            torch.tensor([-0.1]),
            torch.tensor([-0.1]),
            float("nan"),
        )
