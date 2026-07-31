import pytest
import torch

from miniwebwork.m4_algorithms import get_algorithm_spec
from miniwebwork.rl.methods import (
    online_advantages,
    online_policy_loss,
)


def test_m4_algorithm_registry_is_explicit_and_rejects_aliases():
    assert get_algorithm_spec("sft").regime == "offline"
    assert get_algorithm_spec("rsft").advantage_estimator == "verified_best_of_n"
    assert get_algorithm_spec("rloo").advantage_estimator == "leave_one_out"
    assert get_algorithm_spec("grpo").likelihood_ratio == "per_action_token"
    assert get_algorithm_spec("gspo").likelihood_ratio == "per_trajectory_sequence"
    with pytest.raises(ValueError, match="Unsupported M4 algorithm"):
        get_algorithm_spec("ppo")


def test_online_interface_routes_rloo_grpo_and_gspo_by_contract():
    rewards = torch.tensor([0.0, 1.0])
    assert torch.allclose(online_advantages("rloo", rewards), torch.tensor([-1.0, 1.0]))
    assert torch.allclose(online_advantages("grpo", rewards), torch.tensor([-1.0, 1.0]))
    with pytest.raises(ValueError, match="offline"):
        online_advantages("sft", rewards)

    old = torch.zeros((2, 1))
    current = old.clone().requires_grad_(True)
    mask = torch.ones((2, 1), dtype=torch.bool)
    result = online_policy_loss("gspo", current, old, rewards * 2 - 1, mask)
    assert result.trajectory_count == 2
    with pytest.raises(ValueError, match="offline"):
        online_policy_loss("rsft", current, old, rewards * 2 - 1, mask)
