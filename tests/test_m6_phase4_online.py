from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from miniwebwork.webshop_rl.phase4_online import standardized_binary_advantages, trajectory_policy_turn_weights


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_phase4_binary_advantages_are_standardized_and_homogeneous_safe():
    assert standardized_binary_advantages([0, 0, 0, 0]) == (0.0, 0.0, 0.0, 0.0)
    assert standardized_binary_advantages([1, 1, 1, 1]) == (0.0, 0.0, 0.0, 0.0)
    values = standardized_binary_advantages([1, 0, 0, 0])
    assert sum(values) == pytest.approx(0.0)
    assert sum(value * value for value in values) / 4 == pytest.approx(1.0)
    assert values[0] > 0 and all(value < 0 for value in values[1:])


def test_phase5_tail2_credit_only_keeps_last_two_turns_and_normalizes_mass():
    full = trajectory_policy_turn_weights([2, 4, 8], policy_credit_window="full")
    tail = trajectory_policy_turn_weights([2, 4, 8], policy_credit_window="tail2")
    assert [weight * length for weight, length in zip(full, [2, 4, 8])] == pytest.approx([1 / 3] * 3)
    assert [weight * length for weight, length in zip(tail, [2, 4, 8])] == pytest.approx([0, 1 / 2, 1 / 2])
    assert sum(weight * length for weight, length in zip(tail, [2, 4, 8])) == pytest.approx(1.0)


def test_phase5_policy_mask_changes_only_policy_objective_weights():
    torch = pytest.importorskip("torch")
    from miniwebwork.webshop_rl.m6_online_training import strict_grpo_kl_loss

    replay = torch.zeros((1, 2), requires_grad=True)
    batch = {
        "behavior_logprobs": torch.zeros((1, 2)),
        "completion_mask": torch.ones((1, 2), dtype=torch.bool),
        "advantages": torch.ones(1),
        "token_loss_weights": torch.full((1,), 0.5),
        "policy_token_loss_weights": torch.ones(1),
    }
    result = strict_grpo_kl_loss(replay, torch.zeros((1, 2)), batch, clip_epsilon=0.2, kl_coefficient=0.03)
    assert float(result["policy_loss"].detach()) == pytest.approx(-2.0)
    assert float(result["reference_kl"].detach()) == pytest.approx(0.0)


def test_phase5_probe_is_zero_update_and_two_panel():
    source = (SCRIPTS / "run_m6_phase5_tail_credit_probe_job.sh").read_text(encoding="utf-8")
    probe = (ROOT / "src" / "miniwebwork" / "webshop_rl" / "phase5_tail_credit.py").read_text(encoding="utf-8")
    assert source.count("--panel-collection-root") == 2
    assert "step_00/collection" in source and "step_01/collection" in source
    assert "optimizer.step(" not in probe
    assert '"optimizer_steps": 0' in probe


def test_phase4_online_roster_selection_is_deterministic_and_stratified():
    module = importlib.import_module("m6_phase4_build_online_roster")
    goals = {
        f"webshop_goal_{index:05d}": {
            "category": ("fashion", "grocery", "electronics", "garden")[index % 4],
            "attributes": [str(slot) for slot in range(index % 5)],
            "goal_options": ["blue"] if index % 3 == 0 else [],
            "price_upper": 50 if index % 2 == 0 else None,
        }
        for index in range(1000, 1080)
    }
    candidates = list(goals)
    first = module._select(candidates, goals)
    second = module._select(candidates, goals)
    assert first == second
    assert len(first) == len(set(first)) == 40


def test_phase4_online_collector_contract_is_k4x4_full_horizon():
    pytest.importorskip("playwright")
    module = importlib.import_module("m6_collect_policy_success")
    args = Namespace(
        role="train",
        task_roster=Path("roster.json"),
        adapter=Path("adapter"),
        k=4,
        max_model_turns=18,
        max_environment_steps=15,
        maximum_tasks=4,
        task_offset=8,
        maximum_action_tokens=75000,
        replay_prefix_root=None,
    )
    module.validate_phase4_online_rl_contract(args)
    args.k = 8
    with pytest.raises(ValueError, match="must use K4"):
        module.validate_phase4_online_rl_contract(args)


def test_phase4_tuning_collector_contract_is_fresh_k4_full_horizon():
    pytest.importorskip("playwright")
    module = importlib.import_module("m6_collect_policy_success")
    args = Namespace(
        role="train",
        task_roster=Path("tuning.json"),
        adapter=None,
        k=4,
        max_model_turns=18,
        max_environment_steps=15,
        maximum_tasks=None,
        task_offset=0,
        maximum_action_tokens=None,
        replay_prefix_root=None,
    )
    module.validate_phase4_tuning_evaluation_contract(args)
    args.max_environment_steps = 6
    with pytest.raises(ValueError, match="full horizon"):
        module.validate_phase4_tuning_evaluation_contract(args)


def test_phase4_online_job_freezes_verified_configuration():
    source = (SCRIPTS / "run_m6_phase4_online_rl_job.sh").read_text(encoding="utf-8")
    learner = (ROOT / "src" / "miniwebwork" / "webshop_rl" / "phase4_online.py").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=24G" in source
    assert "--mode phase4_online_rl_collection --role train --k 4" in source
    assert "--max-model-turns 18 --max-environment-steps 15" in source
    assert 'steps="${M6_ONLINE_STEPS:-10}"' in source
    assert "learning_rate: float = 3e-6" in learner
    assert "module.eval()" in learner
    assert "optimizer.step()" in learner
    assert "kl_hard_stop: float = 0.01" in learner


def test_phase4_tuning_eval_is_paired_and_has_no_training():
    source = (SCRIPTS / "run_m6_phase4_tuning_eval_job.sh").read_text(encoding="utf-8")
    stats = (SCRIPTS / "m6_phase4_tuning_stats.py").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1%1" in source
    assert "--mode phase4_tuning_evaluation --role train --k 4" in source
    assert "--max-model-turns 18 --max-environment-steps 15" in source
    assert "20260829" in source and "20260830" in source
    assert "optimizer.step(" not in stats
    assert '"raw_sft_rl_chain_passed"' in stats
    assert '"rl_minus_sft_at_least_1pp"' in stats
