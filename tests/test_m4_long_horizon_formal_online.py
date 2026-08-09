from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.formal_online import (
    MAXIMUM_GROUP_TOKEN_RESERVE,
    expected_online_root,
)
from miniwebwork.long_horizon_rl.rollout import admit_next_group


def test_formal_online_roots_are_method_seed_isolated():
    roots = {
        expected_online_root(method, seed)
        for method in ("multi_turn_grpo", "step_aware_gpo")
        for seed in (20260801, 20260802, 20260803)
    }
    assert len(roots) == 6
    assert all("formal/online" in str(root) for root in roots)
    with pytest.raises(ValueError):
        expected_online_root("rloo", 20260801)


def test_formal_budget_closes_only_inside_worst_case_full_k4_reserve():
    assert MAXIMUM_GROUP_TOKEN_RESERVE == 10240
    assert admit_next_group(global_tokens_before_iteration=239760, current_iteration_tokens=0).allowed
    assert not admit_next_group(global_tokens_before_iteration=239761, current_iteration_tokens=0).allowed


def test_formal_online_slurm_is_24h_and_resource_bounded():
    script = Path("scripts/run_m4_long_horizon_formal_online_job.sh").read_text(encoding="utf-8")
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=48G" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "formal_training=true" in script
    assert "readiness_manifest_v2.json" in script
    assert "M4_METHOD" in script and "M4_SEED" in script
