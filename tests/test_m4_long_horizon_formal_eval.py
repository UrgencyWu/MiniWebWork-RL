from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.formal_eval import expected_eval_root


def test_formal_eval_matrix_contains_exactly_seven_model_roots():
    roots = {expected_eval_root("verified_sft", 20260801)}
    roots.update(
        expected_eval_root(method, seed)
        for method in ("multi_turn_grpo", "step_aware_gpo")
        for seed in (20260801, 20260802, 20260803)
    )
    assert len(roots) == 7
    with pytest.raises(ValueError):
        expected_eval_root("verified_sft", 20260802)


def test_formal_eval_slurm_freezes_k4_resources_and_dependency_gate():
    script = Path("scripts/run_m4_long_horizon_formal_eval_job.sh").read_text(encoding="utf-8")
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=48G" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "nvidia-smi -i \"$CUDA_VISIBLE_DEVICES\"" in script
    assert "telemetry_samples=" in script
    source = Path("src/miniwebwork/long_horizon_rl/formal_eval.py").read_text(encoding="utf-8")
    assert "for method in ONLINE_METHODS" in source
    assert "for seed in ONLINE_SEEDS" in source
    assert "expected_trajectory_count\": 480" in source
    assert "training_updates_allowed\": False" in source
