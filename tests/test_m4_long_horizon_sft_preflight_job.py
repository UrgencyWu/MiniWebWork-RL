from pathlib import Path


def test_focused_sft_preflight_job_is_gpu_bounded_and_nonformal():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_m4_long_horizon_sft_preflight_job.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "m4_assert_frozen_git.sh" in script
    assert "--mode preflight" in script
    assert "formal_training=false" in script
    assert "--loop=5" in script
    assert "outputs/m4_long_horizon_credit_v1/preflight" in script
    assert "outputs/m4_long_horizon_credit_v1/formal" not in script


def test_focused_sft_entrypoint_exposes_no_formal_mode():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "m4_long_horizon_sft.py").read_text()
    assert 'choices=("preflight",)' in script
    assert "SFT_PREFLIGHT_MAXIMUM_OPTIMIZER_UPDATES" in script
    assert "assert_formal_submission_closed" in script
    assert "assert_preflight_output" in script
