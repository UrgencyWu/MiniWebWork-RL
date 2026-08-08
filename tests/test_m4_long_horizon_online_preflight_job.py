from pathlib import Path


def test_online_preflight_job_uses_one_modest_24h_allocation_and_stable_resume_root():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_m4_long_horizon_online_preflight_job.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=48G" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "m4_assert_frozen_git.sh" in script
    assert "M4_PREFLIGHT_RUN_NAME" in script
    assert "online_${run_name}" in script
    assert "--loop=5" in script
    assert "formal_training=false" in script
    assert "outputs/m4_long_horizon_credit_v1/preflight" in script
    assert "outputs/m4_long_horizon_credit_v1/formal" not in script
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in script
    assert "unset PYTORCH_ALLOC_CONF" in script
    assert "export PYTORCH_CUDA_ALLOC_CONF" not in script


def test_online_preflight_python_entrypoint_exposes_no_formal_mode():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "m4_long_horizon_online_preflight.py").read_text()
    assert "--collection-only" in script
    assert "--browser-workers" in script
    assert "choices=ALLOWED_BROWSER_WORKERS" in script
    assert "--formal" not in script
