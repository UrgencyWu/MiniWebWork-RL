import json
from pathlib import Path

from miniwebwork.long_horizon_rl.contracts import atomic_write_json
from miniwebwork.long_horizon_rl.formal_sft import prepare_sft_attempt


def test_formal_sft_recovery_archives_incomplete_attempt_without_deletion(tmp_path: Path):
    first, reused = prepare_sft_attempt(tmp_path)
    assert first.name == "attempt-0001"
    assert reused is False
    (first / "partial.bin").write_bytes(b"partial")
    second, reused = prepare_sft_attempt(tmp_path)
    assert second.name == "attempt-0002"
    assert reused is False
    assert (tmp_path / "invalidated_attempts" / "attempt-0001" / "partial.bin").read_bytes() == b"partial"


def test_formal_sft_recovery_reuses_completed_training_for_finalization(tmp_path: Path):
    attempt, _ = prepare_sft_attempt(tmp_path)
    training = attempt / "training"
    training.mkdir()
    atomic_write_json(training / "training_report.json", {"complete": True})
    same, reused = prepare_sft_attempt(tmp_path)
    assert same == attempt
    assert reused is True
    assert not (tmp_path / "invalidated_attempts").exists()


def test_formal_sft_slurm_resources_and_fixed_root():
    script = Path("scripts/run_m4_long_horizon_formal_sft_job.sh").read_text(encoding="utf-8")
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "formal_training=true" in script
    assert "formal/shared_sft/seed_20260801" in script
    assert "readiness_manifest_v2.json" in script


def test_formal_sft_publishes_completion_manifest_after_attempt_commit():
    source = Path("src/miniwebwork/long_horizon_rl/formal_sft.py").read_text(encoding="utf-8")
    attempt_commit = source.index('attempt_record["complete"] = True')
    final_publish = source.index('atomic_write_json(root / FORMAL_SFT_MANIFEST_NAME, manifest)')
    assert attempt_commit < final_publish
