import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from miniwebwork.long_horizon_rl.formal_invocation import (
    _summarize_gpu_telemetry,
    collect_slurm_accounting,
    finish_formal_invocation,
    start_formal_invocation,
    validate_formal_invocation,
)


def test_formal_invocation_is_slurm_bound_and_self_hashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    path = start_formal_invocation(
        root=tmp_path, phase="online", method="multi_turn_grpo", seed=20260801,
        git_sha="a" * 40, expected_cpus=8, expected_gpus=1,
        log_stem="m4_lh_formal_online",
        telemetry_path="logs/m4_lh_formal_online_multi_turn_grpo_20260801_9001_gpu.csv",
    )
    running = validate_formal_invocation(json.loads(path.read_text(encoding="utf-8")))
    assert running["runner_status"] == "RUNNING"
    complete = finish_formal_invocation(path, status="WORKLOAD_COMPLETE")
    assert complete["runner_status"] == "WORKLOAD_COMPLETE"


def test_formal_invocation_rejects_resource_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLURM_JOB_ID", "9002")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="visible-GPU"):
        start_formal_invocation(
            root=tmp_path, phase="sft", method="verified_sft", seed=20260801,
            git_sha="b" * 40, expected_cpus=4, expected_gpus=1,
            log_stem="m4_lh_formal_sft",
            telemetry_path="logs/m4_lh_formal_sft_9002_gpu.csv",
        )


def test_formal_gpu_telemetry_is_single_device_and_summarized(tmp_path: Path):
    path = tmp_path / "gpu.csv"
    path.write_text(
        "2026/08/09 12:00:00.000, 0, GPU-test, 20, 10, 100, 1000, 50\n"
        "2026/08/09 12:00:05.000, 0, GPU-test, 80, 20, 400, 1000, 150\n",
        encoding="utf-8",
    )
    summary = _summarize_gpu_telemetry(path)
    assert summary["sample_count"] == 2
    assert summary["gpu_utilization_fraction_mean"] == 0.5
    assert summary["minimum_vram_headroom_fraction"] == 0.6


def test_formal_accounting_requires_and_reports_completed_gpu_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "9003")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    root = tmp_path / "artifact"
    invocation = start_formal_invocation(
        root=root, phase="online", method="step_aware_gpo", seed=20260801,
        git_sha="c" * 40, expected_cpus=8, expected_gpus=1,
        log_stem="m4_lh_formal_online",
        telemetry_path="logs/m4_lh_formal_online_step_aware_gpo_20260801_9003_gpu.csv",
    )
    finish_formal_invocation(invocation, status="WORKLOAD_COMPLETE")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "m4_lh_formal_online_9003.out").write_text("complete\n", encoding="utf-8")
    (logs / "m4_lh_formal_online_9003.err").write_text("", encoding="utf-8")
    (logs / "m4_lh_formal_online_step_aware_gpo_20260801_9003_gpu.csv").write_text(
        "2026/08/09 12:00:00.000, 0, GPU-test, 90, 20, 400, 1000, 150\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "miniwebwork.long_horizon_rl.formal_invocation.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="9003|COMPLETED|0:0|60|8|48G|billing=8,cpu=8,gres/gpu=1,mem=48G,node=1\n"
        ),
    )
    result = collect_slurm_accounting([root], repo_root=tmp_path)
    assert result["job_count"] == 1
    assert result["jobs"][0]["gpu_telemetry"]["gpu_utilization_fraction_p50"] == 0.9
