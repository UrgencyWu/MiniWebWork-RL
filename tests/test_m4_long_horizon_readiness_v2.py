from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.formal_contract import load_formal_authorization
from miniwebwork.long_horizon_rl.readiness_v2 import formal_suite_evidence


def test_readiness_v2_validates_real_formal_inventory_and_resources():
    authorization = load_formal_authorization()["payload"]
    evidence = formal_suite_evidence(Path.cwd(), authorization)
    assert evidence["passed"] is True
    assert evidence["slurm_resources"]["shared_sft_slurm"]["cpus-per-task"] == "4"
    assert evidence["slurm_resources"]["online_slurm"]["cpus-per-task"] == "8"
    assert evidence["slurm_resources"]["evaluation_slurm"]["cpus-per-task"] == "8"
    assert evidence["slurm_resources"]["analysis_slurm"] == {"time": "24:00:00", "cpus-per-task": "2", "mem": "8G"}
    assert evidence["slurm_resources"]["clean_regression_slurm"]["cpus-per-task"] == "2"
    assert evidence["slurm_resources"]["readiness_slurm"]["cpus-per-task"] == "2"
    for script in (
        Path("scripts/run_m4_long_horizon_formal_sft_job.sh"),
        Path("scripts/run_m4_long_horizon_formal_online_job.sh"),
        Path("scripts/run_m4_long_horizon_formal_eval_job.sh"),
    ):
        source = script.read_text(encoding="utf-8")
        assert 'nvidia-smi -i "$CUDA_VISIBLE_DEVICES"' in source
        assert "--loop=5" in source
        assert "telemetry_samples=" in source


def test_readiness_v2_has_no_hardcoded_ready_gates():
    source = Path("src/miniwebwork/long_horizon_rl/readiness_v2.py").read_text(encoding="utf-8")
    assert '"decision": "READY" if not unmet else "NOT_READY"' in source
    assert '"formal_submission_allowed": authorization' in source
    assert 'generation_gpu_utilization_p50": selected' in source
    assert 'optimizer_action_token_fraction": selected' in source
    assert '"32_lane_saturation_gate": saturation["passed"]' in source


def test_formal_suite_fails_if_entrypoint_inventory_drifts(tmp_path):
    authorization = load_formal_authorization()["payload"]
    with pytest.raises(ValueError, match="missing"):
        formal_suite_evidence(tmp_path, authorization)
