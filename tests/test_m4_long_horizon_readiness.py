from __future__ import annotations

import pytest

from miniwebwork.long_horizon_rl.readiness import (
    JobEvidence,
    _percentile,
    parse_sacct_record,
)


def test_percentile_uses_linear_interpolation():
    assert _percentile([0, 10, 20, 30], 0.5) == 15
    assert _percentile([0, 100], 0.95) == 95


def test_percentile_fails_closed_on_empty_samples():
    with pytest.raises(ValueError, match="empty"):
        _percentile([], 0.5)


def test_parse_sacct_record_selects_top_level_job_and_accepts_cancel_owner():
    job = JobEvidence(7, "recovery", "CANCELLED", "0:0", "out", "err")
    text = "7|CANCELLED by 1003|0:0|00:02:37|8|48G\n7.batch|CANCELLED|0:15|00:02:39|8|\n"
    record = parse_sacct_record(text, job)
    assert record["job_id"] == 7
    assert record["state"] == "CANCELLED by 1003"
    assert record["allocated_cpus"] == 8


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("expected_state", "COMPLETED", "state drift"),
        ("expected_exit_code", "1:0", "exit drift"),
    ],
)
def test_parse_sacct_record_fails_closed_on_state_or_exit_drift(field, value, error):
    values = {
        "job_id": 7,
        "purpose": "recovery",
        "expected_state": "CANCELLED",
        "expected_exit_code": "0:0",
        "stdout_path": "out",
        "stderr_path": "err",
    }
    values[field] = value
    with pytest.raises(ValueError, match=error):
        parse_sacct_record("7|CANCELLED by 1003|0:0|00:02:37|8|48G\n", JobEvidence(**values))


def test_parse_sacct_record_fails_closed_when_top_level_record_missing():
    job = JobEvidence(7, "recovery", "CANCELLED", "0:0", "out", "err")
    with pytest.raises(ValueError, match="missing sacct"):
        parse_sacct_record("7.batch|CANCELLED|0:15|00:02:39|8|\n", job)
