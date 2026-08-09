import copy
from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.formal_contract import (
    EXPECTED_RUNTIME_SHA256,
    EXPECTED_STUDY_SHA256,
    load_formal_authorization,
    formal_output_path,
    validate_formal_authorization,
)


def test_formal_authorization_binds_immutable_preflight_and_narrow_matrix():
    loaded = load_formal_authorization()
    payload = loaded["payload"]
    immutable = payload["immutable_preflight_contracts"]
    assert immutable["study_manifest_sha256"] == EXPECTED_STUDY_SHA256
    assert immutable["runtime_manifest_sha256"] == EXPECTED_RUNTIME_SHA256
    assert payload["formal_matrix"]["online_methods"] == ["multi_turn_grpo", "step_aware_gpo"]
    assert payload["formal_matrix"]["online_seeds"] == [20260801, 20260802, 20260803]
    assert payload["formal_matrix"]["total_model_count"] == 7


def test_formal_gate_resolution_discloses_but_does_not_harden_diagnostics():
    payload = load_formal_authorization()["payload"]
    diagnostics = payload["gate_resolution"]["diagnostics"]
    assert diagnostics["generation_gpu_utilization_p50"]["hard_gate"] is False
    assert diagnostics["generation_gpu_utilization_p50"]["selected_e2e_observed"] == 0.11
    assert diagnostics["optimizer_action_token_fraction"]["hard_gate"] is False
    assert diagnostics["optimizer_action_token_fraction"]["selected_e2e_observed"] == 0.137643


def test_formal_authorization_fails_closed_on_resource_or_method_drift():
    payload = copy.deepcopy(load_formal_authorization()["payload"])
    payload["resource_contract"]["online"]["cpus"] = 32
    with pytest.raises(ValueError, match="resource"):
        validate_formal_authorization(payload)
    payload = copy.deepcopy(load_formal_authorization()["payload"])
    payload["formal_matrix"]["online_methods"].append("rloo")
    with pytest.raises(ValueError, match="methods"):
        validate_formal_authorization(payload)


def test_formal_output_path_rejects_preflight_or_parent_escape(tmp_path):
    accepted = formal_output_path(Path("outputs/m4_long_horizon_credit_v1/formal/shared_sft/x"))
    assert "formal/shared_sft/x" in str(accepted)
    with pytest.raises(ValueError, match="not a formal output"):
        formal_output_path(Path("outputs/m4_long_horizon_credit_v1/preflight/x"))
    with pytest.raises(ValueError, match="not a formal output"):
        formal_output_path(tmp_path / "elsewhere")
