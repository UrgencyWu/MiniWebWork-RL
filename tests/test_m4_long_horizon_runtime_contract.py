import copy

import pytest

from miniwebwork.long_horizon_rl.runtime_contract import (
    PARITY_THRESHOLDS,
    load_online_runtime_contract,
    validate_online_runtime_contract,
)


def test_online_runtime_contract_is_focused_same_gpu_and_preflight_only():
    loaded = load_online_runtime_contract()
    payload = loaded["payload"]
    assert payload["formal_submission_allowed"] is False
    assert payload["generation_contract"]["backend"] == "vllm_async"
    assert payload["generation_contract"]["generation_during_learner"] is False
    assert payload["rollout_contract"]["browser_worker_candidates"] == [1, 2, 4, 8]
    assert payload["rollout_contract"]["maximum_group_token_reserve"] == 10240
    assert payload["learner_contract"]["behavior_policy_staleness"] == 0
    assert {
        key: payload["parity_contract"][key]
        for key in PARITY_THRESHOLDS
    } == PARITY_THRESHOLDS
    assert payload["parity_contract"]["thresholds_frozen_before_gpu_observation"] is True
    assert payload["slurm_contract"] == {
        "gpus": 1,
        "cpus": 8,
        "memory_gb": 48,
        "wall_time": "24:00:00",
        "maximum_concurrent_gpu_jobs": 4,
        "real_resume_smoke_required": True,
    }


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("generation_contract", "temperature", 0.7, "sampling"),
        ("generation_contract", "generation_during_learner", True, "stale"),
        ("rollout_contract", "group_size", 8, "K drift"),
        ("rollout_contract", "action_token_budget_per_method_seed", 200000, "budget"),
        ("rollout_contract", "maximum_concurrent_k4_groups", 3, "concurrent"),
        ("learner_contract", "behavior_policy_staleness", 1, "staleness"),
        ("parity_contract", "replay_maximum_absolute_difference", 0.5, "parity"),
        ("evidence_contract", "turn_charge_before_full_artifact", False, "durability"),
        ("recovery_contract", "identity_mismatch", "warn", "identity"),
    ],
)
def test_online_runtime_contract_fails_closed_on_research_relevant_drift(
    section, field, value, error
):
    payload = copy.deepcopy(load_online_runtime_contract()["payload"])
    payload[section][field] = value
    with pytest.raises(ValueError, match=error):
        validate_online_runtime_contract(payload)
