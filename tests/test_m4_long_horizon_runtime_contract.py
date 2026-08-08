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
    assert payload["generation_contract"]["cuda_allocator_environment"] == (
        "pytorch_allocator_aliases_unset_for_vllm_cumem_sleep"
    )
    assert payload["rollout_contract"]["browser_worker_candidates"] == [1, 2, 4, 8]
    assert payload["rollout_contract"]["maximum_group_token_reserve"] == 10240
    assert payload["learner_contract"]["behavior_policy_staleness"] == 0
    assert payload["model_contract"]["base_model_manifest_sha256"] == (
        "290ecd9ec4eaa1f5ac6927b10e9cb4c600d22aec78a6d743baa8a01d72c1b7a3"
    )
    assert payload["model_contract"]["base_model_functional_file_set_sha256"] == (
        "6b2cdb9cf894cec7eb1dcef2a57682a9d73e22f2a81a58ef3b1f32854f031b85"
    )
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
        ("generation_contract", "cuda_allocator_environment", "expandable", "allocator"),
        ("rollout_contract", "group_size", 8, "K drift"),
        ("rollout_contract", "action_token_budget_per_method_seed", 200000, "budget"),
        ("rollout_contract", "maximum_concurrent_k4_groups", 3, "concurrent"),
        ("learner_contract", "behavior_policy_staleness", 1, "staleness"),
        ("parity_contract", "replay_maximum_absolute_difference", 0.5, "parity"),
        ("evidence_contract", "turn_charge_before_full_artifact", False, "durability"),
        ("recovery_contract", "identity_mismatch", "warn", "identity"),
        (
            "recovery_contract",
            "post_update_commit_pre_wake",
            "claim_pass",
            "post-commit phase",
        ),
    ],
)
def test_online_runtime_contract_fails_closed_on_research_relevant_drift(
    section, field, value, error
):
    payload = copy.deepcopy(load_online_runtime_contract()["payload"])
    payload[section][field] = value
    with pytest.raises(ValueError, match=error):
        validate_online_runtime_contract(payload)
