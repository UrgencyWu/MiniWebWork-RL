from __future__ import annotations

import asyncio
import json

import pytest

from miniwebwork.long_horizon_rl.online_runner import (
    _load_or_write_run_config,
    identity_from_run_state,
    run_online_preflight_once,
    summarize_collection_performance,
)


def _turn(tokens, queue, first, generation):
    return {
        "generated_token_ids": list(range(tokens)),
        "queue_wait_ms": queue,
        "first_token_latency_ms": first,
        "generation_time_ms": generation,
    }


def test_collection_performance_reports_true_hierarchical_counts_and_rates():
    groups = [
        {
            "trajectories": [
                {"turns": [_turn(2, 1, 2, 3), _turn(3, 2, 3, 4)]},
                {"turns": [_turn(4, 3, 4, 5)]},
                {"turns": [_turn(1, 4, 5, 6)]},
                {"turns": [_turn(2, 5, 6, 7)]},
            ]
        }
    ]
    report = summarize_collection_performance(groups, elapsed_seconds=120.0)
    assert report["group_count"] == 1
    assert report["trajectory_count"] == 4
    assert report["model_turn_count"] == 5
    assert report["generated_action_tokens"] == 12
    assert report["trajectories_per_hour"] == 120.0
    assert report["generated_action_tokens_per_hour"] == 360.0
    assert report["trajectory_model_turns"] == {
        "minimum": 1,
        "p50": 1.0,
        "p95": 2.0,
        "maximum": 2,
    }


def test_run_config_is_immutable_and_self_hashed(tmp_path):
    payload = {"schema_version": "x", "seed": 1}
    first = _load_or_write_run_config(tmp_path, payload)
    assert first["run_config_sha256"]
    assert _load_or_write_run_config(tmp_path, payload) == first
    with pytest.raises(ValueError, match="config drift"):
        _load_or_write_run_config(tmp_path, {**payload, "seed": 2})

    tampered = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    tampered["seed"] = 3
    (tmp_path / "run_config.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="config drift"):
        _load_or_write_run_config(tmp_path, payload)


def test_identity_reconstruction_binds_current_policy_and_adapter():
    state = {
        "study_id": "m4_long_horizon_credit_v1",
        "git_sha": "1" * 40,
        "method": "multi_turn_grpo",
        "seed": 20260801,
        "current_iteration_index": 3,
        "current_policy_version": "policy_0003",
        "dataset_manifest_sha256": "2" * 64,
        "seed_manifest_sha256": "3" * 64,
        "prompt_sha256": "4" * 64,
        "credit_formula_version": "public_anchor_macro_micro_v1",
        "task_order_sha256": "5" * 64,
        "base_model_manifest_sha256": "8" * 64,
        "runtime_contract_sha256": "9" * 64,
        "current_adapter": {"sha256": "6" * 64},
        "current_rollout_adapter": {"sha256": "7" * 64},
        "current_adapter_semantic_sha256": "a" * 64,
    }
    identity = identity_from_run_state(state)
    assert identity.iteration_index == 3
    assert identity.policy_version == "policy_0003"
    assert identity.input_adapter_sha256 == "6" * 64
    assert identity.input_rollout_adapter_sha256 == "7" * 64
    assert identity.input_adapter_semantic_sha256 == "a" * 64


def test_recovered_commit_without_complete_report_fails_same_gpu_gate(tmp_path):
    state = {
        "current_iteration_index": 1,
        "current_policy_version": "policy_0001",
        "current_adapter": {"sha256": "a" * 64},
        "current_rollout_adapter": {"sha256": "c" * 64},
        "current_adapter_semantic_sha256": "d" * 64,
        "last_iteration_manifest_sha256": "b" * 64,
    }

    class AdvancedStateStore:
        def reconcile_committed_iterations(self):
            return {"reconciled_iterations": 1, "state": state}

        def load_state(self):
            return dict(state)

    prepared = {
        "root": tmp_path,
        "run_config": {"run_config_sha256": "c" * 64},
        "state_store": AdvancedStateStore(),
    }
    with pytest.raises(RuntimeError, match="same-GPU sleep/wake was not proven"):
        asyncio.run(run_online_preflight_once(prepared))

    assert not (tmp_path / "preflight_report.json").exists()
    recovery = json.loads(
        (tmp_path / "recovered_committed_update.json").read_text(encoding="utf-8")
    )
    assert recovery["result"] == "RECOVERED_COMMITTED_UPDATE_PHASE_GATE_FAILED"
    assert recovery["recovered_after_committed_update"] is True
    assert recovery["current_rollout_adapter_sha256"] == "c" * 64
    assert recovery["current_adapter_semantic_sha256"] == "d" * 64
    assert recovery["same_gpu_phase_switch"]["passed"] is False
    assert recovery["complete"] is False
