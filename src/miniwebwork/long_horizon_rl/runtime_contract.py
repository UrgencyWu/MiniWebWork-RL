"""Frozen, fail-closed contract for the focused single-GPU online runtime."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..m4_long_horizon_protocol import (
    ACTION_TOKEN_CAP,
    GROUP_SIZE,
    MAX_ENVIRONMENT_STEPS,
    MAX_MODEL_TURNS,
    MAX_NEW_TOKENS,
    MAX_SEQUENCE_LENGTH,
    MAX_TASKS_PER_ITERATION,
    ONLINE_LEARNER_CONFIG,
    PROJECT_ROOT,
    STUDY_ID,
    STUDY_MANIFEST_PATH,
)
from .contracts import sha256_file
from .sft_selection import SFT_SELECTION_PATH

RUNTIME_CONTRACT_SCHEMA = "m4_long_horizon_runtime_v1"
RUNTIME_CONTRACT_PATH = PROJECT_ROOT / "data" / "m4_long_horizon_runtime_v1.json"
EXPECTED_EVIDENCE_SCHEMAS = {
    "run_identity_schema": "m4_long_horizon_run_identity_v2",
    "turn_schema": "m4_long_horizon_turn_evidence_v2",
    "trajectory_schema": "m4_long_horizon_trajectory_v2",
    "group_schema": "m4_long_horizon_group_v2",
    "collection_schema": "m4_long_horizon_collection_v2",
    "journal_schema": "m4_long_horizon_attempt_journal_v2",
    "iteration_schema": "m4_long_horizon_iteration_v1",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_online_runtime_contract(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == RUNTIME_CONTRACT_SCHEMA, "runtime schema drift")
    _require(payload.get("study_id") == STUDY_ID, "runtime study drift")
    _require(payload.get("lifecycle_state") == "preflight", "runtime must remain preflight")
    _require(payload.get("formal_submission_allowed") is False, "runtime opened formal training")
    _require(
        payload.get("study_manifest_sha256") == sha256_file(STUDY_MANIFEST_PATH),
        "runtime study-manifest binding drift",
    )
    _require(
        payload.get("sft_preflight_selection_sha256") == sha256_file(SFT_SELECTION_PATH),
        "runtime SFT-selection binding drift",
    )
    _require(
        payload.get("software_contract")
        == {
            "python": "3.11.14",
            "torch": "2.10.0+cu128",
            "transformers": "5.14.1",
            "peft": "0.19.1",
            "accelerate": "1.13.0",
            "safetensors": "0.8.0",
            "vllm": "0.17.0",
            "playwright": "1.61.0",
        },
        "online software contract drift",
    )

    model = payload.get("model_contract", {})
    _require(model.get("base_model_path") == "/data/share/model/Qwen3.5-4B", "base model drift")
    _require(model.get("language_model_only") is True, "language-model-only gate disabled")
    _require(model.get("maximum_model_length") == MAX_SEQUENCE_LENGTH, "model length drift")
    _require(model.get("maximum_new_tokens_per_turn") == MAX_NEW_TOKENS, "turn token cap drift")
    _require(model.get("maximum_model_turns") == MAX_MODEL_TURNS, "model-turn cap drift")
    _require(
        model.get("maximum_environment_steps") == MAX_ENVIRONMENT_STEPS,
        "environment-step cap drift",
    )
    _require(model.get("lora_rank") == 16, "runtime LoRA rank drift")

    generation = payload.get("generation_contract", {})
    _require(generation.get("backend") == "vllm_async", "generation backend drift")
    _require(generation.get("raw_logprobs_required") is True, "raw logprobs disabled")
    _require(generation.get("sampling_logprobs_required") is True, "sampling logprobs disabled")
    _require(
        {
            "temperature": generation.get("temperature"),
            "top_p": generation.get("top_p"),
            "top_k": generation.get("top_k"),
        }
        == {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
        "sampling distribution drift",
    )
    _require(generation.get("maximum_sequences") == 8, "vLLM maximum sequences drift")
    _require(generation.get("enable_sleep_mode") is True, "same-GPU sleep mode disabled")
    _require(generation.get("generation_during_learner") is False, "stale generation enabled")

    rollout = payload.get("rollout_contract", {})
    _require(rollout.get("browser_worker_candidates") == [1, 2, 4, 8], "worker candidates drift")
    _require(rollout.get("group_size") == GROUP_SIZE, "runtime K drift")
    _require(rollout.get("maximum_tasks_per_iteration") == MAX_TASKS_PER_ITERATION, "iteration task cap drift")
    _require(
        rollout.get("action_token_budget_per_method_seed") == ACTION_TOKEN_CAP,
        "runtime token budget drift",
    )
    _require(
        rollout.get("maximum_group_token_reserve")
        == GROUP_SIZE * MAX_MODEL_TURNS * MAX_NEW_TOKENS,
        "group token reserve drift",
    )
    _require(
        rollout.get("incomplete_or_invalid_group")
        == "retain_generated_token_cost_and_resample_entire_group",
        "invalid-group cost semantics drift",
    )

    learner = payload.get("learner_contract", {})
    for field, expected in ONLINE_LEARNER_CONFIG.items():
        _require(learner.get(field) == expected, f"learner {field} drift")
    _require(learner.get("behavior_policy_staleness") == 0, "on-policy staleness drift")
    _require(learner.get("zero_advantage_group") == "skip_and_report", "zero-signal accounting drift")

    evidence = payload.get("evidence_contract", {})
    for field, expected in EXPECTED_EVIDENCE_SCHEMAS.items():
        _require(evidence.get(field) == expected, f"{field} drift")
    _require(evidence.get("turn_charge_before_full_artifact") is True, "turn cost durability disabled")

    recovery = payload.get("recovery_contract", {})
    _require(recovery.get("identity_mismatch") == "fail_closed", "recovery identity gate weakened")
    _require(
        recovery.get("interrupted_update")
        == "archive_partial_stage_and_replay_frozen_collection",
        "interrupted-update recovery drift",
    )

    telemetry = payload.get("telemetry_contract", {})
    _require(telemetry.get("external_interval_seconds") == 5, "telemetry interval drift")
    gates = telemetry.get("gates", {})
    _require(math.isclose(gates.get("rollout_trajectories_per_hour_minimum"), 114.0), "rollout gate drift")
    _require(math.isclose(gates.get("generation_gpu_utilization_p50_minimum"), 0.6), "generation utilization gate drift")
    _require(math.isclose(gates.get("learner_gpu_utilization_p50_minimum"), 0.8), "learner utilization gate drift")
    _require(math.isclose(gates.get("vram_headroom_minimum"), 0.15), "VRAM gate drift")
    _require(math.isclose(gates.get("optimizer_action_token_fraction_target"), 0.2), "optimizer-token target drift")

    slurm = payload.get("slurm_contract", {})
    _require(slurm == {
        "gpus": 1,
        "cpus": 8,
        "memory_gb": 48,
        "wall_time": "24:00:00",
        "maximum_concurrent_gpu_jobs": 4,
        "real_resume_smoke_required": True,
    }, "online Slurm contract drift")


def load_online_runtime_contract(path: Path = RUNTIME_CONTRACT_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_online_runtime_contract(payload)
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}
