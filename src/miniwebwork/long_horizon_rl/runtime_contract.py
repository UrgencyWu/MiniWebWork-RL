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
from .model_manifest import BASE_MODEL_MANIFEST_PATH, validate_base_model_manifest
from .sft_selection import SFT_SELECTION_PATH

RUNTIME_CONTRACT_SCHEMA = "m4_long_horizon_runtime_v8"
RUNTIME_CONTRACT_PATH = PROJECT_ROOT / "data" / "m4_long_horizon_runtime_v8.json"
EXPECTED_EVIDENCE_SCHEMAS = {
    "run_identity_schema": "m4_long_horizon_run_identity_v3",
    "turn_schema": "m4_long_horizon_turn_evidence_v3",
    "trajectory_schema": "m4_long_horizon_trajectory_v3",
    "group_schema": "m4_long_horizon_group_v3",
    "collection_schema": "m4_long_horizon_collection_v3",
    "journal_schema": "m4_long_horizon_attempt_journal_v3",
    "iteration_schema": "m4_long_horizon_iteration_v2",
}
EXPECTED_ADAPTER_VIEW_CONTRACT = {
    "schema": "m4_qwen35_vllm_adapter_view_v1",
    "mapping_contract": "qwen35_peft_model_layers_to_conditional_language_model_v1",
    "canonical_namespace": "base_model.model.model.",
    "rollout_namespace": "base_model.model.model.language_model.",
    "expected_tensor_count": 256,
    "expected_module_count": 128,
    "normalized_semantic_sha256_required": True,
    "atomic_derived_view_required": True,
}
EXPECTED_TURN_LINEAGE = [
    "group_id",
    "attempt_index",
    "trajectory_id",
    "rollout_index",
    "turn_index",
    "request_id",
    "sampling_seed",
    "policy_version",
    "adapter_sha256",
    "rollout_adapter_sha256",
    "adapter_semantic_sha256",
    "prompt_token_ids",
    "generated_token_ids",
    "behavior_logprobs",
    "sampling_logprobs",
]
PARITY_THRESHOLDS = {
    "behavior_sampling_maximum_absolute_difference": 1e-7,
    "replay_mean_absolute_difference": 0.02,
    "replay_p95_absolute_difference": 0.08,
    "replay_p99_absolute_difference": 0.08,
    "replay_p999_absolute_difference": 0.5,
    "replay_initial_ratio_clip_fraction": 0.005,
    "mean_importance_ratio_absolute_deviation": 0.02,
}
PARITY_CALIBRATION = {
    "status": "versioned_after_gpu_preflight_before_formal_training",
    "positive_job_id": 1271,
    "positive_token_count": 4143,
    "positive_mean_absolute_logprob_difference": 0.001631149261297419,
    "positive_p95_absolute_logprob_difference": 0.00043759867548942566,
    "positive_maximum_absolute_logprob_difference": 0.36713528633117676,
    "token_diagnostic_job_id": 1273,
    "positive_p99_absolute_logprob_difference": 0.04820847511291504,
    "positive_initial_ratio_clip_count": 6,
    "positive_initial_ratio_clip_fraction": 0.001448225923244026,
    "negative_job_id": 1267,
    "negative_token_count": 1667,
    "negative_mean_absolute_logprob_difference": 1.1867696967614851,
    "negative_p95_absolute_logprob_difference": 10.127390384674072,
    "negative_maximum_absolute_logprob_difference": 21.51203542947769,
    "runtime_v2_validation_job_id": 1280,
    "runtime_v2_token_count": 4141,
    "runtime_v2_mean_absolute_logprob_difference": 0.002011027746882088,
    "runtime_v2_p95_absolute_logprob_difference": 0.00041890458669513464,
    "runtime_v2_p99_absolute_logprob_difference": 0.05867719650268555,
    "runtime_v2_maximum_absolute_logprob_difference": 0.6765303611755371,
    "runtime_v2_initial_ratio_clip_count": 9,
    "runtime_v2_initial_ratio_clip_fraction": 0.0021733880705143687,
    "batch_shape_diagnostic_job_id": 1281,
    "batch_shape_maximum_absolute_differences": {
        "microbatch_8_repeat_a": 0.6765303611755371,
        "microbatch_8_repeat_b": 0.6765303611755371,
        "microbatch_4": 1.9843209385871887,
        "microbatch_1": 1.2449634075164795,
    },
    "runtime_v3_change": "disable_experimental_qwen35_mamba_prefix_caching_without_threshold_change",
    "runtime_v3_validation_job_id": 1285,
    "runtime_v3_token_count": 4096,
    "runtime_v3_mean_absolute_logprob_difference": 0.001755040691261844,
    "runtime_v3_p95_absolute_logprob_difference": 0.0003290991298854351,
    "runtime_v3_p99_absolute_logprob_difference": 0.03335973620414734,
    "runtime_v3_maximum_absolute_logprob_difference": 1.3349303007125854,
    "runtime_v3_mean_importance_ratio": 0.9995195594653701,
    "runtime_v3_initial_ratio_clip_count": 4,
    "runtime_v3_initial_ratio_clip_fraction": 0.0009765625,
    "runtime_v3_batch_shape_diagnostic_job_id": 1286,
    "runtime_v3_batch_shape_p999_absolute_differences": {
        "microbatch_8_repeat_a": 0.24099883437156677,
        "microbatch_8_repeat_b": 0.24099883437156677,
        "microbatch_4": 0.374053955078125,
        "microbatch_1": 0.28792598843574524,
    },
    "runtime_v3_batch_shape_maximum_absolute_differences": {
        "microbatch_8_repeat_a": 1.6405409574508667,
        "microbatch_8_repeat_b": 1.6405409574508667,
        "microbatch_4": 1.4392859935760498,
        "microbatch_1": 1.7181252241134644,
    },
    "runtime_v4_tail_gate": (
        "retain_mean_p95_p99_clip_fraction_and_mean_ratio;replace_batch_composition_sensitive_"
        "single_point_max_0_5_with_p999_0_5_plus_catastrophic_abs_log_ratio_ln10"
    ),
    "runtime_v4_validation_job_id": 1289,
    "runtime_v4_token_count": 4106,
    "runtime_v4_mean_absolute_logprob_difference": 0.0017542248288511468,
    "runtime_v4_p95_absolute_logprob_difference": 0.0003270015586167574,
    "runtime_v4_p99_absolute_logprob_difference": 0.04642307758331299,
    "runtime_v4_p999_absolute_logprob_difference": 0.29570549726486206,
    "runtime_v4_maximum_absolute_log_ratio": 0.8886593580245972,
    "runtime_v4_mean_importance_ratio": 0.9996992543473918,
    "runtime_v4_initial_ratio_clip_count": 6,
    "runtime_v4_initial_ratio_clip_fraction": 0.0014612761811982464,
    "runtime_v4_optimizer_updates": 2,
    "runtime_v4_parameter_change_norm": 0.037170038295178266,
    "runtime_v4_post_wake_behavior_sampling_maximum_absolute_difference": 0.0,
    "runtime_v5_grpo_job_id": 1295,
    "runtime_v5_grpo_token_count": 8040,
    "runtime_v5_grpo_mean_absolute_logprob_difference": 0.002138247400720372,
    "runtime_v5_grpo_p95_absolute_logprob_difference": 0.00024105655029416084,
    "runtime_v5_grpo_p99_absolute_logprob_difference": 0.04517373442649841,
    "runtime_v5_grpo_p999_absolute_logprob_difference": 0.35196539759635925,
    "runtime_v5_grpo_maximum_absolute_log_ratio": 1.9909225702285767,
    "runtime_v5_grpo_initial_ratio_clip_fraction": 0.0016169154228855722,
    "runtime_v5_grpo_mean_importance_ratio": 1.000584248671697,
    "runtime_v5_step_aware_job_id": 1296,
    "runtime_v5_step_aware_token_count": 8144,
    "runtime_v5_step_aware_mean_absolute_logprob_difference": 0.002127327528750038,
    "runtime_v5_step_aware_p95_absolute_logprob_difference": 0.00039223674684762955,
    "runtime_v5_step_aware_p99_absolute_logprob_difference": 0.05528593063354492,
    "runtime_v5_step_aware_p999_absolute_logprob_difference": 0.2462717890739441,
    "runtime_v5_step_aware_maximum_absolute_log_ratio": 2.6679060459136963,
    "runtime_v5_step_aware_initial_ratio_clip_fraction": 0.0019646365422396855,
    "runtime_v5_step_aware_mean_importance_ratio": 0.9996751473203268,
    "runtime_v6_tail_gate_change": (
        "retain_single_token_maximum_as_diagnostic_only;fail_closed_on_exact_"
        "behavior_sampling_identity_plus_mean_p95_p99_p999_clip_fraction_and_mean_ratio"
    ),
    "runtime_v6_step_aware_validation_job_id": 1302,
    "runtime_v6_step_aware_token_count": 7988,
    "runtime_v6_step_aware_mean_absolute_logprob_difference": 0.0015858009922818366,
    "runtime_v6_step_aware_p95_absolute_logprob_difference": 0.0003283759579062462,
    "runtime_v6_step_aware_p99_absolute_logprob_difference": 0.05106997489929199,
    "runtime_v6_step_aware_p999_absolute_logprob_difference": 0.2336139678955078,
    "runtime_v6_step_aware_maximum_absolute_log_ratio_diagnostic": 0.5526777505874634,
    "runtime_v6_step_aware_initial_ratio_clip_fraction": 0.0012518778167250877,
    "runtime_v6_step_aware_mean_importance_ratio": 0.9998510321734483,
    "runtime_v6_step_aware_optimizer_updates": 2,
    "runtime_v6_step_aware_parameter_change_norm": 0.03685925012247875,
    "runtime_v6_step_aware_post_wake_behavior_sampling_maximum_absolute_difference": 0.0,
    "runtime_v7_saturation_job_id": 1311,
    "runtime_v7_group_count": 16,
    "runtime_v7_token_count": 16638,
    "runtime_v7_mean_absolute_logprob_difference": 0.002583167558184004,
    "runtime_v7_p95_absolute_logprob_difference": 0.0007290001958608627,
    "runtime_v7_p99_absolute_logprob_difference": 0.06945174932479858,
    "runtime_v7_p999_absolute_logprob_difference": 0.2876361310482025,
    "runtime_v7_maximum_absolute_log_ratio_diagnostic": 2.426311492919922,
    "runtime_v7_initial_ratio_clip_fraction": 0.001983411467724486,
    "runtime_v7_mean_importance_ratio": 0.9996460643297014,
    "runtime_v7_optimizer_updates": 6,
    "runtime_v7_parameter_change_norm": 0.06980343223858164,
    "runtime_v7_post_wake_behavior_sampling_maximum_absolute_difference": 0.0,
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
    _require(
        model.get("base_model_manifest_path")
        == str(BASE_MODEL_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "base model manifest path drift",
    )
    _require(
        model.get("base_model_manifest_sha256") == sha256_file(BASE_MODEL_MANIFEST_PATH),
        "base model manifest hash drift",
    )
    checked_model = validate_base_model_manifest(
        BASE_MODEL_MANIFEST_PATH,
        verify_files=False,
    )
    _require(
        model.get("base_model_functional_file_set_sha256")
        == checked_model["payload"]["functional_file_set_sha256"],
        "base model functional file-set drift",
    )
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
    _require(generation.get("request_output_kind") == "final_only", "generation output-kind drift")
    _require(generation.get("raw_logprobs_required") is True, "raw logprobs disabled")
    _require(generation.get("sampling_logprobs_required") is True, "sampling logprobs disabled")
    _require(generation.get("flat_logprobs") is True, "flat logprobs disabled")
    _require(
        {
            "temperature": generation.get("temperature"),
            "top_p": generation.get("top_p"),
            "top_k": generation.get("top_k"),
        }
        == {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
        "sampling distribution drift",
    )
    _require(
        math.isclose(generation.get("gpu_memory_utilization"), 0.5),
        "vLLM memory fraction drift",
    )
    _require(generation.get("maximum_sequences") == 32, "vLLM maximum sequences drift")
    _require(
        generation.get("maximum_full_length_kv_tokens_required") == 196_608,
        "vLLM KV-capacity requirement drift",
    )
    _require(
        generation.get("memory_selection_job_id") == 1302,
        "vLLM memory-selection lineage drift",
    )
    _require(
        generation.get("memory_selection_observed_kv_tokens") == 319_968
        and math.isclose(generation.get("memory_selection_post_wake_headroom"), 0.3644099829395119),
        "vLLM memory-selection evidence drift",
    )
    _require(
        generation.get("enable_prefix_caching") is False,
        "experimental Qwen3.5 Mamba prefix caching re-enabled",
    )
    _require(generation.get("enable_chunked_prefill") is True, "chunked prefill disabled")
    _require(
        generation.get("execution_mode")
        == "eager_due_vllm017_qwen35_packed_lora_warmup_incompatibility",
        "vLLM execution-mode drift",
    )
    _require(generation.get("cuda_graphs_enabled") is False, "vLLM CUDA graphs re-enabled")
    _require(generation.get("enable_sleep_mode") is True, "same-GPU sleep mode disabled")
    _require(
        generation.get("cuda_allocator_environment")
        == "pytorch_allocator_aliases_unset_for_vllm_cumem_sleep",
        "vLLM sleep allocator contract drift",
    )
    _require(
        generation.get("adapter_swap_protocol")
        == "remove_lora_then_sleep_level_2_then_same_gpu_hf_learner_then_wake_and_add_lora",
        "adapter swap protocol drift",
    )
    _require(
        generation.get("post_wake_generation_smoke_required") is True,
        "post-wake generation smoke disabled",
    )
    _require(generation.get("generation_during_learner") is False, "stale generation enabled")

    _require(
        payload.get("adapter_view_contract") == EXPECTED_ADAPTER_VIEW_CONTRACT,
        "adapter-view contract drift",
    )

    rollout = payload.get("rollout_contract", {})
    _require(
        rollout.get("browser_worker_candidates") == [1, 2, 4, 8, 16, 32, 64],
        "worker candidates drift",
    )
    _require(
        rollout.get("maximum_concurrent_k4_groups") == 16,
        "concurrent K4 group count drift",
    )
    _require(
        rollout.get("selected_browser_workers") == 32
        and rollout.get("selected_concurrent_k4_groups") == 8,
        "selected rollout candidate drift",
    )
    _require(
        math.isclose(rollout.get("group_launch_stagger_seconds"), 0.0),
        "group launch stagger drift",
    )
    _require(rollout.get("persistent_browser_per_candidate") is True, "persistent browser disabled")
    _require(
        rollout.get("worker_bridge") == "threadsafe_sync_browser_to_main_asyncio_loop",
        "browser worker bridge drift",
    )
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
        rollout.get("maximum_zero_token_no_progress_batches") == 3,
        "zero-token retry limit drift",
    )
    _require(
        rollout.get("zero_token_no_progress_failure")
        == "fail_closed_job_exit_without_collection_freeze",
        "zero-token failure policy drift",
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

    parity = payload.get("parity_contract", {})
    _require(
        parity.get("thresholds_frozen_before_gpu_observation") is False,
        "parity calibration history drift",
    )
    _require(
        parity.get("thresholds_frozen_before_formal_training") is True,
        "parity thresholds are not frozen before formal training",
    )
    _require(
        parity.get("calibration") == PARITY_CALIBRATION,
        "parity calibration evidence drift",
    )
    for field, expected in PARITY_THRESHOLDS.items():
        _require(math.isclose(parity.get(field), expected), f"parity {field} drift")
    _require(
        "replay_maximum_absolute_log_ratio" not in parity,
        "single-token maximum parity gate reintroduced",
    )
    _require(
        parity.get("failure") == "fail_closed_before_optimizer_step",
        "parity failure policy drift",
    )

    evidence = payload.get("evidence_contract", {})
    for field, expected in EXPECTED_EVIDENCE_SCHEMAS.items():
        _require(evidence.get(field) == expected, f"{field} drift")
    _require(evidence.get("turn_charge_before_full_artifact") is True, "turn cost durability disabled")
    _require(
        evidence.get("required_turn_lineage") == EXPECTED_TURN_LINEAGE,
        "required turn lineage drift",
    )

    recovery = payload.get("recovery_contract", {})
    _require(recovery.get("identity_mismatch") == "fail_closed", "recovery identity gate weakened")
    _require(
        recovery.get("interrupted_update")
        == "archive_partial_stage_and_replay_frozen_collection",
        "interrupted-update recovery drift",
    )
    _require(
        recovery.get("post_update_commit_pre_wake")
        == "preserve_commit_but_fail_same_gpu_phase_gate",
        "post-commit phase recovery drift",
    )

    telemetry = payload.get("telemetry_contract", {})
    _require(telemetry.get("external_interval_seconds") == 5, "telemetry interval drift")
    _require(telemetry.get("candidate_soak_minutes") == 30, "candidate soak drift")
    _require(
        telemetry.get("candidate_soak_scope")
        == "aggregate_collection_phase_on_selected_worker_candidate",
        "candidate soak scope drift",
    )
    gates = telemetry.get("gates", {})
    _require(math.isclose(gates.get("rollout_trajectories_per_hour_minimum"), 114.0), "rollout gate drift")
    _require(gates.get("generation_gpu_utilization_p50_diagnostic_only") is True, "generation P50 diagnostic drift")
    _require(math.isclose(gates.get("generation_gpu_utilization_p95_minimum"), 0.5), "generation active-tail gate drift")
    _require(
        gates.get("saturation_selected_browser_workers") == 32
        and gates.get("saturation_challenger_browser_workers") == 64,
        "rollout saturation candidate drift",
    )
    _require(
        math.isclose(gates.get("saturation_challenger_throughput_improvement_maximum"), 1.1)
        and math.isclose(gates.get("saturation_selected_trajectories_per_hour"), 414.1060445946014)
        and math.isclose(gates.get("saturation_challenger_trajectories_per_hour"), 384.11349008080083)
        and gates["saturation_challenger_trajectories_per_hour"]
        / gates["saturation_selected_trajectories_per_hour"]
        <= gates["saturation_challenger_throughput_improvement_maximum"],
        "rollout saturation evidence drift",
    )
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
