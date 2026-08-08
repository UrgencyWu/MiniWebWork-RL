"""Frozen contracts for the focused M4 long-horizon Agent RL study.

The historical ``m4_rlvr_v3`` study remains diagnostic.  This module gives the
focused study a disjoint identity and deliberately keeps formal submission
closed until a readiness manifest is produced after all preflight gates pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_SCHEMA = "m4_long_horizon_study_v2"
STUDY_ID = "m4_long_horizon_credit_v1"
STUDY_MANIFEST_PATH = PROJECT_ROOT / "data" / "m4_long_horizon_study_v2.json"
PROMPT_CONTRACT = "browser_agent_v4_long_memory"
PROMPT_PATH = PROJECT_ROOT / "prompts" / f"{PROMPT_CONTRACT}.txt"
PROMPT_CONTEXT_CONTRACT = {
    "prompt_version": PROMPT_CONTRACT,
    "max_visible_text_characters": 5000,
    "history_window": 5,
    "max_control_elements": 16,
    "max_link_elements": 16,
    "evidence_memory_contract": "public_observation_v1",
    "max_evidence_entries": 8,
    "max_evidence_text_characters_per_entry": 1000,
    "evidence_page_types": ["product_detail", "supplier_detail"],
    "current_url_contract": "path_only_without_origin_query_or_fragment",
}
FORMAL_METHODS = ("verified_sft", "multi_turn_grpo", "step_aware_gpo")
ONLINE_METHODS = ("multi_turn_grpo", "step_aware_gpo")
ONLINE_SEEDS = (20260801, 20260802, 20260803)
GROUP_SIZE = 4
ACTION_TOKEN_CAP = 250_000
MAX_TASKS_PER_ITERATION = 32
MAX_SEQUENCE_LENGTH = 6_144
MAX_MODEL_TURNS = 20
MAX_ENVIRONMENT_STEPS = 20
MAX_NEW_TOKENS = 128
CREDIT_FORMULA_VERSION = "public_anchor_macro_micro_v1"
CREDIT_ADVANTAGE_EPSILON = 1e-6
CREDIT_MICRO_RETURN_GAMMA = 0.95
CREDIT_MICRO_WEIGHT = 1.0
CREDIT_ANCHOR_VISIT_POLICY = "first_visit_per_trajectory"
TASK_SAMPLER_VERSION = "balanced_cold_then_beta_uncertainty_v1"
TASK_SAMPLER_MINIMUM_WEIGHT = 0.05
SFT_SEED = 20260801
SFT_LEARNING_RATE = 2e-4
SFT_EFFECTIVE_BATCH_SIZE = 16
SFT_MICROBATCH_CANDIDATES = (1, 2, 4, 8)
SFT_VRAM_HEADROOM_MINIMUM = 0.15
SFT_PREFLIGHT_MAXIMUM_OPTIMIZER_UPDATES = 20
SFT_LORA_CONFIG = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}
SFT_STOPPING_RULE = {
    "maximum_epochs": 3,
    "minimum_epochs": 2,
    "plateau_patience_evaluations": 1,
    "dev_nll_minimum_improvement": 0.005,
    "teacher_forced_action_exact_minimum_improvement": 0.002,
    "teacher_forced_schema_valid_minimum_improvement": 0.002,
}
ONLINE_LEARNER_CONFIG = {
    "policy_epochs": 2,
    "trajectory_minibatch_size": 4,
    "learning_rate": 5e-6,
    "clip_epsilon": 0.2,
    "gradient_clip": 1.0,
}
HORIZON_RANGES = {
    "basic": (6, 8),
    "medium": (9, 12),
    "long": (13, 20),
}
SPLIT_COUNTS = {"train": 240, "dev": 72, "test": 120}
DATASET_ID = "m4_long_horizon_v2"
DATASET_ROOT = PROJECT_ROOT / "data" / "tasks" / DATASET_ID
SEED_ROOT = PROJECT_ROOT / "data" / "seed_m4_long_horizon_v2"
ISOLATION_FIELDS = (
    "world_signature",
    "product_signature",
    "supplier_signature",
    "constraint_signature",
    "answer_signature",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolved_project_path(relative_path: str) -> Path:
    _require(isinstance(relative_path, str) and relative_path, "path must be a non-empty string")
    candidate = (PROJECT_ROOT / relative_path).resolve()
    project = PROJECT_ROOT.resolve()
    _require(
        os.path.commonpath((str(candidate), str(project))) == str(project),
        f"path escapes project root: {relative_path}",
    )
    return candidate


def validate_study_manifest(payload: Mapping[str, Any]) -> None:
    """Fail closed on drift in the pre-registered focused-study contract."""

    _require(payload.get("schema_version") == STUDY_SCHEMA, "study schema drift")
    _require(payload.get("study_id") == STUDY_ID, "study id drift")
    _require(payload.get("lifecycle_state") == "preflight", "study must remain preflight")
    _require(payload.get("formal_submission_allowed") is False, "formal submission opened early")
    _require(payload.get("prompt_contract") == PROMPT_CONTRACT, "prompt contract drift")
    _require(PROMPT_PATH.is_file(), "frozen prompt file is missing")
    _require(
        payload.get("prompt_system_sha256") == _sha256(PROMPT_PATH),
        "prompt system hash drift",
    )
    _require(
        payload.get("prompt_context_contract") == PROMPT_CONTEXT_CONTRACT,
        "prompt context contract drift",
    )

    matrix = payload.get("formal_matrix", {})
    shared_sft = matrix.get("shared_sft", {})
    _require(shared_sft.get("algorithm") == FORMAL_METHODS[0], "shared SFT drift")
    _require(shared_sft.get("model_count") == 1, "focused study requires one shared SFT")
    _require(tuple(matrix.get("online_methods", ())) == ONLINE_METHODS, "online method drift")
    _require(tuple(matrix.get("online_seeds", ())) == ONLINE_SEEDS, "online seed drift")
    _require(matrix.get("online_model_count") == 6, "online model count drift")
    _require(matrix.get("total_model_count") == 7, "formal model count drift")
    excluded = set(matrix.get("excluded_formal_algorithms", ()))
    _require(excluded == {"rsft", "rloo", "gspo"}, "excluded algorithm set drift")

    dataset = payload.get("dataset_contract", {})
    _require(dataset.get("dataset_id") == DATASET_ID, "dataset id drift")
    _require(
        _resolved_project_path(dataset.get("task_root", "")) == DATASET_ROOT.resolve(),
        "task root drift",
    )
    _require(
        _resolved_project_path(dataset.get("seed_root", "")) == SEED_ROOT.resolve(),
        "seed root drift",
    )
    for key in ("dataset_manifest_sha256", "seed_manifest_sha256"):
        value = dataset.get(key)
        _require(isinstance(value, str) and len(value) == 64, f"invalid {key}")
    _require(dataset.get("split_counts") == SPLIT_COUNTS, "dataset split count drift")
    _require(dataset.get("horizon_strata") == {key: list(value) for key, value in HORIZON_RANGES.items()}, "horizon contract drift")
    _require(dataset.get("test_outcomes_available_during_preflight") is False, "test outcome gate opened")
    _require(tuple(dataset.get("split_isolation_fields", ())) == ISOLATION_FIELDS, "isolation field drift")
    _require(
        dataset.get("correct_supplier_visit_position_contract")
        == "balanced across positions 1,2,3 in every split",
        "long-task position shortcut contract drift",
    )
    _require(
        dataset.get("winner_assignment")
        == "split_balanced_sha256_permutation_v1",
        "winner assignment contract drift",
    )
    _require(
        dataset.get("identifier_or_name_fixed_optimum_allowed") is False,
        "identifier shortcut was enabled",
    )

    sft = payload.get("sft_contract", {})
    _require(
        sft.get("prompt_evidence_memory")
        == "bounded policy-visible supplier/product observations only",
        "SFT evidence-memory contract drift",
    )
    _require(sft.get("seed") == SFT_SEED, "SFT seed drift")
    _require(sft.get("learning_rate") == SFT_LEARNING_RATE, "SFT learning rate drift")
    _require(sft.get("maximum_sequence_length") == MAX_SEQUENCE_LENGTH, "SFT max length drift")
    _require(sft.get("lora") == SFT_LORA_CONFIG, "SFT LoRA contract drift")
    _require(sft.get("stopping") == SFT_STOPPING_RULE, "SFT stopping rule drift")
    _require(sft.get("microbatch_candidates") == list(SFT_MICROBATCH_CANDIDATES), "SFT microbatch candidates drift")
    _require(sft.get("effective_batch_size") == SFT_EFFECTIVE_BATCH_SIZE, "SFT effective batch drift")
    _require(sft.get("vram_headroom_minimum") == SFT_VRAM_HEADROOM_MINIMUM, "SFT VRAM headroom drift")
    _require(
        sft.get("preflight_maximum_optimizer_updates")
        == SFT_PREFLIGHT_MAXIMUM_OPTIMIZER_UPDATES,
        "SFT preflight update cap drift",
    )

    online = payload.get("online_contract", {})
    _require(online.get("group_size") == GROUP_SIZE, "K drift")
    _require(online.get("generated_action_token_cap_per_seed") == ACTION_TOKEN_CAP, "token budget drift")
    _require(online.get("maximum_tasks_per_iteration") == MAX_TASKS_PER_ITERATION, "iteration size drift")
    _require(online.get("sampling") == {"temperature": 1.0, "top_p": 1.0, "top_k": 0}, "sampling drift")
    _require(online.get("behavior_policy_staleness") == 0, "on-policy staleness drift")
    _require(online.get("incomplete_group_may_update") is False, "partial group update enabled")
    _require(online.get("invalid_and_zero_signal_tokens_count_toward_budget") is True, "cost accounting weakened")
    _require(online.get("maximum_model_turns") == MAX_MODEL_TURNS, "model-turn cap drift")
    _require(online.get("maximum_environment_steps") == MAX_ENVIRONMENT_STEPS, "environment-step cap drift")
    _require(online.get("maximum_new_tokens_per_turn") == MAX_NEW_TOKENS, "new-token cap drift")
    _require(online.get("learner") == ONLINE_LEARNER_CONFIG, "online learner contract drift")
    sampler = online.get("task_sampler", {})
    _require(sampler.get("version") == TASK_SAMPLER_VERSION, "task sampler version drift")
    _require(sampler.get("cold_coverage_before_revisit") is True, "task sampler cold coverage disabled")
    _require(sampler.get("posterior") == "Beta(1,1)", "task sampler posterior drift")
    _require(sampler.get("uncertainty_weight") == "max(0.05, 4*p*(1-p))", "task sampler weight drift")
    _require(sampler.get("balance_key") == "task_family", "task sampler balance drift")
    _require(sampler.get("deterministic_uncertainty_priority_with_sha_tie_break") is True, "task sampler priority drift")
    _require(sampler.get("outcomes_allowed") == "committed train groups only", "task sampler outcome leakage")

    reward = payload.get("reward_contract", {})
    _require(reward.get("verified_success") == 1.0, "success reward drift")
    _require(reward.get("valid_policy_failure") == 0.0, "policy failure reward drift")
    _require("infrastructure_failure" in reward and reward["infrastructure_failure"] is None, "infra failure reward drift")
    _require(reward.get("process_or_format_rewards_in_primary_experiment") is False, "unregistered shaping reward enabled")

    credit = payload.get("credit_assignment_contract", {})
    _require(credit.get("formula_version") == CREDIT_FORMULA_VERSION, "credit formula drift")
    _require(credit.get("advantage_epsilon") == CREDIT_ADVANTAGE_EPSILON, "credit epsilon drift")
    _require(credit.get("micro_return_gamma") == CREDIT_MICRO_RETURN_GAMMA, "credit gamma drift")
    _require(credit.get("micro_advantage_weight") == CREDIT_MICRO_WEIGHT, "credit omega drift")
    _require(credit.get("anchor_visit_policy") == CREDIT_ANCHOR_VISIT_POLICY, "anchor visit policy drift")
    _require(credit.get("anchor_state_source") == "public policy-visible observation plus exact prompt-token context", "anchor source drift")
    _require(credit.get("normalization") == "token mean within turn; turn mean within trajectory; trajectory mean within K group", "credit normalization drift")
    _require(credit.get("no_informative_anchor_fallback") == "exact macro advantage", "credit fallback drift")

    resources = payload.get("resource_contract", {})
    _require(resources.get("maximum_concurrent_gpu_jobs") == 4, "GPU concurrency drift")
    _require(resources.get("gpus_per_job") == 1, "GPU-per-job drift")
    _require(resources.get("wall_time") == "24:00:00", "Slurm wall-time drift")

    outputs = payload.get("output_contract", {})
    roots = [
        _resolved_project_path(outputs[key])
        for key in ("preflight_root", "formal_root", "diagnostic_root", "readiness_root")
    ]
    _require(len(set(roots)) == len(roots), "study output roots overlap exactly")
    root = _resolved_project_path(outputs["root"])
    for child in roots:
        _require(os.path.commonpath((str(root), str(child))) == str(root), "study output escapes root")


def load_study_manifest(path: Path = STUDY_MANIFEST_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_study_manifest(payload)
    return {"path": str(resolved), "sha256": _sha256(resolved), "payload": payload}


def assert_dataset_binding(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Bind the study to the exact checked task and seed manifests."""

    payload = dict(manifest) if manifest is not None else load_study_manifest()["payload"]
    validate_study_manifest(payload)
    contract = payload["dataset_contract"]
    task_root = _resolved_project_path(contract["task_root"])
    seed_root = _resolved_project_path(contract["seed_root"])
    dataset_manifest_path = task_root / "dataset_manifest.json"
    seed_manifest_path = seed_root / "manifest.json"
    _require(dataset_manifest_path.is_file(), "checked dataset manifest is missing")
    _require(seed_manifest_path.is_file(), "checked seed manifest is missing")
    _require(
        _sha256(dataset_manifest_path) == contract["dataset_manifest_sha256"],
        "checked dataset manifest hash mismatch",
    )
    _require(
        _sha256(seed_manifest_path) == contract["seed_manifest_sha256"],
        "checked seed manifest hash mismatch",
    )
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    _require(dataset_manifest.get("dataset_id") == DATASET_ID, "checked dataset id mismatch")
    _require(dataset_manifest.get("split_task_counts") == SPLIT_COUNTS, "checked split count mismatch")
    _require(seed_manifest.get("seed_version") == DATASET_ID, "checked seed version mismatch")
    shortcut_audit = dataset_manifest.get("shortcut_audit", {})
    _require(shortcut_audit.get("balanced") is True, "checked dataset has a position shortcut")
    _require(
        shortcut_audit.get("winner_assignment_version")
        == contract.get("winner_assignment"),
        "checked winner assignment version drift",
    )
    _require(shortcut_audit.get("non_periodic") is True, "checked dataset has an ID-period shortcut")
    expected_positions = {
        split: {
            str(position): task_count // 4 // 3
            for position in (1, 2, 3)
        }
        for split, task_count in SPLIT_COUNTS.items()
    }
    _require(
        shortcut_audit.get("correct_supplier_visit_position_counts")
        == expected_positions,
        "checked dataset position balance drift",
    )
    return {
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
        "seed_manifest_path": str(seed_manifest_path),
        "seed_manifest_sha256": contract["seed_manifest_sha256"],
    }


def assert_preflight_output(path: Path, manifest: Mapping[str, Any] | None = None) -> Path:
    """Allow writes only under the disjoint preflight namespace."""

    payload = dict(manifest) if manifest is not None else load_study_manifest()["payload"]
    validate_study_manifest(payload)
    expected = _resolved_project_path(payload["output_contract"]["preflight_root"])
    requested = Path(path).expanduser()
    actual = requested.resolve() if requested.is_absolute() else (PROJECT_ROOT / requested).resolve()
    _require(os.path.commonpath((str(expected), str(actual))) == str(expected), f"not a preflight output: {actual}")
    return actual


def assert_formal_submission_closed(manifest: Mapping[str, Any] | None = None) -> None:
    payload = dict(manifest) if manifest is not None else load_study_manifest()["payload"]
    validate_study_manifest(payload)
    if payload["formal_submission_allowed"]:
        raise AssertionError("preflight study manifest unexpectedly permits formal submission")


def classify_horizon(oracle_min_env_actions: int) -> str:
    _require(isinstance(oracle_min_env_actions, int) and not isinstance(oracle_min_env_actions, bool), "oracle horizon must be an integer")
    for stratum, (lower, upper) in HORIZON_RANGES.items():
        if lower <= oracle_min_env_actions <= upper:
            return stratum
    raise ValueError(f"oracle horizon outside frozen 6-20 range: {oracle_min_env_actions}")


def deterministic_task_order(task_ids: Iterable[str], seed: int) -> tuple[str, ...]:
    ordered = list(task_ids)
    _require(ordered and all(isinstance(item, str) and item for item in ordered), "task ids must be non-empty strings")
    _require(len(set(ordered)) == len(ordered), "task ids must be unique")
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def audit_horizon_dataset(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Validate counts, oracle horizon evidence, strata, and cross-split isolation."""

    _require(set(rows_by_split) == set(SPLIT_COUNTS), "dataset must contain exactly train/dev/test")
    report: dict[str, Any] = {"splits": {}, "isolation": {}}
    global_task_ids: set[str] = set()
    signatures: dict[str, dict[str, str]] = {field: {} for field in ISOLATION_FIELDS}
    for split, expected_count in SPLIT_COUNTS.items():
        rows = rows_by_split[split]
        _require(len(rows) == expected_count, f"{split} requires {expected_count} tasks")
        counts = {key: 0 for key in HORIZON_RANGES}
        local_ids: set[str] = set()
        for row in rows:
            task_id = row.get("task_id")
            _require(isinstance(task_id, str) and task_id, f"{split} task lacks task_id")
            _require(task_id not in local_ids and task_id not in global_task_ids, f"duplicate task_id: {task_id}")
            local_ids.add(task_id)
            global_task_ids.add(task_id)
            horizon = row.get("oracle_min_env_actions")
            stratum = classify_horizon(horizon)
            _require(row.get("horizon_stratum") == stratum, f"horizon label mismatch: {task_id}")
            oracle_sha = row.get("oracle_trace_sha256")
            _require(isinstance(oracle_sha, str) and len(oracle_sha) == 64, f"invalid oracle trace hash: {task_id}")
            counts[stratum] += 1
            for field in ISOLATION_FIELDS:
                value = row.get(field)
                _require(isinstance(value, str) and value, f"{task_id} lacks {field}")
                prior_split = signatures[field].get(value)
                _require(prior_split in (None, split), f"cross-split {field} leakage: {value}")
                signatures[field][value] = split
        medium_long = counts["medium"] + counts["long"]
        _require(medium_long * 3 >= expected_count * 2, f"{split} has less than two-thirds medium/long tasks")
        _require(all(counts[key] > 0 for key in counts), f"{split} must contain every horizon stratum")
        report["splits"][split] = {
            "task_count": expected_count,
            "horizon_counts": counts,
            "medium_long_fraction": medium_long / expected_count,
        }
    report["isolation"] = {field: len(values) for field, values in signatures.items()}
    report["valid"] = True
    return report
