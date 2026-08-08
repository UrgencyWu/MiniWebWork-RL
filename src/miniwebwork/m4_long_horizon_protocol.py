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
STUDY_SCHEMA = "m4_long_horizon_study_v1"
STUDY_ID = "m4_long_horizon_credit_v1"
STUDY_MANIFEST_PATH = PROJECT_ROOT / "data" / "m4_long_horizon_study_v1.json"
PROMPT_CONTRACT = "browser_agent_v3_compact"
FORMAL_METHODS = ("verified_sft", "multi_turn_grpo", "step_aware_gpo")
ONLINE_METHODS = ("multi_turn_grpo", "step_aware_gpo")
ONLINE_SEEDS = (20260801, 20260802, 20260803)
GROUP_SIZE = 4
ACTION_TOKEN_CAP = 250_000
MAX_TASKS_PER_ITERATION = 32
HORIZON_RANGES = {
    "basic": (6, 8),
    "medium": (9, 12),
    "long": (13, 20),
}
SPLIT_COUNTS = {"train": 240, "dev": 72, "test": 120}
DATASET_ID = "m4_long_horizon_v1"
DATASET_ROOT = PROJECT_ROOT / "data" / "tasks" / DATASET_ID
SEED_ROOT = PROJECT_ROOT / "data" / "seed_m4_long_horizon_v1"
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

    online = payload.get("online_contract", {})
    _require(online.get("group_size") == GROUP_SIZE, "K drift")
    _require(online.get("generated_action_token_cap_per_seed") == ACTION_TOKEN_CAP, "token budget drift")
    _require(online.get("maximum_tasks_per_iteration") == MAX_TASKS_PER_ITERATION, "iteration size drift")
    _require(online.get("sampling") == {"temperature": 1.0, "top_p": 1.0, "top_k": 0}, "sampling drift")
    _require(online.get("behavior_policy_staleness") == 0, "on-policy staleness drift")
    _require(online.get("incomplete_group_may_update") is False, "partial group update enabled")
    _require(online.get("invalid_and_zero_signal_tokens_count_toward_budget") is True, "cost accounting weakened")

    reward = payload.get("reward_contract", {})
    _require(reward.get("verified_success") == 1.0, "success reward drift")
    _require(reward.get("valid_policy_failure") == 0.0, "policy failure reward drift")
    _require("infrastructure_failure" in reward and reward["infrastructure_failure"] is None, "infra failure reward drift")
    _require(reward.get("process_or_format_rewards_in_primary_experiment") is False, "unregistered shaping reward enabled")

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
