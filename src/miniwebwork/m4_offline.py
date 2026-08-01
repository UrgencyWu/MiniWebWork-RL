"""Auditable data gates and command planning for the M4 offline controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .m4_protocol import (
    COLLECTED_ACTION_TOKEN_CAP,
    M4_TRAIN_TASK_COUNT,
    M4RunConfig,
    ONLINE_PASS_ACTION_TOKEN_CAP,
    RSFT_TRAIN_TASKS_PER_PASS,
    build_m4_run_manifest,
    m4_task_roster_sha256,
)

OFFLINE_SUPERVISION_PASSES = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"M4 offline manifest not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"M4 offline manifest must be an object: {path}")
    return value


def _validate_jsonl_split(path: Path, *, expected_split: str, task_prefix: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"M4 offline JSONL not found: {path}")
    task_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != expected_split:
            raise PermissionError(
                f"M4 offline row {path}:{line_number} has split {row.get('split')!r}, "
                f"expected {expected_split!r}"
            )
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id.startswith(task_prefix):
            raise PermissionError(
                f"M4 offline row {path}:{line_number} does not belong to {task_prefix}"
            )
        task_ids.add(task_id)
    if not task_ids:
        raise ValueError(f"M4 offline JSONL contains no rows: {path}")
    return task_ids


def _validate_sft_corpus(data_dir: Path) -> dict[str, Any]:
    manifest = _load_json(data_dir / "manifest.json")
    train = manifest.get("train")
    dev = manifest.get("dev")
    if not isinstance(train, dict) or not isinstance(dev, dict):
        raise ValueError("M4 oracle SFT corpus must contain train and dev manifests")
    required_train = {
        "dataset_id": "m4_oracle_sft_v1",
        "task_source_dataset_id": "m4_rlvr_v1",
        "task_split": "train",
        "purpose": "offline_training",
        "sample_split": "train",
    }
    required_dev = {
        "dataset_id": "m4_oracle_sft_v1",
        "task_source_dataset_id": "m4_rlvr_v1",
        "task_split": "dev",
        "purpose": "model_selection",
        "sample_split": "dev",
    }
    for name, expected, actual in (("train", required_train, train), ("dev", required_dev, dev)):
        mismatches = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise PermissionError(f"M4 SFT {name} manifest violates split boundary: {mismatches}")
    if train.get("task_count") != 240 or dev.get("task_count") != 72:
        raise ValueError("M4 SFT corpus must cover all 240 train and 72 dev tasks")
    for filename, expected_hash in (
        ("train.jsonl", manifest.get("train_sha256")),
        ("valid.jsonl", manifest.get("valid_sha256")),
    ):
        path = data_dir / filename
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise ValueError(f"M4 SFT corpus hash mismatch for {filename}")
    train_task_ids = _validate_jsonl_split(
        data_dir / "train.jsonl", expected_split="train", task_prefix="M4-TRAIN-"
    )
    dev_task_ids = _validate_jsonl_split(
        data_dir / "valid.jsonl", expected_split="dev", task_prefix="M4-DEV-"
    )
    if len(train_task_ids) != 240 or len(dev_task_ids) != 72:
        raise ValueError("M4 SFT rows do not cover every declared train/dev task")
    return {
        "kind": "oracle_sft",
        "data_dir": str(data_dir),
        "train_samples": train.get("sample_count"),
        "dev_samples": dev.get("sample_count"),
        "train_records_sha256": manifest["train_sha256"],
        "valid_records_sha256": manifest["valid_sha256"],
        "selection_boundary": manifest.get("selection_boundary"),
    }


def _validate_rsft_corpus(data_dir: Path) -> dict[str, Any]:
    manifest = _load_json(data_dir / "manifest.json")
    if manifest.get("dataset_id") != "m4_rsft_tokenized_v1" or manifest.get("algorithm") != "rsft":
        raise ValueError("M4 RSFT corpus has an unexpected dataset or algorithm identity")
    if manifest.get("split") != "train" or manifest.get("source_passes") != OFFLINE_SUPERVISION_PASSES:
        raise PermissionError("M4 RSFT corpus must contain exactly two train-only source passes")
    if manifest.get("selected_task_count", 0) <= 0:
        raise ValueError("M4 RSFT corpus has no selected successful trajectories")
    if not isinstance(manifest.get("source_adapter_sha256"), str):
        raise ValueError("M4 RSFT corpus does not identify its initial adapter")
    source_task_ids = manifest.get("source_task_ids")
    if (
        not isinstance(source_task_ids, list)
        or len(source_task_ids) != RSFT_TRAIN_TASKS_PER_PASS
        or any(not isinstance(task_id, str) or not task_id for task_id in source_task_ids)
        or len(set(source_task_ids)) != len(source_task_ids)
    ):
        raise ValueError("M4 RSFT corpus must declare the fixed unique source task roster")
    if manifest.get("source_task_count") != RSFT_TRAIN_TASKS_PER_PASS:
        raise ValueError("M4 RSFT corpus source task count drifted from the fixed budgeted roster")
    if manifest.get("source_task_universe_count") != M4_TRAIN_TASK_COUNT:
        raise ValueError("M4 RSFT corpus train universe does not match the frozen 240-task split")
    expected_coverage = RSFT_TRAIN_TASKS_PER_PASS / M4_TRAIN_TASK_COUNT
    coverage = manifest.get("source_task_coverage_fraction")
    if not isinstance(coverage, (int, float)) or abs(float(coverage) - expected_coverage) > 1e-12:
        raise ValueError("M4 RSFT corpus source task coverage is inconsistent with the fixed roster")
    if manifest.get("source_task_roster_sha256") != m4_task_roster_sha256(source_task_ids):
        raise ValueError("M4 RSFT corpus source task roster hash mismatch")
    if manifest.get("source_pass_action_token_cap") != ONLINE_PASS_ACTION_TOKEN_CAP:
        raise ValueError("M4 RSFT corpus pass cap drifted from the uniform generation budget")
    per_pass_tokens = manifest.get("source_collected_action_tokens_per_pass")
    if (
        not isinstance(per_pass_tokens, list)
        or len(per_pass_tokens) != OFFLINE_SUPERVISION_PASSES
        or any(not isinstance(value, int) or value < 0 or value > ONLINE_PASS_ACTION_TOKEN_CAP for value in per_pass_tokens)
        or manifest.get("source_total_collected_action_tokens") != sum(per_pass_tokens)
        or sum(per_pass_tokens) > COLLECTED_ACTION_TOKEN_CAP
    ):
        raise ValueError("M4 RSFT corpus generated action-token accounting is invalid")
    if manifest.get("source_roster_overlap_count") != RSFT_TRAIN_TASKS_PER_PASS:
        raise ValueError("M4 RSFT corpus passes must share the complete fixed source roster")
    overlap_fraction = manifest.get("source_roster_overlap_fraction")
    if not isinstance(overlap_fraction, (int, float)) or abs(float(overlap_fraction) - 1.0) > 1e-12:
        raise ValueError("M4 RSFT corpus pass-roster overlap must be exactly one")
    if manifest.get("source_pass_indices") != [1, 2]:
        raise ValueError("M4 RSFT corpus must preserve ordered source pass provenance")
    if (
        manifest.get("selected_task_count", 0)
        + manifest.get("unselected_task_count", 0)
        != manifest.get("source_task_count")
    ):
        raise ValueError("M4 RSFT selection counts do not cover the source task roster")
    selection_audit = manifest.get("selection_audit")
    if not isinstance(selection_audit, list) or len(selection_audit) != RSFT_TRAIN_TASKS_PER_PASS:
        raise ValueError("M4 RSFT corpus selection audit does not cover the fixed source roster")
    selection_task_ids = [item.get("task_id") for item in selection_audit if isinstance(item, dict)]
    if len(selection_task_ids) != len(selection_audit) or set(selection_task_ids) != set(source_task_ids):
        raise ValueError("M4 RSFT selection audit task IDs do not match the fixed source roster")
    if sum(item.get("selected_episode_id") is not None for item in selection_audit) != manifest.get("selected_task_count"):
        raise ValueError("M4 RSFT selection audit selected count is inconsistent")
    expected_hash = manifest.get("records_sha256")
    if not isinstance(expected_hash, str) or _sha256(data_dir / "train.jsonl") != expected_hash:
        raise ValueError("M4 RSFT corpus hash mismatch for train.jsonl")
    train_task_ids = _validate_jsonl_split(
        data_dir / "train.jsonl", expected_split="train", task_prefix="M4-TRAIN-"
    )
    if not train_task_ids.issubset(set(source_task_ids)):
        raise ValueError("M4 RSFT optimizer rows contain tasks outside the fixed source roster")
    return {
        "kind": "verified_rsft",
        "data_dir": str(data_dir),
        "train_samples": manifest.get("sample_count"),
        "selected_task_count": manifest.get("selected_task_count"),
        "unselected_task_count": manifest.get("unselected_task_count"),
        "records_sha256": expected_hash,
        "source_adapter_sha256": manifest["source_adapter_sha256"],
        "source_pass_indices": manifest.get("source_pass_indices"),
        "source_task_count": manifest["source_task_count"],
        "source_task_coverage_fraction": float(coverage),
        "source_total_collected_action_tokens": manifest["source_total_collected_action_tokens"],
        "selection_boundary": manifest.get("selection_boundary"),
    }


def build_m4_offline_training_plan(
    algorithm_id: str,
    seed: int,
    *,
    train_data_dir: Path,
    validation_data_dir: Path | None = None,
    task_root: Path | None = None,
    seed_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve an offline-control run without starting a GPU process.

    The plan binds the generic LoRA trainer to an immutable M4 manifest and
    denies test labels, dev optimizer labels, RSFT oracle fallback, or a larger
    supervised completion-token allowance than the online action-token budget.
    """
    config = M4RunConfig(algorithm_id, seed, "train")
    if config.algorithm.regime != "offline":
        raise ValueError(f"{algorithm_id} is not an M4 offline control")
    kwargs: dict[str, Path] = {}
    if task_root is not None:
        kwargs["task_root"] = Path(task_root)
    if seed_dir is not None:
        kwargs["seed_dir"] = Path(seed_dir)
    run_manifest = build_m4_run_manifest(config, **kwargs)
    train_data_dir = Path(train_data_dir).expanduser().resolve()
    validation_data_dir = (
        Path(validation_data_dir).expanduser().resolve()
        if validation_data_dir is not None
        else train_data_dir
    )
    if algorithm_id == "sft":
        train_data = _validate_sft_corpus(train_data_dir)
        validation_data = train_data if validation_data_dir == train_data_dir else _validate_sft_corpus(validation_data_dir)
    else:
        train_data = _validate_rsft_corpus(train_data_dir)
        validation_data = _validate_sft_corpus(validation_data_dir)
    return {
        "schema_version": "m4_offline_training_plan_v1",
        "run_manifest": run_manifest,
        "algorithm_id": algorithm_id,
        "study_seed": seed,
        "supervision_passes": OFFLINE_SUPERVISION_PASSES,
        "max_supervised_completion_tokens": COLLECTED_ACTION_TOKEN_CAP,
        "budget_semantics": (
            "completion-only supervised label-token upper bound; realized labels and zero-label "
            "truncation are audited separately from online generated action tokens"
        ),
        "train_data": train_data,
        "validation_data": validation_data,
        "selection_boundary": "valid.jsonl is evaluation/model-selection only; it is never optimizer input",
    }
