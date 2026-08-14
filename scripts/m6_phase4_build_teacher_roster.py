#!/usr/bin/env python3
"""Freeze a stratified 16-task all-failure roster for the Phase4 teacher probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m6_phase4_build_data_rosters import _proportional_quotas  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import publish_immutable_json, sha256_json  # noqa: E402
from miniwebwork.long_horizon_rl.model_manifest import (  # noqa: E402
    build_base_model_manifest,
    validate_base_model_manifest,
)
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402

ALLOWED_TEACHER_MODELS = {
    Path("/data/share/model/Qwen3.5-9B").resolve(),
    Path("/data/share/model/Qwen3.6-35B-A3B-FP8").resolve(),
}
DEFAULT_TEACHER_MODEL = Path("/data/share/model/Qwen3.5-9B")
PROBE_TASK_COUNT = 16
PROBE_SEED = 20260826
MAXIMUM_BUCKET_SHARE_GAP = 0.08


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_hashed(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    _require(value.get("content_sha256") == _self_hash(value), f"M6 Phase4 teacher source hash drift: {path}")
    if schema is not None:
        _require(value.get("schema_version") == schema, f"M6 Phase4 teacher source schema drift: {path}")
    return value


def _dominant_failure_class(row: Mapping[str, Any]) -> str:
    counts = row.get("failure_classes")
    _require(isinstance(counts, Mapping) and counts, "M6 Phase4 teacher candidate lacks failure classes")
    return min(
        (str(name) for name in counts),
        key=lambda name: (-int(counts[name]), name),
    )


def _selection_bucket(row: Mapping[str, Any]) -> str:
    return f"{row['bucket']}|dominant_{_dominant_failure_class(row)}"


def _rank(namespace: str, task_id: str) -> bytes:
    return hashlib.sha256(f"{namespace}|{PROBE_SEED}|{task_id}".encode()).digest()


def _select(rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], float]:
    _require(len(rows) >= PROBE_TASK_COUNT, "M6 Phase4 teacher candidate population is too small")
    by_bucket: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_bucket.setdefault(_selection_bucket(row), []).append(row)
    population_counts = {name: len(values) for name, values in by_bucket.items()}
    quotas = _proportional_quotas(population_counts, PROBE_TASK_COUNT)
    selected: list[dict[str, Any]] = []
    for bucket, values in by_bucket.items():
        ranked = sorted(values, key=lambda row: (_rank("m6-phase4-teacher-select-v1", str(row["task_id"])), str(row["task_id"])))
        selected.extend(dict(row) for row in ranked[: quotas[bucket]])
    selected.sort(key=lambda row: (_rank("m6-phase4-teacher-order-v1", str(row["task_id"])), str(row["task_id"])))
    selected_counts = Counter(_selection_bucket(row) for row in selected)
    maximum_gap = max(
        abs(selected_counts.get(bucket, 0) / PROBE_TASK_COUNT - count / len(rows))
        for bucket, count in population_counts.items()
    )
    _require(len(selected) == PROBE_TASK_COUNT, "M6 Phase4 teacher selection size drift")
    _require(len({row["task_id"] for row in selected}) == PROBE_TASK_COUNT, "M6 Phase4 teacher tasks duplicated")
    _require(len(selected_counts) >= 4, "M6 Phase4 teacher probe lacks failure-stratum coverage")
    _require(maximum_gap <= MAXIMUM_BUCKET_SHARE_GAP, "M6 Phase4 teacher bucket share gap exceeded")
    return selected, population_counts, dict(selected_counts), maximum_gap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-audit", type=Path, required=True)
    parser.add_argument("--teacher-candidates", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    args = parser.parse_args()

    teacher_model = args.teacher_model.expanduser().resolve()
    _require(teacher_model in ALLOWED_TEACHER_MODELS, "M6 Phase4 teacher model is not a frozen candidate")

    protocol = load_protocol()
    split = validate_split_lock(json.loads(args.split_lock.read_text(encoding="utf-8")))
    _require(split["protocol_sha256"] == protocol["sha256"], "M6 Phase4 teacher split/protocol drift")
    student_audit = _load_hashed(args.student_audit, schema="m6_phase4_student_prescan_audit_v1")
    candidates = _load_hashed(args.teacher_candidates, schema="m6_phase4_teacher_candidates_v1")
    _require(student_audit["decision"]["student_prescan_integrity_passed"] is True, "M6 Phase4 student prescan integrity failed")
    _require(student_audit["decision"]["teacher_supplement_required"] is True, "M6 Phase4 teacher supplement was not triggered")
    _require(student_audit["decision"]["future_online_mixed_task_roster_ready"] is False, "M6 Phase4 teacher probe cannot bypass a ready student roster")
    _require(candidates["teacher_trajectories_allowed_for_on_policy_grpo"] is False, "M6 Phase4 teacher/on-policy separation drift")
    rows = candidates.get("rows")
    _require(isinstance(rows, list) and len(rows) == student_audit["teacher_candidate_task_count"], "M6 Phase4 teacher candidate count drift")
    _require(all(row.get("content_sha256") == _self_hash(row) for row in rows), "M6 Phase4 teacher candidate row hash drift")
    _require(all(int(row.get("student_strict_success_count", -1)) == 0 for row in rows), "M6 Phase4 teacher roster includes a student success")
    selected, population_counts, selected_counts, maximum_gap = _select(rows)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "teacher_base_model_manifest.json"
    if manifest_path.is_file():
        model_manifest = validate_base_model_manifest(
            path=manifest_path,
            expected_base_model=teacher_model,
            verify_files=True,
        )
    else:
        model_manifest = build_base_model_manifest(
            base_model=teacher_model,
            destination=manifest_path,
        )
        model_manifest = validate_base_model_manifest(
            path=manifest_path,
            expected_base_model=teacher_model,
            verify_files=True,
        )

    task_ids = [str(row["task_id"]) for row in selected]
    train_ids = set(split["roles"]["train"]["task_ids"])
    _require(set(task_ids) <= train_ids, "M6 Phase4 teacher roster escaped train role")
    roster = {
        "schema_version": "m6_rl_curriculum_v1",
        "study_id": protocol["payload"]["study_id"],
        "development_only": True,
        "formal_training": False,
        "purpose": "phase4_larger_model_teacher_probe_on_student_all_failure_tasks",
        "selection": "student_all_failure_category_constraint_failure_stratified_hash_v1",
        "selection_seed": PROBE_SEED,
        "source_split_lock_content_sha256": split["content_sha256"],
        "protocol_sha256": protocol["sha256"],
        "git_sha": protocol["git_sha"],
        "source_role": "train",
        "source_student_audit_content_sha256": student_audit["content_sha256"],
        "source_teacher_candidates_content_sha256": candidates["content_sha256"],
        "teacher_model": str(teacher_model),
        "teacher_base_model_manifest_path": str(manifest_path),
        "teacher_base_model_manifest_sha256": model_manifest["sha256"],
        "teacher_base_model_functional_sha256": model_manifest["payload"]["functional_file_set_sha256"],
        "student_strict_success_count_on_selected_tasks": 0,
        "candidate_task_count": len(rows),
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "selected_candidate_content_sha256": [row["content_sha256"] for row in selected],
        "population_bucket_counts": dict(sorted(population_counts.items())),
        "selected_bucket_counts": dict(sorted(selected_counts.items())),
        "maximum_bucket_share_gap": maximum_gap,
        "maximum_allowed_bucket_share_gap": MAXIMUM_BUCKET_SHARE_GAP,
        "task_order_sha256": sha256_json(task_ids),
        "probe_K": 4,
        "max_model_turns": 18,
        "max_environment_steps": 15,
        "minimum_verified_strict_success_tasks": 4,
        "minimum_verified_distinct_strict_success_trajectories": 8,
        "maximum_supplement_trajectories_per_task": 2,
        "teacher_data_allowed_for_on_policy_grpo": False,
    }
    roster["content_sha256"] = _self_hash(roster)
    publish_immutable_json(output / "teacher_roster.json", roster)
    print(json.dumps(roster, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
