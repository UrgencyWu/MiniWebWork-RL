from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import pytest

from miniwebwork.data_generation.m4_long_horizon import (
    DATASET_ID,
    DATASET_MANIFEST_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    SPLIT_WORLD_COUNTS,
    assert_long_horizon_split_purpose,
    build_long_horizon_dataset,
    validate_long_horizon_dataset,
)
from miniwebwork.db import get_connection, init_schema
from miniwebwork.seed import seed_database
from miniwebwork.tasks import validate_tasks


def _build(tmp_path: Path):
    task_root = tmp_path / "tasks"
    seed_root = tmp_path / "seed"
    manifest = build_long_horizon_dataset(task_root, seed_dir=seed_root)
    return task_root, seed_root, manifest


def test_long_horizon_dataset_has_focused_counts_and_real_long_stratum(tmp_path: Path):
    task_root, seed_root, manifest = _build(tmp_path)
    assert manifest["dataset_id"] == DATASET_ID
    assert manifest["task_count"] == 432
    assert manifest["split_task_counts"] == {"train": 240, "dev": 72, "test": 120}
    assert manifest["horizon_audit"]["splits"]["train"]["horizon_counts"] == {
        "basic": 60,
        "medium": 120,
        "long": 60,
    }
    assert manifest["horizon_audit"]["splits"]["dev"]["horizon_counts"] == {
        "basic": 18,
        "medium": 36,
        "long": 18,
    }
    assert manifest["horizon_audit"]["splits"]["test"]["horizon_counts"] == {
        "basic": 30,
        "medium": 60,
        "long": 30,
    }
    validation = validate_long_horizon_dataset(task_root, seed_dir=seed_root)
    assert validation["valid"], validation["errors"]


def test_reference_traces_are_7_10_12_18_and_long_requires_three_supplier_visits(tmp_path: Path):
    task_root, _, _ = _build(tmp_path)
    oracle = [
        json.loads(line)
        for line in (task_root / "train" / "train_oracle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    first_world = [row for row in oracle if row["world_id"] == "W001"]
    horizons = {row["task_type"]: row["oracle_min_env_actions"] for row in first_world}
    assert horizons == {
        "exact_product": 7,
        "cheapest_feasible": 12,
        "highest_reliability_supplier": 18,
        "no_feasible_product": 10,
    }
    long_task = next(
        row for row in first_world if row["task_type"] == "highest_reliability_supplier"
    )
    required = long_task["workflow_requirements"]["required_supplier_detail_ids"]
    assert len(required) == 3
    assert len(set(required)) == 3
    assert sum(
        step.get("target_testid", "").startswith("supplier-link-")
        for step in long_task["reference_trace"]
    ) == 3


def test_supplier_and_world_ids_are_disjoint_across_splits(tmp_path: Path):
    task_root, _, _ = _build(tmp_path)
    split_specs = {}
    all_specs = [
        json.loads(line)
        for line in (task_root / "spec.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for split in SPLIT_WORLD_COUNTS:
        rows = [row for row in all_specs if row["split"] == split]
        split_specs[split] = {
            "world": {row["world_signature"] for row in rows},
            "product": {row["product_signature"] for row in rows},
            "supplier": {row["supplier_signature"] for row in rows},
            "constraint": {row["constraint_signature"] for row in rows},
            "answer": {row["answer_signature"] for row in rows},
        }
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for field in split_specs[first]:
            assert split_specs[first][field].isdisjoint(split_specs[second][field])


def test_split_purpose_and_database_oracles_fail_closed(tmp_path: Path):
    task_root, seed_root, _ = _build(tmp_path)
    assert_long_horizon_split_purpose(task_root / "train", "online_training")
    assert_long_horizon_split_purpose(task_root / "dev", "model_selection")
    assert_long_horizon_split_purpose(task_root / "test", "final_evaluation")
    with pytest.raises(PermissionError):
        assert_long_horizon_split_purpose(task_root / "test", "online_training")

    db_path = tmp_path / "long-horizon.db"
    with closing(get_connection(str(db_path))) as connection:
        init_schema(connection)
        seed_database(connection, seed_dir=seed_root)
        for split, worlds in SPLIT_WORLD_COUNTS.items():
            result = validate_tasks(connection, task_dir=task_root / split)
            assert result["valid"], (split, result["errors"])
            assert result["public_count"] == worlds * 4


def test_checked_files_bind_public_oracle_and_seed_hashes(tmp_path: Path):
    task_root, _, manifest = _build(tmp_path)
    checked = json.loads(
        (task_root / DATASET_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert checked == manifest
    for split in SPLIT_WORLD_COUNTS:
        split_manifest = json.loads(
            (task_root / split / SPLIT_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert split_manifest["task_count"] == SPLIT_WORLD_COUNTS[split] * 4
        assert split_manifest["medium_long_fraction"] == 0.75
