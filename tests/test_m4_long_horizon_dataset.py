from __future__ import annotations

import json
from collections import Counter
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


def test_long_task_correct_supplier_has_no_fixed_role_or_visit_position(tmp_path: Path):
    task_root, _, manifest = _build(tmp_path)
    assert manifest["shortcut_audit"] == {
        "contract": "correct long-task supplier must be balanced across visit positions 1,2,3",
        "neutral_feasible_supplier_roles": ["A", "B", "C"],
        "supplier_visit_roles": ["B", "A", "C"],
        "winner_assignment_version": "split_balanced_sha256_permutation_v1",
        "winner_role_counts": {
            "train": {"A": 20, "B": 20, "C": 20},
            "dev": {"A": 6, "B": 6, "C": 6},
            "test": {"A": 10, "B": 10, "C": 10},
        },
        "correct_supplier_visit_position_counts": {
            "train": {"1": 20, "2": 20, "3": 20},
            "dev": {"1": 6, "2": 6, "3": 6},
            "test": {"1": 10, "2": 10, "3": 10},
        },
        "balanced": True,
        "world_index_modulo3_agreement_counts": {
            "train": 17,
            "dev": 5,
            "test": 8,
        },
        "world_index_modulo3_agreement_fraction": {
            "train": 17 / 60,
            "dev": 5 / 18,
            "test": 8 / 30,
        },
        "maximum_allowed_modulo3_agreement_fraction": 0.5,
        "non_periodic": True,
    }

    train_oracle = [
        json.loads(line)
        for line in (task_root / "train" / "train_oracle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    positions = Counter()
    winning_roles = Counter()
    modulo_agreements = 0
    for row in train_oracle:
        if row["task_type"] != "highest_reliability_supplier":
            continue
        supplier_id = row["expected_product_id"].replace("PRD", "SUP", 1)
        required = row["workflow_requirements"]["required_supplier_detail_ids"]
        positions[required.index(supplier_id) + 1] += 1
        winning_roles[supplier_id.rsplit("-", 1)[-1]] += 1
        simple_role = ("A", "B", "C")[(int(row["world_id"][1:]) - 1) % 3]
        modulo_agreements += supplier_id.endswith(f"-{simple_role}")
    assert positions == Counter({1: 20, 2: 20, 3: 20})
    assert winning_roles == Counter({"A": 20, "B": 20, "C": 20})
    assert modulo_agreements == 17


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
