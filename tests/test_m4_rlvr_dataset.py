"""Regression tests for the audited M4 train/dev/final-test world contract."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from pathlib import Path

import pytest

from miniwebwork.data_generation.m4_rlvr import (
    DATASET_ID,
    DATASET_MANIFEST_FILENAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED_DIR,
    SPLIT_MANIFEST_FILENAME,
    SPLIT_WORLD_COUNTS,
    assert_m4_split_purpose,
    build_m4_rlvr_dataset,
    validate_m4_rlvr_dataset,
)
from miniwebwork.db import get_connection, init_schema
from miniwebwork.seed import seed_database, validate_seed
from miniwebwork.tasks import validate_tasks


ROOT = Path(__file__).resolve().parents[1]
CHECKED_DATASET_DIR = ROOT / "data" / "tasks" / "m4_rlvr_v1"
CHECKED_SEED_DIR = ROOT / "data" / "seed_m4_rlvr_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checked_m4_dataset_matches_the_deterministic_builder(tmp_path: Path):
    generated_dataset = tmp_path / "tasks" / "m4_rlvr_v1"
    generated_seed = tmp_path / "seed_m4_rlvr_v1"
    generated_manifest = build_m4_rlvr_dataset(
        generated_dataset,
        seed_dir=generated_seed,
    )

    assert generated_manifest == json.loads(
        (CHECKED_DATASET_DIR / DATASET_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    for root, generated_root in (
        (CHECKED_DATASET_DIR, generated_dataset),
        (CHECKED_SEED_DIR, generated_seed),
    ):
        checked_files = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
        generated_files = sorted(
            path.relative_to(generated_root)
            for path in generated_root.rglob("*")
            if path.is_file()
        )
        assert generated_files == checked_files
        for relative in checked_files:
            assert (generated_root / relative).read_bytes() == (root / relative).read_bytes()


def test_m4_manifest_counts_roles_and_freeze_gate():
    manifest = json.loads(
        (CHECKED_DATASET_DIR / DATASET_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["dataset_id"] == DATASET_ID
    assert manifest["world_count"] == 108
    assert manifest["task_count"] == 432
    assert manifest["split_task_counts"] == {"train": 240, "dev": 72, "test": 120}
    assert manifest["isolation_contract"] == {
        "worlds_disjoint_across_splits": True,
        "constraint_signatures_disjoint_across_splits": True,
        "selected_product_answers_disjoint_across_splits": True,
        "public_instructions_disjoint_across_splits": True,
        "test_role": "frozen_final_evaluation",
    }

    train = assert_m4_split_purpose(CHECKED_DATASET_DIR / "train", "online_training")
    dev = assert_m4_split_purpose(CHECKED_DATASET_DIR / "dev", "model_selection")
    test = assert_m4_split_purpose(CHECKED_DATASET_DIR / "test", "final_evaluation")
    assert train["may_update_model"] is True
    assert dev["may_update_model"] is False
    assert test["may_update_model"] is False
    with pytest.raises(PermissionError, match="cannot be used for 'online_training'"):
        assert_m4_split_purpose(CHECKED_DATASET_DIR / "test", "online_training")


def test_m4_dataset_recomputes_against_its_own_versioned_seed(tmp_path: Path):
    assert validate_seed(CHECKED_SEED_DIR)["valid"]
    assert validate_m4_rlvr_dataset(CHECKED_DATASET_DIR, seed_dir=CHECKED_SEED_DIR)["valid"]

    db_path = tmp_path / "m4.db"
    with closing(get_connection(str(db_path))) as connection:
        init_schema(connection)
        seed_database(connection, seed_dir=CHECKED_SEED_DIR)
        counts = connection.execute("SELECT COUNT(*) AS count FROM products").fetchone()
        assert counts["count"] == 432
        for split, worlds in SPLIT_WORLD_COUNTS.items():
            result = validate_tasks(connection, task_dir=CHECKED_DATASET_DIR / split)
            assert result["valid"], (split, result["errors"])
            assert result["public_count"] == worlds * 4


def test_m4_split_hashes_cover_the_checked_payloads():
    for split in SPLIT_WORLD_COUNTS:
        split_dir = CHECKED_DATASET_DIR / split
        manifest = json.loads(
            (split_dir / SPLIT_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert manifest["public_sha256"] == _sha256(split_dir / f"{split}_public.jsonl")
        assert manifest["oracle_sha256"] == _sha256(split_dir / f"{split}_oracle.jsonl")
