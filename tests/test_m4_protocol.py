import json
from pathlib import Path

import pytest

from miniwebwork.m4_protocol import (
    M4RunConfig,
    STUDY_SEEDS,
    build_m4_run_manifest,
    write_m4_run_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "data" / "tasks" / "m4_rlvr_v1"
SEED_DIR = ROOT / "data" / "seed_m4_rlvr_v1"


def test_primary_matrix_configs_are_preflightable_without_test_leakage():
    for algorithm in ("sft", "rsft", "rloo", "grpo", "gspo"):
        for seed in STUDY_SEEDS:
            config = M4RunConfig(algorithm, seed, "train")
            manifest = build_m4_run_manifest(config, task_root=TASK_ROOT, seed_dir=SEED_DIR)
            assert manifest["config"]["seed"] == seed
            assert manifest["resolved_split"] == "train"
            assert manifest["hashes"]["train_public.jsonl"]
            if algorithm in {"sft", "rsft"}:
                assert "offline_training" in manifest["split_manifest"]["allowed_purposes"]


def test_protocol_rejects_unregistered_seed_and_wrong_online_collection_contract():
    with pytest.raises(ValueError, match="seed must be one of"):
        M4RunConfig("grpo", 7, "train").validate(task_root=TASK_ROOT)
    with pytest.raises(ValueError, match="group_size is fixed"):
        M4RunConfig("gspo", STUDY_SEEDS[0], "train", group_size=8).validate(
            task_root=TASK_ROOT
        )


def test_final_test_manifest_is_evaluation_only_and_write_is_idempotent(tmp_path: Path):
    manifest = build_m4_run_manifest(
        M4RunConfig("grpo", STUDY_SEEDS[0], "final_test"),
        task_root=TASK_ROOT,
        seed_dir=SEED_DIR,
    )
    assert manifest["resolved_split"] == "test"
    assert manifest["split_manifest"]["may_update_model"] is False

    output = tmp_path / "run_manifest.json"
    write_m4_run_manifest(output, manifest)
    write_m4_run_manifest(output, manifest)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == manifest
    changed = {**manifest, "git_sha": "different"}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_m4_run_manifest(output, changed)
