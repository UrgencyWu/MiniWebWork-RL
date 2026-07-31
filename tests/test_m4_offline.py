import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from miniwebwork.m4_offline import build_m4_offline_training_plan


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "data" / "tasks" / "m4_rlvr_v1"
SEED_DIR = ROOT / "data" / "seed_m4_rlvr_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sft_corpus(path: Path, train_task_id: str = "M4-TRAIN-W001-CHEAPEST_FEASIBLE") -> Path:
    path.mkdir()
    train = path / "train.jsonl"
    valid = path / "valid.jsonl"
    train_rows = [
        {"split": "train", "task_id": train_task_id}
    ] + [
        {"split": "train", "task_id": f"M4-TRAIN-W{world:03d}-CHEAPEST_FEASIBLE"}
        for world in range(2, 241)
    ]
    valid_rows = [
        {"split": "dev", "task_id": f"M4-DEV-W{world:03d}-CHEAPEST_FEASIBLE"}
        for world in range(61, 133)
    ]
    train.write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
    valid.write_text("".join(json.dumps(row) + "\n" for row in valid_rows), encoding="utf-8")
    manifest = {
        "train": {
            "dataset_id": "m4_oracle_sft_v1",
            "task_source_dataset_id": "m4_rlvr_v1",
            "task_split": "train",
            "purpose": "offline_training",
            "sample_split": "train",
            "task_count": 240,
            "sample_count": len(train_rows),
        },
        "dev": {
            "dataset_id": "m4_oracle_sft_v1",
            "task_source_dataset_id": "m4_rlvr_v1",
            "task_split": "dev",
            "purpose": "model_selection",
            "sample_split": "dev",
            "task_count": 72,
            "sample_count": len(valid_rows),
        },
        "train_sha256": _sha256(train),
        "valid_sha256": _sha256(valid),
        "selection_boundary": "dev examples are evaluation/model-selection only, never optimizer samples",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _load_runner_module():
    path = ROOT / "scripts" / "m4_train_offline.py"
    spec = importlib.util.spec_from_file_location("m4_train_offline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sft_offline_plan_binds_two_passes_and_cap_to_the_m4_manifest(tmp_path: Path):
    data_dir = _write_sft_corpus(tmp_path / "sft")
    plan = build_m4_offline_training_plan(
        "sft",
        20260801,
        train_data_dir=data_dir,
        task_root=TASK_ROOT,
        seed_dir=SEED_DIR,
    )

    assert plan["supervision_passes"] == 2
    assert plan["max_supervised_completion_tokens"] == 250_000
    assert plan["train_data"]["kind"] == "oracle_sft"
    assert plan["run_manifest"]["resolved_split"] == "train"

    runner = _load_runner_module()
    command = runner._trainer_command(
        plan,
        output_dir=tmp_path / "run",
        initial_adapter=tmp_path / "adapter",
        base_model="model",
        max_length=8192,
        learning_rate=2e-4,
        batch_size=1,
        grad_accum=16,
    )
    assert command[command.index("--epochs") + 1] == "2"
    assert command[command.index("--initial-adapter") + 1].endswith("adapter")
    assert command[command.index("--max-supervised-completion-tokens") + 1] == "250000"


def test_sft_offline_plan_rejects_test_task_rows_even_when_hashes_match(tmp_path: Path):
    data_dir = _write_sft_corpus(tmp_path / "bad", train_task_id="M4-TEST-W091-CHEAPEST_FEASIBLE")
    with pytest.raises(PermissionError, match="M4-TRAIN-"):
        build_m4_offline_training_plan(
            "sft",
            20260801,
            train_data_dir=data_dir,
            task_root=TASK_ROOT,
            seed_dir=SEED_DIR,
        )
