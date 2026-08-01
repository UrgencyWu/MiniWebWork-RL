import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from miniwebwork.m4_offline import build_m4_offline_training_plan
from miniwebwork.m4_protocol import RSFT_TRAIN_TASKS_PER_PASS, m4_task_roster_sha256


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
            "dataset_id": "m4_oracle_sft_v2",
            "task_source_dataset_id": "m4_rlvr_v1",
            "task_split": "train",
            "purpose": "offline_training",
            "sample_split": "train",
            "task_count": 240,
            "sample_count": len(train_rows),
            "prompt_contract": "browser_agent_v3_compact",
        },
        "dev": {
            "dataset_id": "m4_oracle_sft_v2",
            "task_source_dataset_id": "m4_rlvr_v1",
            "task_split": "dev",
            "purpose": "model_selection",
            "sample_split": "dev",
            "task_count": 72,
            "sample_count": len(valid_rows),
            "prompt_contract": "browser_agent_v3_compact",
        },
        "train_sha256": _sha256(train),
        "valid_sha256": _sha256(valid),
        "prompt_contract": "browser_agent_v3_compact",
        "prompt_system_sha256": "prompt-hash",
        "context_contract": {"prompt_version": "browser_agent_v3_compact"},
        "selection_boundary": "dev examples are evaluation/model-selection only, never optimizer samples",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_rsft_corpus(path: Path) -> Path:
    path.mkdir()
    source_task_ids = [
        f"M4-TRAIN-W{world:03d}-CHEAPEST_FEASIBLE"
        for world in range(1, RSFT_TRAIN_TASKS_PER_PASS + 1)
    ]
    train = path / "train.jsonl"
    rows = [{"split": "train", "task_id": source_task_ids[0]}]
    train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selection_audit = [
        {
            "task_id": task_id,
            "selected_episode_id": "selected" if index == 0 else None,
        }
        for index, task_id in enumerate(source_task_ids)
    ]
    manifest = {
        "dataset_id": "m4_rsft_tokenized_v1",
        "algorithm": "rsft",
        "split": "train",
        "source_passes": 2,
        "source_adapter_sha256": "initial-adapter",
        "source_pass_indices": [1, 2],
        "source_task_universe_count": 240,
        "source_task_count": RSFT_TRAIN_TASKS_PER_PASS,
        "source_task_ids": source_task_ids,
        "source_task_roster_sha256": m4_task_roster_sha256(source_task_ids),
        "source_task_coverage_fraction": RSFT_TRAIN_TASKS_PER_PASS / 240,
        "source_pass_action_token_cap": 125_000,
        "source_collected_action_tokens_per_pass": [100, 100],
        "source_total_collected_action_tokens": 200,
        "source_roster_overlap_count": RSFT_TRAIN_TASKS_PER_PASS,
        "source_roster_overlap_fraction": 1.0,
        "selected_task_count": 1,
        "unselected_task_count": RSFT_TRAIN_TASKS_PER_PASS - 1,
        "sample_count": len(rows),
        "records_sha256": _sha256(train),
        "selection_audit": selection_audit,
        "selection_boundary": "verified best-of-n train-only selection",
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


def test_sft_offline_plan_binds_effective_label_target_to_the_m4_manifest(tmp_path: Path):
    data_dir = _write_sft_corpus(tmp_path / "sft")
    plan = build_m4_offline_training_plan(
        "sft",
        20260801,
        train_data_dir=data_dir,
        task_root=TASK_ROOT,
        seed_dir=SEED_DIR,
    )

    assert plan["supervision_passes"] == 1
    assert plan["max_supervised_completion_tokens"] == 250_000
    assert plan["train_data"]["kind"] == "oracle_sft"
    assert plan["run_manifest"]["resolved_split"] == "train"

    runner = _load_runner_module()
    command = runner._trainer_command(
        plan,
        output_dir=tmp_path / "run",
        initial_adapter=tmp_path / "adapter",
        base_model="model",
        max_length=6144,
        learning_rate=2e-4,
        batch_size=1,
        grad_accum=16,
    )
    assert command[command.index("--epochs") + 1] == "1"
    assert command[command.index("--initial-adapter") + 1].endswith("adapter")
    assert command[command.index("--max-supervised-completion-tokens") + 1] == "250000"
    assert command[command.index("--target-supervised-completion-tokens") + 1] == "250000"
    assert command[command.index("--budget-selection-seed") + 1] == "20260801"
    assert command[command.index("--max-zero-completion-label-fraction") + 1] == "0.0"


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


def test_rsft_offline_plan_accepts_only_the_fixed_12_task_generation_roster(tmp_path: Path):
    rsft_data = _write_rsft_corpus(tmp_path / "rsft")
    validation_data = _write_sft_corpus(tmp_path / "validation")
    plan = build_m4_offline_training_plan(
        "rsft",
        20260801,
        train_data_dir=rsft_data,
        validation_data_dir=validation_data,
        task_root=TASK_ROOT,
        seed_dir=SEED_DIR,
    )

    assert plan["train_data"]["kind"] == "verified_rsft"
    assert plan["train_data"]["source_task_count"] == 12
    assert plan["train_data"]["source_task_coverage_fraction"] == 0.05
    assert plan["train_data"]["source_total_collected_action_tokens"] == 200


def test_rsft_offline_plan_rejects_legacy_full_roster_claim_under_fixed_cap(tmp_path: Path):
    rsft_data = _write_rsft_corpus(tmp_path / "bad_rsft")
    manifest_path = rsft_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_task_count"] = 240
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source task count drifted"):
        build_m4_offline_training_plan(
            "rsft",
            20260801,
            train_data_dir=rsft_data,
            validation_data_dir=_write_sft_corpus(tmp_path / "validation"),
            task_root=TASK_ROOT,
            seed_dir=SEED_DIR,
        )
