from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/m6_monotonic_posttraining_v1"
D35_CORPUS = STUDY / "phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v1/replay_weighted_corpus_v1"


def _module():
    name = "m6_phase10d_train_s4_d35"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/m6_phase10d_train_s4_d35.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_s4_d35_keeps_the_frozen_4b_paradigm_config():
    module = _module()
    assert module.BASE_MODEL == Path("/data/share/model/Qwen3.5-4B")
    assert module.RAW35_PRODUCER_MODEL == "/data/share/model/Qwen3.5-35B-A3B"
    assert module.SEED == 20260866
    assert module.LEARNING_RATE == 2e-5
    assert module.RAW_REFERENCE_KL == 0.03
    assert module.IMITATION_PER_UPDATE == 9
    assert module.RETENTION_PER_UPDATE == 1
    assert module.MAXIMUM_EPOCHS == 1
    assert module.LORA_CONFIG == {"r": 16, "alpha": 32, "dropout": 0.05}
    assert module.CAPABILITY_WEIGHTS == {"nav": 0.20, "match": 0.40, "finish": 0.40}


def test_resolve_4b_targets_uses_sft4_suffixes_and_excludes_mtp():
    module = _module()
    names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mtp.q_proj",
        "mtp.layers.0.q_proj",
        "model.layers.0.self_attn.rotary_emb",
    ]
    assert module.resolve_4b_targets(names) == (
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.q_proj",
    )
    with pytest.raises(ValueError, match="no LoRA targets"):
        module.resolve_4b_targets(["model.layers.0.self_attn.rotary_emb"])


def test_frozen_d35_corpus_loads_with_expected_bindings():
    module = _module()
    train_rows, train_binding = module.load_d35_rows(D35_CORPUS, "train")
    dev_rows, dev_binding = module.load_d35_rows(D35_CORPUS, "dev")
    assert len(train_rows) == 975
    assert len(dev_rows) == 109
    assert train_binding["manifest_file_sha256"] == module.FROZEN_D35["manifest_sha256"]
    assert train_binding["jsonl_sha256"] == module.FROZEN_D35["train_sha256"]
    assert dev_binding["jsonl_sha256"] == module.FROZEN_D35["dev_sha256"]
    assert {row["capability"] for row in train_rows} <= set(module.CAPABILITY_WEIGHTS)


def test_one_epoch_schedule_covers_975_rows_in_108_updates():
    module = _module()
    schedule = module.paradigm.build_update_schedule(975, 975, seed=module.SEED)
    used = [index for item in schedule for index in item["imitation_indices"]]
    assert len(schedule) == 108
    assert len(used) == 972
    assert len(set(used)) == 972
    assert all(len(item["imitation_indices"]) == 9 for item in schedule)


def test_wrapper_freezes_gpu1_probe_default_and_4b_base():
    source = (ROOT / "scripts/run_m6_phase10d_s4_d35_job.sh").read_text()
    assert "#SBATCH --gres=gpu:1" in source
    assert "M6_PHASE10D_S4D35_MODE:-probe" in source
    assert "--base-model /data/share/model/Qwen3.5-4B" in source
    assert "replay_weighted_corpus_v1" in source
    assert "m6_phase10d_train_s4_d35.py" in source
