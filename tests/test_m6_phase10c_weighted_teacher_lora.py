from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _module():
    name = "m6_phase10c_train_weighted_teacher_lora"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/m6_phase10c_train_weighted_teacher_lora.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True
        assert enable_thinking is False
        values = [1, 2]
        prompt_messages = messages if add_generation_prompt else messages[:-1]
        for message in prompt_messages:
            values.extend([3, len(str(message["content"])) + 10])
        values.append(20)
        if not add_generation_prompt:
            values.extend([30, len(str(messages[-1]["content"])) + 40, 21])
        return values


def _row(capability: str, task: str, trajectory: str, weight: float) -> dict:
    return {
        "messages": [{"role": "user", "content": "state"}],
        "completion": '{"command":"click[item]"}',
        "task_id": task,
        "trajectory_id": trajectory,
        "turn_index": 1,
        "capability": capability,
        "row_loss_weight": weight,
    }


def test_tokenization_masks_every_prompt_token():
    module = _module()
    value = module.tokenize_weighted_row(_row("nav", "task-a", "traj-a", 0.2), FakeTokenizer())
    assert all(label == -100 for label in value.tokenized.labels[: value.tokenized.prompt_tokens])
    assert value.tokenized.completion_label_tokens > 0
    assert value.tokenized.input_ids[value.tokenized.prompt_tokens :] == value.tokenized.labels[value.tokenized.prompt_tokens :]


def test_probe_selection_realizes_capability_mass_with_unique_tasks():
    module = _module()
    examples = []
    for capability, count in (("nav", 3), ("match", 4), ("finish", 4)):
        for index in range(count):
            examples.append(module.tokenize_weighted_row(
                _row(capability, f"{capability}-{index}", f"traj-{capability}-{index}", 1 / 11),
                FakeTokenizer(),
            ))
    selected = module.select_probe_examples(examples)
    assert len(selected) == 5
    assert len({item.tokenized.task_id for item in selected}) == 5
    assert math.isclose(sum(item.row_weight for item in selected), 1.0)
    assert {
        capability: sum(item.row_weight for item in selected if item.capability == capability)
        for capability in module.CAPABILITY_WEIGHTS
    } == module.CAPABILITY_WEIGHTS


class FakeTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = torch.nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, torch.nn.Linear(2, 2, bias=False))
        layer.linear_attn = torch.nn.Module()
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            setattr(layer.linear_attn, name, torch.nn.Linear(2, 2, bias=False))
        self.visual = torch.nn.Module()
        self.visual.out_proj = torch.nn.Linear(2, 2, bias=False)


def test_target_resolution_covers_text_mixers_and_excludes_vision():
    module = _module()
    targets = module.resolve_text_target_modules(FakeTextModel())
    assert len(targets) == 9
    assert all("visual" not in name for name in targets)
    assert {name.rsplit(".", 1)[-1] for name in targets} == set(module.TEXT_TARGET_SUFFIXES)


def test_weighted_manifest_and_rows_are_fail_closed(tmp_path: Path):
    module = _module()
    root = tmp_path / "corpus"
    root.mkdir()
    rows = [
        {**_row("nav", "n", "tn", 0.2), "source": "raw35_replay_strict_self_exploration", "loss_hierarchy": "capability_task_path_action_row_token_mean_v1"},
        {**_row("match", "m", "tm", 0.4), "source": "raw35_replay_strict_self_exploration", "loss_hierarchy": "capability_task_path_action_row_token_mean_v1"},
        {**_row("finish", "f", "tf", 0.4), "source": "raw35_replay_strict_self_exploration", "loss_hierarchy": "capability_task_path_action_row_token_mean_v1"},
    ]
    (root / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "schema_version": "m6_phase10c_weighted_raw35_sft_corpus_v1",
        "base_model": str(module.BASE_MODEL),
        "capability_weights": module.CAPABILITY_WEIGHTS,
        "loss_hierarchy": "capability_task_path_action_row_token_mean_v1",
        "task_overlap": 0,
        "train_action_row_count": 3,
        "train_task_count": 3,
    }
    manifest["content_sha256"] = module._self_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    loaded, _ = module.load_and_validate_rows(root, "train")
    assert len(loaded) == 3
    rows[0]["row_loss_weight"] = 0.3
    (root / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="loss mass"):
        module.load_and_validate_rows(root, "train")


def test_wrappers_freeze_probe_and_formal_resources():
    probe = (ROOT / "scripts/run_m6_phase10c_teacher_sft_probe_job.sh").read_text()
    train = (ROOT / "scripts/run_m6_phase10c_teacher_sft_train_job.sh").read_text()
    for value in (probe, train):
        assert "#SBATCH --gres=gpu:2" in value
        assert "#SBATCH --mem=96G" in value
        assert "/data/share/model/Qwen3.5-35B-A3B" in value
        assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in value
    assert "--mode probe" in probe
    assert "--mode train" in train


def test_two_gpu_loader_reserves_more_than_half_each_device():
    module = _module()
    source = (ROOT / "scripts/m6_phase10c_train_weighted_teacher_lora.py").read_text()
    assert "total_memory * 0.45" in source
    assert "torch.cuda.device_count() == 2" in source
    assert source.index("load_raw35_lora(BASE_MODEL)") < source.index("torch.cuda.reset_peak_memory_stats(index)")
