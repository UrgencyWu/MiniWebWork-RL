from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _module():
    name = "m6_phase10c_train_4b_paradigm_teacher_lora"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/m6_phase10c_train_4b_paradigm_teacher_lora.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_formal_schedule_matches_4b_9_to_1_one_epoch_contract():
    module = _module()
    schedule = module.build_update_schedule(975, 975, seed=module.SEED)
    used = [index for item in schedule for index in item["imitation_indices"]]
    assert len(schedule) == 108
    assert len(used) == 972
    assert len(set(used)) == 972
    assert all(len(item["imitation_indices"]) == 9 for item in schedule)
    assert all(0 <= item["retention_index"] < 975 for item in schedule)


def test_probe_schedule_has_two_real_optimizer_updates():
    module = _module()
    schedule = module.build_update_schedule(18, 18, seed=module.SEED)
    assert len(schedule) == 2
    assert {index for item in schedule for index in item["imitation_indices"]} == set(range(18))


def test_used_epoch_weights_are_exactly_20_40_40():
    module = _module()
    examples = []
    for capability, count in (("nav", 10), ("match", 11), ("finish", 12)):
        for index in range(count):
            tokenized = type("T", (), {"sample_id": f"{capability}-{index}"})()
            examples.append(module.weighted.WeightedExample(tokenized, capability, f"t-{index}", module.CAPABILITY_WEIGHTS[capability] / count))
    schedule = module.build_update_schedule(len(examples), len(examples), seed=module.SEED)
    normalized = module.renormalize_used_weights(examples, schedule)
    used = {index for item in schedule for index in item["imitation_indices"]}
    observed = {
        capability: sum(normalized[index].row_weight for index in used if normalized[index].capability == capability)
        for capability in module.CAPABILITY_WEIGHTS
    }
    assert all(math.isclose(observed[key], value, abs_tol=1e-12) for key, value in module.CAPABILITY_WEIGHTS.items())


class FakeTextMoE(torch.nn.Module):
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
        layer.mlp = torch.nn.Module()
        layer.mlp.shared_expert = torch.nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp.shared_expert, name, torch.nn.Linear(2, 2, bias=False))
        layer.mlp.experts = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
        layer.mlp.gate = torch.nn.Linear(2, 2, bias=False)
        self.mtp = torch.nn.Module()
        self.mtp.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.mtp.layers[0].self_attn = torch.nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.mtp.layers[0].self_attn, name, torch.nn.Linear(2, 2, bias=False))
        self.mtp.layers[0].linear_attn = torch.nn.Module()
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            setattr(self.mtp.layers[0].linear_attn, name, torch.nn.Linear(2, 2, bias=False))


def test_targets_add_shared_mlp_without_routed_experts_or_router():
    module = _module()
    targets = module.resolve_4b_paradigm_targets(FakeTextMoE())
    assert len(targets) == 12
    assert sum("shared_expert" in name for name in targets) == 3
    assert all("experts." not in name and ".mtp." not in name and not name.startswith("mtp.") for name in targets)
    assert all(not name.endswith("mlp.gate") for name in targets)


def test_k3_raw_retention_kl_is_zero_at_identical_policy_and_nonnegative():
    module = _module()
    raw = torch.tensor([[0.0, -0.2, -0.5]])
    mask = torch.tensor([[True, True, False]])
    assert module.sampled_raw_action_kl(raw, raw, mask).item() == 0.0
    shifted = raw + torch.tensor([[0.2, -0.1, 0.0]])
    assert module.sampled_raw_action_kl(shifted, raw, mask).item() > 0.0


def test_wrapper_freezes_4b_paradigm_resources_and_mode():
    source = (ROOT / "scripts/run_m6_phase10c_4b_paradigm_sft_job.sh").read_text()
    assert "#SBATCH --gres=gpu:2" in source
    assert "#SBATCH --mem=96G" in source
    assert "M6_PHASE10C_4B_SFT_MODE:-probe" in source
    assert "--base-model /data/share/model/Qwen3.5-35B-A3B" in source
    assert "m6_phase10c_train_4b_paradigm_teacher_lora.py" in source
