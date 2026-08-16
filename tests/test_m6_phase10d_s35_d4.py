from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    name = "m6_phase10d_train_s35_d4"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/m6_phase10d_train_s35_d4.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Example:
    def __init__(self, sample_id: str, task_id: str, tokens: int):
        self.sample_id = sample_id
        self.task_id = task_id
        self.completion_label_tokens = tokens


def test_formal_schedule_preserves_d4_one_epoch_contract():
    module = _module()
    schedule = module.build_update_schedule(2209)
    used = [index for item in schedule for index in item["imitation_indices"]]
    assert len(schedule) == 245
    assert len(used) == 2205
    assert len(set(used)) == 2205
    assert 2209 - len(used) == 4
    assert all(len(item["imitation_indices"]) == 9 for item in schedule)


def test_probe_is_first_two_frozen_formal_updates():
    module = _module()
    formal = module.build_update_schedule(2209)
    probe = module.active_schedule(formal, "probe")
    assert probe == formal[:2]
    assert len({index for item in probe for index in item["imitation_indices"]}) == 18


def test_token_objective_matches_original_d4_batch_normalization():
    module = _module()
    examples = [Example("a", "t1", 1), Example("b", "t2", 3), Example("c", "t3", 6)]
    observed = module.token_weighted_coefficients(examples)
    assert observed == (0.1, 0.3, 0.6)
    assert math.isclose(sum(observed), 1.0, abs_tol=1e-12)


def test_s35_d4_config_keeps_successful_4b_paradigm_values():
    module = _module()
    config = module.s35_d4_config()
    assert config.maximum_sequence_tokens == 8192
    assert config.learning_rate == 2e-5
    assert config.raw_reference_kl == 0.03
    assert config.imitation_per_update == 9
    assert config.retention_per_update == 1
    assert config.maximum_epochs == 1
    assert config.lora == {"r": 16, "alpha": 32, "dropout": 0.05}
    assert config.chat_template_kwargs == {"enable_thinking": False}


def test_cross_model_retention_never_uses_raw4_token_ids():
    module = _module()
    source = (ROOT / "scripts/m6_phase10d_train_s35_d4.py").read_text()
    assert "historical_raw4_retention_token_ids_used\": False" in source
    assert "adapter-disabled Raw35 reference" in source
    assert module.FROZEN_D4["retention_sha256"]


def test_wrapper_freezes_probe_default_and_two_gpu_model_parallel():
    source = (ROOT / "scripts/run_m6_phase10d_s35_d4_job.sh").read_text()
    assert "#SBATCH --gres=gpu:2" in source
    assert "#SBATCH --mem=96G" in source
    assert "M6_PHASE10D_MODE:-probe" in source
    assert "--base-model /data/share/model/Qwen3.5-35B-A3B" in source
    assert "m6_phase10d_train_s35_d4.py" in source
