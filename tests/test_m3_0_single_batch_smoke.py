import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m3_0_single_batch_smoke.py"
    spec = importlib.util.spec_from_file_location("m3_0_single_batch_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CheckpointedLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False


class _FakeTrainableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _CheckpointedLayer()
        self.dropout = torch.nn.Dropout(0.05)
        self.config = SimpleNamespace(use_cache=True)
        self.input_grads_enabled = False
        self.checkpoint_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpoint_kwargs = gradient_checkpointing_kwargs
        self.layer.gradient_checkpointing = True

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


def test_memory_efficient_training_enables_checkpointing_without_dropout():
    smoke = _load_smoke_module()
    model = _FakeTrainableModel()

    runtime = smoke._enable_memory_efficient_training(model)

    assert runtime["gradient_checkpointing"] is True
    assert runtime["checkpoint_api"] == "non_reentrant"
    assert runtime["checkpointed_layers"] == 1
    assert runtime["dropout_modules_forced_eval"] == 1
    assert runtime["model_training"] is True
    assert runtime["use_cache"] is False
    assert model.input_grads_enabled is True
    assert model.checkpoint_kwargs == {"use_reentrant": False}
    assert model.layer.training is True
    assert model.dropout.training is False
