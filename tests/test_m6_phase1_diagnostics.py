from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_phase1_diagnostics.py"
    spec = importlib.util.spec_from_file_location("m6_phase1_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_auc_handles_ties_and_perfect_separation():
    module = _module()
    assert module._auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert module._auc([0, 1], [0.5, 0.5]) == 0.5


def test_auc_requires_both_classes():
    module = _module()
    assert module._auc([0, 0], [0.1, 0.2]) is None


def test_cosine_identity_and_orthogonal():
    module = _module()
    assert module._cosine([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert module._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_iteration_root_from_group_path():
    module = _module()
    path = Path("run/iteration_0/collection/groups/g0000.json")
    assert module._iteration_root(path) == Path("run/iteration_0")


def test_pairing_contract_requires_same_roster_seed_and_k():
    module = _module()
    left = {"task_rows": {"a": {}}, "rollout_seed": 7, "K": 8}
    right = {"task_rows": {"a": {}}, "rollout_seed": 7, "K": 8}
    assert all(module._pairing_contract(left, right).values())
    right["rollout_seed"] = 8
    assert module._pairing_contract(left, right)["same_rollout_seed"] is False


def test_seen_diagnostic_contract_is_read_only(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_collect_policy_success.py"
    spec = importlib.util.spec_from_file_location("m6_collect_policy_success", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    args = Namespace(
        role="mini_train",
        task_roster=Path("roster.json"),
        adapter=Path("adapter"),
        max_model_turns=6,
        max_environment_steps=6,
        maximum_action_tokens=None,
    )
    module.validate_diagnostic_evaluation_contract(args)
    args.maximum_action_tokens = 1
    with pytest.raises(ValueError, match="training token budget"):
        module.validate_diagnostic_evaluation_contract(args)


def test_active_adapter_name_handles_property_and_method():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_phase1_gpu_probe.py"
    spec = importlib.util.spec_from_file_location("m6_phase1_gpu_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class PropertyModel:
        active_adapters = ["default"]

    class MethodModel:
        def active_adapters(self):
            return ["policy"]

    assert module._active_adapter_name(PropertyModel()) == "default"
    assert module._active_adapter_name(MethodModel()) == "policy"


def test_gpu_probe_captures_policy_parameters_before_frozen_reference_load():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "m6_phase1_gpu_probe.py"
    ).read_text(encoding="utf-8")
    capture = source.index("parameters = [parameter for parameter in model.parameters()")
    reference = source.index("model.load_adapter(str(adapter), adapter_name=REFERENCE_ADAPTER_NAME")
    assert capture < reference


def test_gpu_probe_does_not_delete_returned_gradient_vector():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "m6_phase1_gpu_probe.py"
    ).read_text(encoding="utf-8")
    assert "del model, tokenizer, parameters, vector" not in source
    assert "return vector, norm, loss_value" in source
