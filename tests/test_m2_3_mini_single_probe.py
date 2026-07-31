import importlib.util
from pathlib import Path

import pytest


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m2_3_mini_single_probe.py"
    spec = importlib.util.spec_from_file_location("m2_3_mini_single_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_labels_keep_adapter_identity_explicit():
    probe = _load_probe_module()

    assert probe._resolve_policy_label("A", None) == "A_M2.2R"
    assert probe._resolve_policy_label("B", None) == "B_M2.3-mini"
    assert probe._resolve_policy_label("custom", "M3_one_batch") == "M3_one_batch"


@pytest.mark.parametrize("label", [None, "", "   "])
def test_custom_policy_requires_nonblank_label(label):
    probe = _load_probe_module()

    with pytest.raises(ValueError):
        probe._resolve_policy_label("custom", label)
