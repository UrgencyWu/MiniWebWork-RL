import importlib.util
from pathlib import Path

from miniwebwork.m4_protocol import M4RunConfig


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "m4_run_online.py"
    spec = importlib.util.spec_from_file_location("m4_run_online", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_driver_resolves_two_collection_update_passes_with_adapter_lineage(tmp_path: Path):
    module = _load_module()
    plan = module._commands(
        M4RunConfig("rloo", 20260801, "train"),
        initial_adapter=tmp_path / "initial",
        output_dir=tmp_path / "run",
    )
    assert len(plan) == 2
    assert "--train-pass-index" in plan[0]["collect"]
    assert plan[0]["collect"][plan[0]["collect"].index("--train-pass-index") + 1] == "1"
    assert plan[1]["collect"][plan[1]["collect"].index("--train-pass-index") + 1] == "2"
    assert plan[1]["input_adapter"].endswith("pass_1/update/updated_adapter")
