import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rsft_runner_is_importable_without_starting_collection():
    path = ROOT / "scripts" / "m4_run_rsft.py"
    spec = importlib.util.spec_from_file_location("m4_run_rsft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._single_artifact
