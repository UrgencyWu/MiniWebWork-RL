import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "m4_analyze_final.py"
    spec = importlib.util.spec_from_file_location("m4_analyze_final", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_analysis_input_parser_requires_declared_method_seed_matrix_syntax(tmp_path: Path):
    module = _load_module()
    algorithm, seed, path = module._parse_input_spec(f"gspo:20260801:{tmp_path / 'artifact.json'}")
    assert (algorithm, seed, path.name) == ("gspo", 20260801, "artifact.json")
    with pytest.raises(Exception, match="ALGORITHM"):
        module._parse_input_spec("gspo:20260801")
