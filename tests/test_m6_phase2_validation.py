from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _dropout_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_dropout_probe.py"
    spec = importlib.util.spec_from_file_location("m6_phase2_dropout_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _readiness_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_buy_readiness.py"
    spec = importlib.util.spec_from_file_location("m6_phase2_buy_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase2_coefficient_of_variation():
    module = _dropout_module()
    assert module._coefficient_of_variation([2.0, 2.0, 2.0]) == 0.0
    assert module._coefficient_of_variation([1.0, 3.0]) == pytest.approx(0.5)


def test_phase2_gradient_cosine_is_bounded_and_precise():
    torch = pytest.importorskip("torch")
    module = _dropout_module()
    left = torch.ones(2_000_003, dtype=torch.float32)
    right = left.clone()
    assert module._gradient_cosine(left, right, torch, chunk_size=100_000) == 1.0
    assert module._gradient_cosine(left, -right, torch, chunk_size=100_000) == -1.0


def test_phase2_dropout_probe_has_no_optimizer_step():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_dropout_probe.py"
    ).read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert '"optimizer_steps": 0' in source


def test_phase2_dropout_probe_forces_only_dropout_modules_to_eval():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_dropout_probe.py"
    ).read_text(encoding="utf-8")
    assert "model.train()" in source
    assert "isinstance(module, torch.nn.Dropout)" in source
    assert "module.eval()" in source


def test_phase2_job_is_bounded_and_single_gpu():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_m6_phase2_dropout_probe_job.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --time=02:00:00" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=24G" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "test ! -e \"$output\"" in source


def _evidence(*, page_type: str, item: float, option: float = 0.0, option_count: int = 0):
    module = _readiness_module()
    value = {
        "policy_visible_input_only": True,
        "gradient_attached": False,
        "boundary": "nonterminal_public_state",
        "page_type": page_type,
        "item_attribute_match_fraction": item,
        "selected_option_fraction": option,
        "selected_option_count": option_count,
    }
    value["content_sha256"] = module._self_hash(value)
    return value


def test_phase2_readiness_ignores_search_results():
    module = _readiness_module()
    assert module._readiness_from_evidence(_evidence(page_type="search_results", item=1.0)) == 0.0


def test_phase2_readiness_reserves_option_weight_only_when_required():
    module = _readiness_module()
    assert module._readiness_from_evidence(_evidence(page_type="item", item=0.75)) == pytest.approx(0.75)
    assert module._readiness_from_evidence(
        _evidence(page_type="item", item=0.75, option=0.5, option_count=2)
    ) == pytest.approx(0.7)


def test_phase2_readiness_auc_handles_ties():
    module = _readiness_module()
    assert module._auc([0, 1], [0.5, 0.5]) == 0.5


def test_phase2_p0_submit_has_no_false_dependency():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "submit_m6_phase2_p0.sh"
    ).read_text(encoding="utf-8")
    assert source.count("sbatch --parsable") == 2
    assert "--dependency" not in source
    assert "git status --porcelain --untracked-files=no" in source
