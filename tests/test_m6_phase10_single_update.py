from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from miniwebwork.webshop_rl.phase10_corrective import row_loss_coefficients, validate_source_weights


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(source: str, task: str, state: str, path: str):
    return {"source": source, "task_id": task, "state_id": state, "path_id": path}


def test_hierarchical_coefficients_give_each_source_its_exact_mass():
    rows = [
        _row("new", "n1", "s1", "p1"),
        _row("new", "n1", "s1", "p1"),
        _row("old_sft", "o1", "s1", "p1"),
        _row("current_student", "c1", "s1", "p1"),
    ]
    tokens = [2, 3, 7, 11]
    coefficients = row_loss_coefficients(rows, tokens, {"new": 0.60, "old_sft": 0.25, "current_student": 0.15})
    masses = {}
    for row, count, coefficient in zip(rows, tokens, coefficients):
        masses[row["source"]] = masses.get(row["source"], 0.0) + count * coefficient
    assert masses == pytest.approx({"new": 0.60, "old_sft": 0.25, "current_student": 0.15})


def test_source_weights_fail_closed():
    with pytest.raises(ValueError, match="weights drift"):
        validate_source_weights({"new": 0.50, "old_sft": 0.25, "current_student": 0.25})


def test_exact_rehearsal_selector_never_uses_near_match():
    pytest.importorskip("playwright")
    module = _module("m6_phase10_probe_builder", "scripts/m6_phase10_build_single_update_probe.py")

    class Tokenizer:
        pass

    paths = {"p1": [{"task_id": "t1", "trajectory_id": "p1", "turn_index": 1}]}
    module._path_tokens = lambda rows, tokenizer, config: 24
    with pytest.raises(ValueError, match="no exact"):
        module._select_exact_old_path(paths, target_rows=1, target_tokens=25, tokenizer=Tokenizer(), config=object(), excluded_tasks=set())


def test_control_feasibility_reports_exact_absence_without_fallback():
    pytest.importorskip("playwright")
    module = _module("m6_phase10_probe_feasibility", "scripts/m6_phase10_build_single_update_probe.py")
    paths = {
        "p1": [{"task_id": "t1"}],
        "p2": [{"task_id": "t2"}, {"task_id": "t2"}],
    }
    module._path_tokens = lambda rows, tokenizer, config: 24 if len(rows) == 1 else 26
    result = module.control_feasibility(
        paths,
        target_rows=2,
        target_tokens=25,
        tokenizer=object(),
        config=object(),
        excluded_tasks=set(),
    )
    assert result["control_feasible"] is False
    assert result["exact_matching_old_sft_path_count"] == 0
    assert result["approximate_fallback_allowed"] is False


def test_probe_contract_forbids_raw_reference_and_freezes_one_update():
    module = (ROOT / "src" / "miniwebwork" / "webshop_rl" / "phase10_corrective.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "m6_phase10_run_single_update_probe.py").read_text(encoding="utf-8")
    job = (ROOT / "scripts" / "run_m6_phase10_single_update_probe_job.sh").read_text(encoding="utf-8")
    assert "disable_adapter" not in module and "raw_reference_kl" not in module
    assert "optimizer.step()" in runner
    assert "completed_updates=1" in runner
    assert "old_sft" in runner and "current_student" in runner
    assert "all_prefix_labels_masked" in runner
    assert "#SBATCH --gres=gpu:1" in job and "#SBATCH --time=02:00:00" in job
