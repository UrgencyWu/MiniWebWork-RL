import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m3_0_generate_report.py"
    spec = importlib.util.spec_from_file_location("m3_0_generate_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _update():
    return {
        "complete": True,
        "passed": True,
        "formal_update": True,
        "code_git_sha": "abc",
        "source_artifact": "artifact.json",
        "source_artifact_sha256": "artifact-sha",
        "source_adapter_sha256": "base-sha",
        "checkpoint_path": "adapter",
        "checkpoint_adapter_sha256": "update-sha",
        "sampling_distribution": {"temperature": 1.0},
        "selected_group": {"task_id": "TASK", "valid_for_grpo_update": True},
        "pre_update_replay_audit": {"max_abs_difference": 0.0},
        "optimizer": {"changed_parameter_tensors": 1},
    }


def _paired(delta: float, ci: list[float], pvalue: float):
    metric = {
        "overall_success_rate": 0.5,
        "infrastructure_errors": 0,
        "feasible_success_rate": 0.5,
        "false_no_solution_count": 0,
    }
    return {
        "comparable_pairs": 8,
        "policy_a": "M2.2R",
        "policy_b": "M3",
        "policy_a_metrics": metric,
        "policy_b_metrics": metric,
        "paired_success_rate_delta_b_minus_a": delta,
        "task_bootstrap_95ci_delta_b_minus_a": ci,
        "exact_mcnemar_pvalue": pvalue,
        "experiment_identity": {},
        "termination_reason_a": {},
        "termination_reason_b": {},
    }


def test_report_preserves_negative_result_conclusion():
    module = _load_module()

    report = module.build_report(
        _update(),
        _paired(-0.125, [-0.25, 0.0], 0.5),
        update_path=Path("update.json"),
        paired_path=Path("paired.json"),
    )

    assert "no improvement" in report["paired_frozen_evaluation"]["conclusion"]
    markdown = module.render_markdown(report)
    assert "Infrastructure-invalid trajectories" in markdown


def test_report_identifies_significant_positive_paired_evidence():
    module = _load_module()

    report = module.build_report(
        _update(),
        _paired(0.25, [0.125, 0.5], 0.01),
        update_path=Path("update.json"),
        paired_path=Path("paired.json"),
    )

    assert "positive paired evidence" in report["paired_frozen_evaluation"]["conclusion"]
