import importlib.util
from itertools import combinations
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("sft", "rsft", "rloo", "grpo", "gspo")
SEEDS = (20260801, 20260802, 20260803)


def _load_module():
    path = ROOT / "scripts" / "m4_render_study_summary.py"
    spec = importlib.util.spec_from_file_location("m4_render_study_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rate(numerator: int, denominator: int):
    return {"numerator": numerator, "denominator": denominator, "value": numerator / denominator if denominator else None}


def _summary():
    return {
        "complete": True,
        "valid_attempts": 1,
        "primary_task_macro_success": 0.5,
        "raw_attempt_success_rate": 0.5,
        "primary_task_cluster_bootstrap_95ci": [0.25, 0.75],
        "failure_taxonomy": {"primary_failure_counts": {"output_format_failure": 1}},
        "cost": {
            "action_tokens": 10,
            "model_turns": 1,
            "environment_steps": 1,
            "reported_wall_seconds": 2.0,
        },
        "trajectory_cost_task_macro": {
            metric: {"task_count": 1, "mean": value, "task_cluster_bootstrap_95ci": [value, value]}
            for metric, value in (("environment_steps", 1.0), ("model_turns", 1.0), ("action_tokens", 10.0))
        },
        "trajectory_length_strata": {
            "0-4": {"valid_rollouts": 1, "successes": 1, "success_rate": _rate(1, 1)},
            "5-9": {"valid_rollouts": 0, "successes": 0, "success_rate": _rate(0, 0)},
            "10-14": {"valid_rollouts": 0, "successes": 0, "success_rate": _rate(0, 0)},
            "15-20": {"valid_rollouts": 0, "successes": 0, "success_rate": _rate(0, 0)},
        },
        "json_action_quality": {
            "strict_json_failure_rate": _rate(0, 1),
            "schema_invalid_rate": _rate(0, 1),
            "fallback_recovered_rate": _rate(0, 1),
            "environment_action_failure_rate": _rate(0, 1),
        },
        "adapter_lineage": {
            "verified": True,
            "supervision_audit": {
                "mode": "supervised",
                "completion_tokens_per_epoch": 10,
                "planned_supervised_completion_tokens": 20,
                "zero_completion_label_sample_count": 0,
                "sample_count": 1,
                "zero_completion_label_at_max_length_sample_count": 0,
            },
        },
        "frozen_task_roster_sha256": "frozen-roster",
    }


def _report():
    matrix = {algorithm: {str(seed): _summary() for seed in SEEDS} for algorithm in ALGORITHMS}
    comparisons = {
        f"{second}_minus_{first}": {
            str(seed): {
                "paired_primary_delta_b_minus_a": 0.0,
                "paired_task_cluster_bootstrap_95ci": [0.0, 0.0],
                "paired_permutation_pvalue": 1.0,
            }
            for seed in SEEDS
        }
        for first, second in combinations(sorted(ALGORITHMS), 2)
    }
    return {
        "schema_version": "m4_final_report_v3",
        "complete": True,
        "matrix": matrix,
        "aggregates": {
            algorithm: {"population_std_primary_task_macro_success": 0.0}
            for algorithm in ALGORITHMS
        },
        "pairwise_task_clustered_comparisons": comparisons,
        "audit": {
            "common_initial_adapter_sha256": "shared-adapter",
            "canonical_initial_adapter": {"path": "/canonical/adapter", "sha256": "shared-adapter"},
            "frozen_task_roster_sha256": "frozen-roster",
            "adapter_lineage_verified": True,
            "designated_initial_adapter_verified": True,
            "standard_final_artifact_path_verified": True,
            "frozen_record_roster_verified": True,
        },
    }


def test_v3_renderer_requires_audited_mechanism_evidence_and_renders_json_denominators():
    module = _load_module()
    rendered = module._render(_report())
    assert "完整性与谱系门禁" in rendered
    assert "变长轨迹与任务级成本" in rendered
    assert "JSON 动作质量" in rendered
    assert "离线监督 token 审计" in rendered
    assert "0/3 (0.000)" in rendered


def test_v3_renderer_fails_closed_when_json_or_trajectory_evidence_is_missing():
    module = _load_module()
    report = _report()
    del report["matrix"]["sft"]["20260801"]["json_action_quality"]
    with pytest.raises(ValueError, match="missing v3 report field"):
        module._render(report)
