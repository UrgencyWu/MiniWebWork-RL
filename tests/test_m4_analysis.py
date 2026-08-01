import pytest

from miniwebwork.m4_analysis import (
    aggregate_m4_seed_summaries,
    paired_task_permutation_analysis,
    summarize_m4_evaluation,
)


def _records(policy_b: bool = False):
    records = []
    for task_index in range(2):
        for rollout_index in range(2):
            success = task_index == 0 and (policy_b or rollout_index == 0)
            records.append(
                {
                    "task_id": f"T{task_index}",
                    "task_type": "no_feasible_product" if task_index else "cheapest_feasible",
                    "rollout_index": rollout_index,
                    "rollout_valid": True,
                    "success": success,
                    "reward": float(success),
                    "model_turns": 1,
                    "environment_steps": 1,
                    "elapsed_s": 0.5,
                    "termination_reason": "verified_submission" if success else "premature_finish",
                    "verification": {"failure_reasons": []},
                    "turns": [
                        {
                            "strict_json_success": True,
                            "schema_valid": True,
                            "output_tokens": 3,
                            "action_result": {"success": True},
                        }
                    ],
                }
            )
    return records


def test_m4_evaluation_is_task_clustered_and_cost_auditable():
    summary = summarize_m4_evaluation(
        _records(),
        expected_task_count=2,
        expected_rollouts_per_task=2,
        bootstrap_samples=100,
    )
    assert summary["complete"] is True
    assert summary["primary_task_macro_success"] == 0.25
    assert summary["raw_attempt_success_rate"] == 0.25
    assert summary["cost"]["action_tokens"] == 12
    assert summary["failure_taxonomy"]["policy_failures"] == 3


def test_m4_evaluation_keeps_infrastructure_visible_and_pairing_is_task_level():
    a_records = _records()
    b_records = _records(policy_b=True)
    a_records[-1]["rollout_valid"] = False
    a_records[-1]["reward"] = None
    a_records[-1]["success"] = False
    a = summarize_m4_evaluation(
        a_records,
        expected_task_count=2,
        expected_rollouts_per_task=2,
        bootstrap_samples=100,
    )
    b = summarize_m4_evaluation(
        b_records,
        expected_task_count=2,
        expected_rollouts_per_task=2,
        bootstrap_samples=100,
    )
    assert a["complete"] is False
    assert a["infrastructure_attempts"] == 1
    paired = paired_task_permutation_analysis(
        a, b, permutation_samples=200, bootstrap_samples=100
    )
    assert paired["comparable_tasks"] == 1
    assert paired["paired_primary_delta_b_minus_a"] == 0.5

    aggregate = aggregate_m4_seed_summaries([a, b])
    assert aggregate["seed_count"] == 2
    assert aggregate["total_action_tokens"] == 24


def test_m4_evaluation_rejects_duplicate_task_rollout_identity():
    records = _records()
    records.append(records[0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        summarize_m4_evaluation(records, expected_task_count=2, expected_rollouts_per_task=2)


def test_m4_analysis_reads_strict_rollout_steps_for_cost_and_failure_taxonomy():
    records = _records()
    records[1]["success"] = False
    records[1]["reward"] = 0.0
    records[1]["turns"] = []
    records[1]["steps"] = [
        {
            "strict_json_success": False,
            "schema_valid": False,
            "schema_errors": ["malformed_json"],
            "generated_token_ids": [1, 2, 3, 4],
            "env_action_success": None,
        }
    ]
    summary = summarize_m4_evaluation(
        records,
        expected_task_count=2,
        expected_rollouts_per_task=2,
        bootstrap_samples=100,
        reported_wall_seconds=9.5,
    )

    assert summary["cost"]["action_tokens"] == 13
    assert summary["cost"]["reported_wall_seconds"] == 9.5
    assert summary["failure_taxonomy"]["primary_failure_counts"]["output_format_failure"] == 1


def test_m4_analysis_reports_path_length_strata_cost_ci_and_json_action_denominators():
    records = _records()
    for record, environment_steps in zip(records, (1, 5, 10, 15)):
        record["environment_steps"] = environment_steps
        record["turns"][0]["fallback_used"] = False
    records[1]["turns"][0].update(
        {
            "strict_json_success": False,
            "schema_valid": False,
            "schema_errors": ["malformed_json"],
        }
    )
    records[1]["turns"][0].pop("action_result")
    records[2]["turns"][0].update(
        {
            "fallback_used": True,
            "action_result": {"success": False, "error_code": "invalid_target"},
        }
    )

    summary = summarize_m4_evaluation(
        records,
        expected_task_count=2,
        expected_rollouts_per_task=2,
        bootstrap_samples=100,
    )

    strata = summary["trajectory_length_strata"]
    assert [strata[label]["valid_rollouts"] for label in ("0-4", "5-9", "10-14", "15-20")] == [1, 1, 1, 1]
    assert sum(item["valid_rollouts"] for item in strata.values()) == summary["valid_attempts"]
    assert summary["trajectory_cost_task_macro"]["environment_steps"]["task_count"] == 2
    assert len(summary["trajectory_cost_task_macro"]["action_tokens"]["task_cluster_bootstrap_95ci"]) == 2

    quality = summary["json_action_quality"]
    assert quality["model_decisions"] == 4
    assert quality["strict_json_failure_rate"] == {"numerator": 1, "denominator": 4, "value": 0.25}
    assert quality["schema_invalid_rate"] == {"numerator": 1, "denominator": 4, "value": 0.25}
    assert quality["fallback_recovered_rate"] == {"numerator": 1, "denominator": 4, "value": 0.25}
    assert quality["schema_error_rates"]["malformed_json"]["denominator"] == 4
    assert quality["environment_action_failure_rate"] == {
        "numerator": 1,
        "denominator": 3,
        "value": 1 / 3,
    }


def test_m4_json_quality_excludes_infrastructure_rollouts_from_decision_denominators():
    records = _records()
    records[-1]["rollout_valid"] = False
    records[-1]["turns"][0].update(
        {"strict_json_success": False, "schema_valid": False, "schema_errors": ["malformed_json"]}
    )

    summary = summarize_m4_evaluation(
        records,
        expected_task_count=2,
        expected_rollouts_per_task=2,
        bootstrap_samples=100,
    )

    quality = summary["json_action_quality"]
    assert quality["valid_rollouts"] == 3
    assert quality["model_decisions"] == 3
    assert quality["strict_json_failure_rate"]["numerator"] == 0
