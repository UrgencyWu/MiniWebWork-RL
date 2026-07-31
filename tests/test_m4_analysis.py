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
