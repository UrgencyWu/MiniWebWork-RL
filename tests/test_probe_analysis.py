import pytest

from miniwebwork.probe_analysis import analyze_probe_pair


def _artifact(policy: str, successes: list[bool]) -> dict:
    records = []
    for index, success in enumerate(successes):
        records.append(
            {
                "task_id": "TASK-1",
                "task_type": "no_feasible_product",
                "rollout_index": index,
                "rollout_valid": True,
                "success": success,
                "termination_reason": "verified_submission" if success else "premature_finish",
            }
        )
    return {
        "complete": True,
        "policy": policy,
        "task_source_sha256": "tasks",
        "split": "valid",
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 0,
        "K": len(successes),
        "seed": 7,
        "prompt_contract": "browser_agent_v2",
        "records": records,
    }


def test_paired_analysis_counts_discordant_outcomes():
    result = analyze_probe_pair(
        _artifact("A", [True, True, False, False]),
        _artifact("B", [True, False, True, False]),
        bootstrap_samples=100,
        bootstrap_seed=3,
    )

    assert result["comparable_pairs"] == 4
    assert result["a_successes"] == 2
    assert result["b_successes"] == 2
    assert result["paired_table"] == {
        "both_success": 1,
        "a_only": 1,
        "b_only": 1,
        "both_fail": 1,
    }
    assert result["paired_success_rate_delta_b_minus_a"] == 0.0
    assert result["exact_mcnemar_pvalue"] == 1.0


def test_paired_analysis_rejects_mismatched_distribution():
    artifact_a = _artifact("A", [False, True])
    artifact_b = _artifact("B", [False, True])
    artifact_b["top_k"] = 50

    with pytest.raises(ValueError, match="experiment identity"):
        analyze_probe_pair(artifact_a, artifact_b)


def test_paired_analysis_excludes_infrastructure_pairs():
    artifact_a = _artifact("A", [False, True])
    artifact_b = _artifact("B", [True, True])
    artifact_a["records"][0]["rollout_valid"] = False

    result = analyze_probe_pair(
        artifact_a,
        artifact_b,
        bootstrap_samples=50,
    )

    assert result["total_pairs"] == 2
    assert result["comparable_pairs"] == 1
    assert result["a_infrastructure"] == 1
    assert result["a_successes"] == 1
    assert result["b_successes"] == 1
