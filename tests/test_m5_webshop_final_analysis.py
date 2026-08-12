from __future__ import annotations

from miniwebwork.webshop_rl.final_analysis import (
    FINAL_ANALYSIS_SCHEMA,
    bootstrap_mean_ci,
    classify_failure,
    hierarchical_seed_task_bootstrap_ci,
    holm_adjust,
    paired_sign_permutation_pvalue,
)


def test_final_schema_does_not_claim_preregistration():
    assert FINAL_ANALYSIS_SCHEMA == "m5_webshop_final_analysis_v1"
    source = __import__(
        "miniwebwork.webshop_rl.final_analysis", fromlist=["render_markdown"]
    ).render_markdown.__code__.co_consts
    assert "预注册" not in " ".join(item for item in source if isinstance(item, str))


def test_statistics_are_deterministic_and_task_clustered():
    values = [0.0, 0.25, 0.5, 1.0]
    assert bootstrap_mean_ci(values, samples=1_000, seed=7) == bootstrap_mean_ci(
        values, samples=1_000, seed=7
    )
    grouped = {1: values, 2: list(reversed(values)), 3: values}
    assert hierarchical_seed_task_bootstrap_ci(
        grouped, samples=1_000, seed=9
    ) == hierarchical_seed_task_bootstrap_ci(grouped, samples=1_000, seed=9)
    assert paired_sign_permutation_pvalue(
        [1.0, 1.0, 1.0, 1.0], samples=1_000, seed=11
    ) < 0.2


def test_failure_taxonomy_is_mutually_exclusive_and_exhaustive():
    assert classify_failure({"success": True}) is None
    assert classify_failure(
        {"success": False, "termination_reason": "purchase", "reward": 0.4}
    ) == "partial_match_purchase"
    assert classify_failure(
        {"success": False, "termination_reason": "purchase", "reward": 0.0}
    ) == "zero_match_purchase"
    assert classify_failure(
        {
            "success": False,
            "termination_reason": "max_environment_steps",
            "last_page": "search_results",
        }
    ) == "search_navigation_exhaustion"
    assert classify_failure(
        {
            "success": False,
            "termination_reason": "max_model_turns",
            "last_page": "item",
        }
    ) == "item_configuration_exhaustion"
    assert classify_failure(
        {"success": False, "termination_reason": "model_output_failure_limit"}
    ) == "output_format_failure"
    assert classify_failure(
        {"success": False, "termination_reason": "unknown"}
    ) == "other_policy_failure"


def test_holm_adjustment_is_monotone_and_never_smaller_than_raw():
    raw = {"a": 0.01, "b": 0.02, "c": 0.9}
    adjusted = holm_adjust(raw)
    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.9}
    assert all(adjusted[name] >= value for name, value in raw.items())
