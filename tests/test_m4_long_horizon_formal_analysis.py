from pathlib import Path

from miniwebwork.long_horizon_rl.formal_analysis import (
    _bootstrap_mean,
    _hierarchical_seed_task_bootstrap,
    _paired_sign_permutation_pvalue,
    _seed_stratified_task_sign_permutation_pvalue,
)


def test_task_cluster_bootstrap_and_paired_permutation_are_deterministic():
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert _bootstrap_mean(values, samples=1000, seed=7) == _bootstrap_mean(values, samples=1000, seed=7)
    assert _paired_sign_permutation_pvalue(values, samples=1000, seed=9) == _paired_sign_permutation_pvalue(values, samples=1000, seed=9)
    grouped = {1: values, 2: list(reversed(values)), 3: values}
    assert _hierarchical_seed_task_bootstrap(grouped, samples=1000, seed=11) == _hierarchical_seed_task_bootstrap(grouped, samples=1000, seed=11)
    assert _seed_stratified_task_sign_permutation_pvalue(grouped, samples=1000, seed=13) == _seed_stratified_task_sign_permutation_pvalue(grouped, samples=1000, seed=13)


def test_analysis_declares_required_audit_and_failure_cost_slices():
    source = Path("src/miniwebwork/long_horizon_rl/formal_analysis.py").read_text(encoding="utf-8")
    assert "all_seven_eval_manifests_validated" in source
    assert "primary_credit_assignment_comparison" in source
    assert "failure_categories" in source
    assert "invalidated_action_tokens" in source
    assert "trajectory_json_analysis" in source
    assert '"trajectory_count": 3360' in source
    assert "hierarchical_seed_then_task_bootstrap_95ci" in source
    assert "training_dynamics" in source
    assert "collect_slurm_accounting" in source
    assert "task_pass_at_4" in source
