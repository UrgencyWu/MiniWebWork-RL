"""Prospective M6 evaluation-size simulation using historical paired tasks."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .long_horizon_rl.contracts import sha256_json

POWER_SCHEMA = "m6_prospective_power_v1"
N_CANDIDATES = (1000, 1500, 2000)
TARGET_EFFECT = 0.03


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_differences(values: Sequence[float], label: str) -> list[float]:
    output = [float(value) for value in values]
    _require(len(output) >= 100, f"M6 historical comparison has too few paired tasks: {label}")
    _require(all(math.isfinite(value) for value in output), f"M6 historical comparison is non-finite: {label}")
    return output


def _design_power(
    values: Sequence[float],
    *,
    n_tasks: int,
    effect: float,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate P(normal-approx paired-task lower bound > 0).

    Each outer draw is a task-cluster resample from the centered historical
    paired differences with the frozen minimum effect added.  The CI uses the
    resampled paired-task standard error; this keeps 20,000-design simulation
    practical while retaining the task as the only resampling unit.
    """

    centered = [float(value) - statistics.fmean(values) for value in values]
    means: list[float] = []
    lower_bounds: list[float] = []
    backend = "numpy_empirical_task_cluster_resampling"
    try:
        import numpy as np

        source = np.asarray(centered, dtype=np.float64)
        rng = np.random.default_rng(seed)
        batch_size = min(256, simulations)
        remaining = simulations
        while remaining:
            current = min(batch_size, remaining)
            indices = rng.integers(0, len(source), size=(current, n_tasks))
            draws = source[indices] + effect
            batch_means = draws.mean(axis=1)
            batch_stderr = draws.std(axis=1, ddof=1) / math.sqrt(n_tasks)
            means.extend(float(item) for item in batch_means)
            lower_bounds.extend(float(item) for item in batch_means - 1.959963984540054 * batch_stderr)
            remaining -= current
    except ImportError:
        # Developer-only diagnostic fallback. A formal split may not be frozen
        # from this approximation; ``build_power_report`` makes that explicit.
        backend = "gaussian_approximation_fallback_not_formal"
        rng = random.Random(seed)
        source_sd = statistics.stdev(centered)
        standard_error = source_sd / math.sqrt(n_tasks)
        for _ in range(simulations):
            mean = rng.gauss(effect, standard_error)
            means.append(mean)
            lower_bounds.append(mean - 1.959963984540054 * standard_error)
    positive_lower = sum(value > 0.0 for value in lower_bounds)
    ordered_means = sorted(means)
    ordered_lowers = sorted(lower_bounds)
    return {
        "n_tasks": n_tasks,
        "simulations": simulations,
        "injected_effect": effect,
        "resampling_backend": backend,
        "probability_ci_lower_above_zero": positive_lower / simulations,
        "simulated_mean_effect_p025": ordered_means[max(0, int(0.025 * simulations) - 1)],
        "simulated_mean_effect_p975": ordered_means[min(simulations - 1, int(0.975 * simulations))],
        "simulated_ci_lower_median": ordered_lowers[simulations // 2],
    }


def build_power_report(
    *,
    comparisons: Mapping[str, Sequence[float]],
    simulations: int = 20_000,
    seed: int = 20260812,
    target_power: float = 0.8,
) -> dict[str, Any]:
    _require(simulations >= 20_000, "M6 prospective power requires at least 20,000 simulations")
    _require(0.0 < target_power < 1.0, "M6 target power is invalid")
    _require(len(comparisons) >= 3, "M6 power report requires Raw-SFT, SFT-RL and Raw-RL histories")
    validated = {
        str(name): _finite_differences(values, str(name))
        for name, values in sorted(comparisons.items())
    }
    comparison_reports: dict[str, Any] = {}
    selected_by_comparison: dict[str, int | None] = {}
    for comparison_index, (name, values) in enumerate(validated.items()):
        candidates = {
            str(n_tasks): _design_power(
                values,
                n_tasks=n_tasks,
                effect=TARGET_EFFECT,
                simulations=simulations,
                seed=seed + comparison_index * 100_000 + n_tasks,
            )
            for n_tasks in N_CANDIDATES
        }
        passing = [
            n_tasks
            for n_tasks in N_CANDIDATES
            if candidates[str(n_tasks)]["probability_ci_lower_above_zero"] >= target_power
        ]
        selected = min(passing) if passing else None
        selected_by_comparison[name] = selected
        comparison_reports[name] = {
            "historical_paired_task_count": len(values),
            "historical_mean_removed_before_simulation": statistics.fmean(values),
            "historical_sample_standard_deviation": statistics.stdev(values),
            "candidates": candidates,
            "selected_minimum_n": selected,
        }
    empirical_backend = all(
        candidate["resampling_backend"] == "numpy_empirical_task_cluster_resampling"
        for item in comparison_reports.values()
        for candidate in item["candidates"].values()
    )
    passed = empirical_backend and all(value is not None for value in selected_by_comparison.values())
    selected_n = max(value for value in selected_by_comparison.values() if value is not None) if passed else None
    report = {
        "schema_version": POWER_SCHEMA,
        "study_id": "m6_monotonic_posttraining_v1",
        "prospective_only": True,
        "m6_outcomes_read": False,
        "method": "empirical task-cluster resampling with paired normal-approximation CI",
        "empirical_resampling_backend_available": empirical_backend,
        "minimum_target_effect": TARGET_EFFECT,
        "target_power": target_power,
        "simulations_per_candidate": simulations,
        "seed": seed,
        "candidate_n": list(N_CANDIDATES),
        "comparisons": comparison_reports,
        "passed": passed,
        "selected_n_eval": selected_n,
        "decision": "FREEZE_N_EVAL" if passed else "STOP_BEFORE_TRAINING",
    }
    report["content_sha256"] = sha256_json(report)
    return report


def validate_power_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == POWER_SCHEMA, "M6 power schema drift")
    _require(value.get("m6_outcomes_read") is False, "M6 power analysis leaked M6 outcomes")
    _require(value.get("simulations_per_candidate", 0) >= 20_000, "M6 power simulation budget drift")
    selected = value.get("selected_n_eval")
    _require(selected is None or selected in N_CANDIDATES, "M6 selected N_eval drift")
    expected_passed = all(
        item.get("selected_minimum_n") in N_CANDIDATES
        for item in value.get("comparisons", {}).values()
    ) and value.get("empirical_resampling_backend_available") is True
    _require(value.get("passed") is expected_passed, "M6 power decision drift")
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 power self-hash drift")
    return value
