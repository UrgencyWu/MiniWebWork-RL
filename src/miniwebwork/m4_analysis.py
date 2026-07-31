"""Task-clustered M4 final-evaluation summaries and paired inference."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable

from .model_agent.failure_analysis import classify_failures


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def task_cluster_bootstrap_ci(
    values: list[float],
    *,
    samples: int = 10_000,
    seed: int = 20260801,
) -> list[float]:
    """Non-parametric CI that resamples tasks, not within-task rollouts."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(
        _mean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def _action_token_count(record: dict[str, Any]) -> int:
    total = 0
    for turn in record.get("turns", []):
        if isinstance(turn, dict):
            value = turn.get("output_tokens")
            if isinstance(value, int) and value >= 0:
                total += value
            else:
                generated = turn.get("generated_token_ids", [])
                total += len(generated) if isinstance(generated, list) else 0
    return total


def summarize_m4_evaluation(
    records: list[dict[str, Any]],
    *,
    expected_task_count: int = 120,
    expected_rollouts_per_task: int = 4,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260801,
) -> dict[str, Any]:
    """Summarize one M4 checkpoint on its frozen evaluation rollout records.

    The primary score is the macro mean of per-task valid-rollout reward means.
    Infrastructure-invalid attempts never become zero rewards; incomplete tasks
    stay visible and make the artifact incomplete rather than improving a rate.
    """
    if expected_task_count <= 0 or expected_rollouts_per_task <= 0:
        raise ValueError("expected task and rollout counts must be positive")
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[str, int]] = set()
    for record in records:
        task_id = record.get("task_id")
        rollout_index = record.get("rollout_index")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("every M4 evaluation record needs a non-empty task_id")
        if not isinstance(rollout_index, int) or rollout_index < 0:
            raise ValueError(f"{task_id}: rollout_index must be a non-negative integer")
        key = (task_id, rollout_index)
        if key in seen_keys:
            raise ValueError(f"duplicate M4 evaluation record: {key}")
        seen_keys.add(key)
        rows[task_id].append(record)

    per_task: list[dict[str, Any]] = []
    incomplete_task_ids: list[str] = []
    for task_id, task_records in sorted(rows.items()):
        task_records.sort(key=lambda item: int(item["rollout_index"]))
        if len(task_records) != expected_rollouts_per_task:
            incomplete_task_ids.append(task_id)
        valid = [record for record in task_records if bool(record.get("rollout_valid", True))]
        rewards = [float(record.get("reward", 1.0 if record.get("success") else 0.0)) for record in valid]
        if len(valid) != expected_rollouts_per_task:
            incomplete_task_ids.append(task_id)
        per_task.append(
            {
                "task_id": task_id,
                "task_type": task_records[0].get("task_type", "unknown"),
                "attempted_rollouts": len(task_records),
                "valid_rollouts": len(valid),
                "infrastructure_rollouts": len(task_records) - len(valid),
                "successes": sum(bool(record.get("success")) for record in valid),
                "success_mean": (
                    _mean(rewards)
                    if len(valid) == expected_rollouts_per_task
                    else None
                ),
                "action_tokens": sum(_action_token_count(record) for record in task_records),
                "model_turns": sum(int(record.get("model_turns", 0)) for record in task_records),
                "environment_steps": sum(int(record.get("environment_steps", 0)) for record in task_records),
            }
        )

    if len(per_task) != expected_task_count:
        incomplete_task_ids.extend(
            [f"missing_task_count:{len(per_task)}/{expected_task_count}"]
        )
    valid_task_values = [
        float(row["success_mean"])
        for row in per_task
        if row["success_mean"] is not None
    ]
    failure = classify_failures(records)
    valid_attempts = [record for record in records if bool(record.get("rollout_valid", True))]
    task_type_values: dict[str, list[float]] = defaultdict(list)
    for row in per_task:
        if row["success_mean"] is not None:
            task_type_values[str(row["task_type"])].append(float(row["success_mean"]))
    return {
        "schema_version": "m4_evaluation_summary_v1",
        "complete": not incomplete_task_ids,
        "expected_task_count": expected_task_count,
        "expected_rollouts_per_task": expected_rollouts_per_task,
        "observed_tasks": len(per_task),
        "observed_attempts": len(records),
        "valid_attempts": len(valid_attempts),
        "infrastructure_attempts": len(records) - len(valid_attempts),
        "primary_task_macro_success": _mean(valid_task_values),
        "primary_task_cluster_bootstrap_95ci": task_cluster_bootstrap_ci(
            valid_task_values, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "raw_attempt_success_rate": (
            sum(bool(record.get("success")) for record in valid_attempts) / len(valid_attempts)
            if valid_attempts
            else 0.0
        ),
        "task_type_macro_success": {
            task_type: _mean(values) for task_type, values in sorted(task_type_values.items())
        },
        "failure_taxonomy": failure["summary"],
        "cost": {
            "action_tokens": sum(row["action_tokens"] for row in per_task),
            "model_turns": sum(row["model_turns"] for row in per_task),
            "environment_steps": sum(row["environment_steps"] for row in per_task),
            "reported_wall_seconds": sum(
                float(record.get("elapsed_s", 0.0)) for record in records
            ),
        },
        "incomplete_task_ids": sorted(set(incomplete_task_ids)),
        "per_task": per_task,
    }


def paired_task_permutation_analysis(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    *,
    permutation_samples: int = 20_000,
    permutation_seed: int = 20260801,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Paired task-cluster comparison for two final-test checkpoint summaries."""
    if permutation_samples <= 0:
        raise ValueError("permutation_samples must be positive")
    rows_a = {row["task_id"]: row for row in summary_a.get("per_task", [])}
    rows_b = {row["task_id"]: row for row in summary_b.get("per_task", [])}
    if set(rows_a) != set(rows_b):
        raise ValueError("paired M4 summaries have different task IDs")
    task_ids = sorted(rows_a)
    differences: list[float] = []
    excluded: list[str] = []
    for task_id in task_ids:
        a_value = rows_a[task_id].get("success_mean")
        b_value = rows_b[task_id].get("success_mean")
        if a_value is None or b_value is None:
            excluded.append(task_id)
            continue
        differences.append(float(b_value) - float(a_value))
    if not differences:
        raise ValueError("paired M4 summaries have no valid comparable tasks")
    observed = _mean(differences)
    rng = random.Random(permutation_seed)
    extreme = 0
    for _ in range(permutation_samples):
        permuted = _mean(
            difference if rng.getrandbits(1) else -difference
            for difference in differences
        )
        if abs(permuted) >= abs(observed):
            extreme += 1
    return {
        "schema_version": "m4_paired_task_analysis_v1",
        "comparable_tasks": len(differences),
        "excluded_tasks": excluded,
        "paired_primary_delta_b_minus_a": observed,
        "paired_task_cluster_bootstrap_95ci": task_cluster_bootstrap_ci(
            differences, samples=bootstrap_samples, seed=permutation_seed
        ),
        "paired_permutation_pvalue": (extreme + 1) / (permutation_samples + 1),
        "per_task_differences": [
            {"task_id": task_id, "delta_b_minus_a": float(rows_b[task_id]["success_mean"]) - float(rows_a[task_id]["success_mean"])}
            for task_id in task_ids
            if rows_a[task_id].get("success_mean") is not None
            and rows_b[task_id].get("success_mean") is not None
        ],
    }


def aggregate_m4_seed_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one method's three seed-level final-test summaries."""
    if not summaries:
        raise ValueError("at least one M4 seed summary is required")
    values = [float(summary["primary_task_macro_success"]) for summary in summaries]
    mean = _mean(values)
    population_std = math.sqrt(_mean((value - mean) ** 2 for value in values))
    return {
        "seed_count": len(summaries),
        "seed_primary_task_macro_success": values,
        "mean_primary_task_macro_success": mean,
        "population_std_primary_task_macro_success": population_std,
        "all_summaries_complete": all(bool(summary.get("complete")) for summary in summaries),
        "total_action_tokens": sum(
            int(summary.get("cost", {}).get("action_tokens", 0)) for summary in summaries
        ),
        "total_reported_wall_seconds": sum(
            float(summary.get("cost", {}).get("reported_wall_seconds", 0.0))
            for summary in summaries
        ),
    }
