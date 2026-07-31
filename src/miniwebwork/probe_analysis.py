"""Paired analysis for canonical rollout artifacts."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any


def _identity(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_source_sha256": artifact.get("task_source_sha256"),
        "split": artifact.get("split"),
        "temperature": artifact.get("temperature"),
        "top_p": artifact.get("top_p"),
        "top_k": artifact.get("top_k"),
        "K": artifact.get("K"),
        "seed": artifact.get("seed"),
        "prompt_contract": artifact.get("prompt_contract"),
    }


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["task_id"]), int(record["rollout_index"])


def _exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    lower = min(a_only, b_only)
    probability = sum(
        math.comb(discordant, value) for value in range(lower + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _task_bootstrap_ci(
    per_task: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not per_task:
        return [0.0, 0.0]

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(per_task) for _ in per_task]
        denominator = sum(item["comparable_pairs"] for item in selected)
        if denominator == 0:
            continue
        delta = sum(
            item["b_successes"] - item["a_successes"] for item in selected
        ) / denominator
        deltas.append(delta)

    deltas.sort()
    return [_quantile(deltas, 0.025), _quantile(deltas, 0.975)]


def analyze_probe_pair(
    artifact_a: dict[str, Any],
    artifact_b: dict[str, Any],
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260731,
) -> dict[str, Any]:
    """Compare two canonical artifacts using paired task/rollout identities."""
    if not artifact_a.get("complete") or not artifact_b.get("complete"):
        raise ValueError("both rollout artifacts must have complete=true")

    identity_a = _identity(artifact_a)
    identity_b = _identity(artifact_b)
    if identity_a != identity_b:
        raise ValueError(
            "A/B artifacts do not share the same experiment identity: "
            f"A={identity_a}, B={identity_b}"
        )

    records_a = {_record_key(record): record for record in artifact_a.get("records", [])}
    records_b = {_record_key(record): record for record in artifact_b.get("records", [])}
    if len(records_a) != len(artifact_a.get("records", [])):
        raise ValueError("artifact A contains duplicate task/rollout records")
    if len(records_b) != len(artifact_b.get("records", [])):
        raise ValueError("artifact B contains duplicate task/rollout records")
    if set(records_a) != set(records_b):
        missing_in_a = sorted(set(records_b) - set(records_a))
        missing_in_b = sorted(set(records_a) - set(records_b))
        raise ValueError(
            f"A/B pair keys differ; missing_in_a={missing_in_a[:5]}, "
            f"missing_in_b={missing_in_b[:5]}"
        )

    task_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "task_id": "",
            "task_type": "",
            "total_pairs": 0,
            "comparable_pairs": 0,
            "a_successes": 0,
            "b_successes": 0,
            "a_infrastructure": 0,
            "b_infrastructure": 0,
        }
    )
    both_success = a_only = b_only = both_fail = 0
    comparable_pairs = 0
    a_infrastructure = b_infrastructure = 0
    termination_a: Counter[str] = Counter()
    termination_b: Counter[str] = Counter()

    for key in sorted(records_a):
        record_a = records_a[key]
        record_b = records_b[key]
        task_id, _ = key
        row = task_rows[task_id]
        row["task_id"] = task_id
        row["task_type"] = record_a.get("task_type") or record_b.get("task_type", "")
        row["total_pairs"] += 1

        valid_a = bool(record_a.get("rollout_valid"))
        valid_b = bool(record_b.get("rollout_valid"))
        if not valid_a:
            a_infrastructure += 1
            row["a_infrastructure"] += 1
        if not valid_b:
            b_infrastructure += 1
            row["b_infrastructure"] += 1
        if not (valid_a and valid_b):
            continue

        comparable_pairs += 1
        row["comparable_pairs"] += 1
        success_a = bool(record_a.get("success"))
        success_b = bool(record_b.get("success"))
        row["a_successes"] += int(success_a)
        row["b_successes"] += int(success_b)
        termination_a[str(record_a.get("termination_reason", ""))] += 1
        termination_b[str(record_b.get("termination_reason", ""))] += 1

        if success_a and success_b:
            both_success += 1
        elif success_a:
            a_only += 1
        elif success_b:
            b_only += 1
        else:
            both_fail += 1

    per_task = sorted(task_rows.values(), key=lambda row: row["task_id"])
    a_successes = both_success + a_only
    b_successes = both_success + b_only
    delta = (
        (b_successes - a_successes) / comparable_pairs
        if comparable_pairs
        else 0.0
    )

    return {
        "schema_version": "1.1",
        "experiment_identity": identity_a,
        "policy_a": artifact_a.get("policy"),
        "policy_b": artifact_b.get("policy"),
        "total_pairs": len(records_a),
        "comparable_pairs": comparable_pairs,
        "a_infrastructure": a_infrastructure,
        "b_infrastructure": b_infrastructure,
        "a_successes": a_successes,
        "b_successes": b_successes,
        "a_success_rate": a_successes / comparable_pairs if comparable_pairs else 0.0,
        "b_success_rate": b_successes / comparable_pairs if comparable_pairs else 0.0,
        "paired_success_rate_delta_b_minus_a": delta,
        "paired_table": {
            "both_success": both_success,
            "a_only": a_only,
            "b_only": b_only,
            "both_fail": both_fail,
        },
        "exact_mcnemar_pvalue": _exact_mcnemar_pvalue(a_only, b_only),
        "task_bootstrap_95ci_delta_b_minus_a": _task_bootstrap_ci(
            per_task,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "termination_reason_a": dict(sorted(termination_a.items())),
        "termination_reason_b": dict(sorted(termination_b.items())),
        "per_task": per_task,
    }
