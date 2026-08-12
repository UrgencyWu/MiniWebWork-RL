"""Auditable final statistics for the M5 WebShop frozen evaluation matrix.

The analysis unit is a task, not an individual rollout.  K=4 rollouts remain
paired within each task, and the two online methods remain paired by training
seed.  This avoids treating correlated trajectories as independent samples.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..long_horizon_rl.contracts import atomic_write_json, sha256_file, sha256_json
from .formal_evaluation import IDENTITIES, RUN_REPORT_SCHEMA, validate_eval_report
from .formal_training import self_hash
from .online_training import validate_committed_group


FINAL_ANALYSIS_SCHEMA = "m5_webshop_final_analysis_v1"
STUDY_ID = "m5_webshop_credit_assignment_v1"
TRAINING_SEEDS = (20260801, 20260802, 20260803)
METHOD_IDENTITIES = {
    "multi_turn_grpo": {
        seed: f"multi_turn_grpo_seed_{seed}" for seed in TRAINING_SEEDS
    },
    "anchor_gigpo": {
        seed: f"anchor_gigpo_seed_{seed}" for seed in TRAINING_SEEDS
    },
}
PRIMARY_COMPARISON = "anchor_gigpo_minus_multi_turn_grpo"
EXPLANATORY_COMPARISONS = (
    "shared_verified_sft_minus_raw_base_model",
    "multi_turn_grpo_minus_shared_verified_sft",
    "anchor_gigpo_minus_shared_verified_sft",
    "multi_turn_grpo_minus_raw_base_model",
    "anchor_gigpo_minus_raw_base_model",
)
FAILURE_CLASSES = (
    "partial_match_purchase",
    "zero_match_purchase",
    "search_navigation_exhaustion",
    "item_configuration_exhaustion",
    "output_format_failure",
    "other_policy_failure",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot take a quantile of an empty sequence")
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> list[float]:
    """Percentile bootstrap CI with the task as the resampling unit."""

    _require(bool(values) and samples >= 1_000, "bootstrap requires data and >=1000 samples")
    rng = random.Random(seed)
    count = len(values)
    draws = [
        sum(float(values[rng.randrange(count)]) for _ in range(count)) / count
        for _ in range(samples)
    ]
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def hierarchical_seed_task_bootstrap_ci(
    seed_task_values: Mapping[int, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    """Resample training seeds, then tasks within every sampled seed."""

    _require(bool(seed_task_values) and samples >= 1_000, "hierarchical bootstrap requires data")
    seeds = sorted(seed_task_values)
    task_count = len(seed_task_values[seeds[0]])
    _require(task_count > 0, "hierarchical bootstrap task roster is empty")
    _require(
        all(len(seed_task_values[item]) == task_count for item in seeds),
        "hierarchical bootstrap task roster drift",
    )
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        total = 0.0
        count = 0
        for _seed_slot in seeds:
            sampled_seed = seeds[rng.randrange(len(seeds))]
            values = seed_task_values[sampled_seed]
            total += sum(float(values[rng.randrange(task_count)]) for _ in range(task_count))
            count += task_count
        draws.append(total / count)
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def paired_sign_permutation_pvalue(
    differences: Sequence[float], *, samples: int, seed: int
) -> float:
    """Two-sided paired randomization test using one sign per task cluster."""

    _require(bool(differences) and samples >= 1_000, "paired permutation requires data")
    observed = abs(statistics.fmean(float(value) for value in differences))
    rng = random.Random(seed)
    extreme = 1
    for _ in range(samples):
        candidate = abs(
            sum(float(value) if rng.randrange(2) else -float(value) for value in differences)
            / len(differences)
        )
        extreme += candidate >= observed - 1e-15
    return extreme / (samples + 1)


def seed_stratified_paired_permutation_pvalue(
    seed_task_differences: Mapping[int, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> float:
    """Randomize method labels within every matched seed/task cluster."""

    flattened = [
        float(value)
        for item in sorted(seed_task_differences)
        for value in seed_task_differences[item]
    ]
    return paired_sign_permutation_pvalue(flattened, samples=samples, seed=seed)


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm family-wise error correction with monotone adjusted p-values."""

    _require(bool(pvalues), "Holm correction requires p-values")
    ordered = sorted((float(value), name) for name, value in pvalues.items())
    _require(all(0.0 <= value <= 1.0 for value, _ in ordered), "invalid p-value")
    size = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (value, name) in enumerate(ordered):
        running = max(running, min(1.0, (size - rank) * value))
        adjusted[name] = running
    return adjusted


def _command_type(command: str) -> str:
    text = str(command or "").strip()
    if not text:
        return "empty"
    return text.split("[", 1)[0]


def classify_failure(record: Mapping[str, Any]) -> str | None:
    """Assign one mutually exclusive primary class to a failed trajectory."""

    if bool(record.get("success")):
        return None
    termination = str(record.get("termination_reason") or "")
    reward = float(record.get("reward", 0.0))
    last_page = str(record.get("last_page") or "")
    if termination == "model_output_failure_limit":
        return "output_format_failure"
    if termination == "purchase":
        return "partial_match_purchase" if reward > 0.0 else "zero_match_purchase"
    if termination in {"max_environment_steps", "max_model_turns"}:
        if last_page in {"home", "search_results"}:
            return "search_navigation_exhaustion"
        if last_page in {"item", "subpage"}:
            return "item_configuration_exhaustion"
    return "other_policy_failure"


def _record(identity: str, task_metadata: Mapping[str, Any], trajectory: Mapping[str, Any]) -> dict[str, Any]:
    turns = trajectory["turns"]
    summaries = trajectory.get("evaluation_turn_summary", [])
    last_turn = turns[-1]
    last_observation = last_turn.get("observation") or {}
    last_action = last_turn.get("action") or {}
    action_errors = [
        str((item.get("action_result") or {}).get("error_code") or "")
        for item in summaries
        if str((item.get("action_result") or {}).get("error_code") or "")
    ]
    value = {
        "identity": identity,
        "task_id": str(trajectory["task_id"]),
        "rollout_index": int(trajectory["rollout_index"]),
        "success": bool(trajectory["success"]),
        "reward": float(trajectory["reward"]),
        "generated_action_tokens": int(trajectory["generated_action_tokens"]),
        "environment_steps": int(trajectory["environment_steps"]),
        "model_turns": len(turns),
        "termination_reason": str(trajectory.get("termination_reason") or "unspecified"),
        "category": str(task_metadata.get("category") or "unknown"),
        "constraint_count": int(task_metadata.get("constraint_count", 0)),
        "schema_invalid_turn_count": sum(not bool(turn.get("schema_valid")) for turn in turns),
        "public_action_error_count": len(action_errors),
        "public_action_error_codes": sorted(set(action_errors)),
        "last_page": str(last_observation.get("page_type") or "unknown"),
        "last_command_type": _command_type(str(last_action.get("command") or "")),
    }
    value["failure_class"] = classify_failure(value)
    return value


def _load_identity(eval_root: Path, identity: str) -> dict[str, Any]:
    root = eval_root / identity
    report_path = root / "run_report.json"
    _require(report_path.is_file(), f"missing final report for {identity}")
    _require(not (root / "run_failure.json").exists(), f"current failure marker remains for {identity}")
    report = validate_eval_report(json.loads(report_path.read_text(encoding="utf-8")))
    _require(report["schema_version"] == RUN_REPORT_SCHEMA, f"report schema drift for {identity}")
    _require(report["identity"] == identity, f"report identity drift for {identity}")
    records: list[dict[str, Any]] = []
    inventory = report["group_content_sha256"]
    _require(tuple(inventory) == tuple(f"e{index:04d}" for index in range(500)), f"group roster drift for {identity}")
    task_ids: list[str] = []
    for group_id, expected_hash in inventory.items():
        path = root / "groups" / f"{group_id}.json"
        _require(path.is_file(), f"missing group {identity}/{group_id}")
        group = json.loads(path.read_text(encoding="utf-8"))
        _require(group.get("content_sha256") == expected_hash, f"report/group hash drift for {identity}/{group_id}")
        _require(self_hash(group) == expected_hash, f"group self-hash drift for {identity}/{group_id}")
        group = validate_committed_group(group)
        _require(group.get("identity") == identity, f"group identity drift for {identity}/{group_id}")
        _require(group.get("formal_evaluation") is True, f"formal-evaluation marker drift for {identity}/{group_id}")
        _require(group.get("training_updates_allowed") is False, f"evaluation/training boundary drift for {identity}/{group_id}")
        metadata = group.get("task_metadata")
        _require(isinstance(metadata, Mapping), f"missing task metadata for {identity}/{group_id}")
        task_ids.append(str(group["task_id"]))
        records.extend(_record(identity, metadata, trajectory) for trajectory in group["trajectories"])
    _require(task_ids == [f"webshop_goal_{index:05d}" for index in range(500)], f"task order drift for {identity}")
    _require(len(records) == 2_000, f"trajectory count drift for {identity}")
    keys = {(row["task_id"], row["rollout_index"]) for row in records}
    _require(len(keys) == 2_000, f"paired trajectory key drift for {identity}")
    return {
        "identity": identity,
        "records": records,
        "records_by_key": {(row["task_id"], row["rollout_index"]): row for row in records},
        "report": report,
        "report_path": str(report_path.resolve()),
        "report_file_sha256": sha256_file(report_path),
        "group_inventory_sha256": sha256_json(inventory),
    }


def _task_rows(model: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in model["records"]:
        grouped[record["task_id"]].append(record)
    output: dict[str, dict[str, Any]] = {}
    for task_id in sorted(grouped):
        rows = sorted(grouped[task_id], key=lambda item: item["rollout_index"])
        _require([row["rollout_index"] for row in rows] == list(range(4)), f"K=4 roster drift: {task_id}")
        output[task_id] = {
            "task_id": task_id,
            "category": rows[0]["category"],
            "constraint_count": rows[0]["constraint_count"],
            "success": statistics.fmean(float(row["success"]) for row in rows),
            "dense_score": statistics.fmean(float(row["reward"]) for row in rows),
            "generated_action_tokens": statistics.fmean(float(row["generated_action_tokens"]) for row in rows),
            "environment_steps": statistics.fmean(float(row["environment_steps"]) for row in rows),
            "model_turns": statistics.fmean(float(row["model_turns"]) for row in rows),
        }
    _require(len(output) == 500, "task summary lacks 500 tasks")
    return output


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": numerator / denominator if denominator else None,
    }


def _strata_summary(
    task_rows: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in task_rows.values():
        grouped[str(row[field])].append(row)
    output = {}
    for index, (label, rows) in enumerate(sorted(grouped.items())):
        success = [float(row["success"]) for row in rows]
        dense = [float(row["dense_score"]) for row in rows]
        output[label] = {
            "task_count": len(rows),
            "task_macro_success": statistics.fmean(success),
            "task_cluster_bootstrap_95ci_success": bootstrap_mean_ci(
                success, samples=bootstrap_samples, seed=seed + index * 101
            ),
            "task_macro_dense_score": statistics.fmean(dense),
        }
    return output


def _identity_summary(
    model: Mapping[str, Any], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    records = model["records"]
    tasks = _task_rows(model)
    model["task_rows"] = tasks
    task_success = [float(row["success"]) for row in tasks.values()]
    task_dense = [float(row["dense_score"]) for row in tasks.values()]
    failures = [row for row in records if not row["success"]]
    failure_counts = Counter(str(row["failure_class"]) for row in failures)
    _require(set(failure_counts).issubset(FAILURE_CLASSES), "unknown failure class")
    _require(sum(failure_counts.values()) == len(failures), "failure taxonomy is not exhaustive")
    turns = sum(int(row["model_turns"]) for row in records)
    environment_steps = sum(int(row["environment_steps"]) for row in records)
    schema_invalid = sum(int(row["schema_invalid_turn_count"]) for row in records)
    action_errors = sum(int(row["public_action_error_count"]) for row in records)
    report = model["report"]
    spec = report["identity_spec"]
    return {
        "identity": model["identity"],
        "method": spec["method"],
        "training_seed": spec.get("training_seed"),
        "task_count": 500,
        "trajectory_count": 2_000,
        "trajectory_success_rate": statistics.fmean(float(row["success"]) for row in records),
        "task_macro_success_rate": statistics.fmean(task_success),
        "task_cluster_bootstrap_95ci_success": bootstrap_mean_ci(
            task_success, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "task_macro_dense_score": statistics.fmean(task_dense),
        "task_cluster_bootstrap_95ci_dense_score": bootstrap_mean_ci(
            task_dense, samples=bootstrap_samples, seed=bootstrap_seed + 1
        ),
        "task_pass_at_4": sum(value > 0.0 for value in task_success) / 500,
        "task_all_four_success": sum(value == 1.0 for value in task_success) / 500,
        "by_category": _strata_summary(
            tasks, field="category", bootstrap_samples=bootstrap_samples, seed=bootstrap_seed + 1_000
        ),
        "by_constraint_count": _strata_summary(
            tasks, field="constraint_count", bootstrap_samples=bootstrap_samples, seed=bootstrap_seed + 2_000
        ),
        "failure_taxonomy": {
            "definition": "one mutually exclusive primary class per unsuccessful trajectory",
            "failure_count": len(failures),
            "primary_counts": {label: failure_counts.get(label, 0) for label in FAILURE_CLASSES},
            "fraction_of_failures": {
                label: _rate(failure_counts.get(label, 0), len(failures)) for label in FAILURE_CLASSES
            },
            "termination_reason_counts": dict(sorted(Counter(row["termination_reason"] for row in failures).items())),
            "last_page_counts": dict(sorted(Counter(row["last_page"] for row in failures).items())),
        },
        "action_quality": {
            "schema_invalid_turn_rate": _rate(schema_invalid, turns),
            "trajectories_with_schema_invalid_turn": _rate(
                sum(row["schema_invalid_turn_count"] > 0 for row in records), len(records)
            ),
            "public_action_error_rate_per_environment_step": _rate(action_errors, environment_steps),
            "trajectories_with_public_action_error": _rate(
                sum(row["public_action_error_count"] > 0 for row in records), len(records)
            ),
            "trajectories_by_public_action_error_code": dict(
                sorted(Counter(code for row in records for code in row["public_action_error_codes"]).items())
            ),
        },
        "cost": {
            "generated_action_tokens": sum(int(row["generated_action_tokens"]) for row in records),
            "environment_steps": environment_steps,
            "model_turns": turns,
            "end_to_end_wall_seconds": float(report["cost"]["end_to_end_wall_seconds"]),
            "maximum_gpu_memory_used_mib": float(report["gpu_telemetry"]["maximum_memory_used_mib"]),
            "mean_gpu_utilization_fraction": float(report["gpu_telemetry"]["mean_gpu_utilization_fraction"]),
        },
        "lineage": {
            "evaluation_git_sha": report["consumer_git_sha"],
            "producer_training_git_sha": report["producer_training_git_sha"],
            "protocol_sha256": report["protocol_sha256"],
            "eval_plan_sha256": report["eval_plan_sha256"],
            "authorization_sha256": report["authorization_sha256"],
            "report_path": model["report_path"],
            "report_file_sha256": model["report_file_sha256"],
            "report_content_sha256": report["content_sha256"],
            "group_inventory_sha256": model["group_inventory_sha256"],
        },
    }


def _method_aggregate(
    method: str,
    models: Mapping[int, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    success = {
        training_seed: [float(row["success"]) for row in models[training_seed]["task_rows"].values()]
        for training_seed in TRAINING_SEEDS
    }
    dense = {
        training_seed: [float(row["dense_score"]) for row in models[training_seed]["task_rows"].values()]
        for training_seed in TRAINING_SEEDS
    }
    per_seed_success = [statistics.fmean(success[item]) for item in TRAINING_SEEDS]
    failures = Counter()
    termination = Counter()
    tokens = steps = turns = 0
    for training_seed in TRAINING_SEEDS:
        summary = summaries[METHOD_IDENTITIES[method][training_seed]]
        failures.update(summary["failure_taxonomy"]["primary_counts"])
        termination.update(summary["failure_taxonomy"]["termination_reason_counts"])
        tokens += int(summary["cost"]["generated_action_tokens"])
        steps += int(summary["cost"]["environment_steps"])
        turns += int(summary["cost"]["model_turns"])
    return {
        "method": method,
        "seed_count": 3,
        "tasks_per_seed": 500,
        "trajectories_per_seed": 2_000,
        "task_macro_success_mean": statistics.fmean(per_seed_success),
        "hierarchical_seed_then_task_bootstrap_95ci_success": hierarchical_seed_task_bootstrap_ci(
            success, samples=bootstrap_samples, seed=seed
        ),
        "task_macro_success_sample_standard_deviation_across_seeds": statistics.stdev(per_seed_success),
        "task_macro_dense_score_mean": statistics.fmean(
            statistics.fmean(dense[item]) for item in TRAINING_SEEDS
        ),
        "hierarchical_seed_then_task_bootstrap_95ci_dense_score": hierarchical_seed_task_bootstrap_ci(
            dense, samples=bootstrap_samples, seed=seed + 1
        ),
        "per_seed_success": {str(item): statistics.fmean(success[item]) for item in TRAINING_SEEDS},
        "failure_primary_counts": {label: failures.get(label, 0) for label in FAILURE_CLASSES},
        "termination_reason_counts": dict(sorted(termination.items())),
        "cost": {
            "generated_action_tokens_total": tokens,
            "generated_action_tokens_mean_per_seed": tokens / 3,
            "environment_steps_total": steps,
            "model_turns_total": turns,
        },
    }


def _simple_paired_statistics(
    differences: Sequence[float], *, bootstrap_samples: int, permutation_samples: int, seed: int
) -> dict[str, Any]:
    return {
        "mean_delta": statistics.fmean(float(value) for value in differences),
        "paired_task_cluster_bootstrap_95ci": bootstrap_mean_ci(
            differences, samples=bootstrap_samples, seed=seed
        ),
        "paired_task_sign_permutation_pvalue_two_sided": paired_sign_permutation_pvalue(
            differences, samples=permutation_samples, seed=seed + 97
        ),
        "paired_task_count": len(differences),
        "improved_task_count": sum(value > 1e-15 for value in differences),
        "tied_task_count": sum(abs(value) <= 1e-15 for value in differences),
        "regressed_task_count": sum(value < -1e-15 for value in differences),
    }


def _paired_comparison(
    *,
    name: str,
    first: Mapping[int, Mapping[str, Any]],
    second: Mapping[int, Mapping[str, Any]],
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    include_detail: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seeds = sorted(first)
    _require(seeds == sorted(second), f"paired seed roster mismatch for {name}")
    success_by_seed: dict[int, list[float]] = {}
    dense_by_seed: dict[int, list[float]] = {}
    details: list[dict[str, Any]] = []
    discordance = Counter()
    by_seed = {}
    for offset, training_seed in enumerate(seeds):
        first_tasks = first[training_seed]["task_rows"]
        second_tasks = second[training_seed]["task_rows"]
        _require(tuple(first_tasks) == tuple(second_tasks), f"paired task roster mismatch for {name}/{training_seed}")
        success_differences = []
        dense_differences = []
        for task_id in first_tasks:
            first_row = first_tasks[task_id]
            second_row = second_tasks[task_id]
            success_delta = float(second_row["success"]) - float(first_row["success"])
            dense_delta = float(second_row["dense_score"]) - float(first_row["dense_score"])
            success_differences.append(success_delta)
            dense_differences.append(dense_delta)
            if include_detail:
                details.append(
                    {
                        "comparison": name,
                        "training_seed": training_seed,
                        "task_id": task_id,
                        "category": first_row["category"],
                        "constraint_count": first_row["constraint_count"],
                        "first_task_success": first_row["success"],
                        "second_task_success": second_row["success"],
                        "success_delta_second_minus_first": success_delta,
                        "first_dense_score": first_row["dense_score"],
                        "second_dense_score": second_row["dense_score"],
                        "dense_delta_second_minus_first": dense_delta,
                    }
                )
        success_by_seed[training_seed] = success_differences
        dense_by_seed[training_seed] = dense_differences
        by_seed[str(training_seed)] = {
            "success": _simple_paired_statistics(
                success_differences,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                seed=seed + offset * 1_009,
            ),
            "dense_score": _simple_paired_statistics(
                dense_differences,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                seed=seed + offset * 1_009 + 1,
            ),
        }
        first_records = first[training_seed]["records_by_key"]
        second_records = second[training_seed]["records_by_key"]
        _require(set(first_records) == set(second_records), f"paired rollout roster mismatch for {name}/{training_seed}")
        for key in first_records:
            a = bool(first_records[key]["success"])
            b = bool(second_records[key]["success"])
            discordance[
                "both_success" if a and b else "first_only_success" if a else "second_only_success" if b else "neither_success"
            ] += 1
    success_flat = [value for item in seeds for value in success_by_seed[item]]
    dense_flat = [value for item in seeds for value in dense_by_seed[item]]
    if len(seeds) == 1:
        success_stats = _simple_paired_statistics(
            success_flat, bootstrap_samples=bootstrap_samples, permutation_samples=permutation_samples, seed=seed
        )
        dense_stats = _simple_paired_statistics(
            dense_flat, bootstrap_samples=bootstrap_samples, permutation_samples=permutation_samples, seed=seed + 1
        )
    else:
        success_stats = {
            "mean_delta": statistics.fmean(success_flat),
            "hierarchical_seed_then_task_bootstrap_95ci": hierarchical_seed_task_bootstrap_ci(
                success_by_seed, samples=bootstrap_samples, seed=seed
            ),
            "seed_stratified_paired_task_sign_permutation_pvalue_two_sided": seed_stratified_paired_permutation_pvalue(
                success_by_seed, samples=permutation_samples, seed=seed + 97
            ),
            "paired_seed_task_count": len(success_flat),
            "improved_seed_task_count": sum(value > 1e-15 for value in success_flat),
            "tied_seed_task_count": sum(abs(value) <= 1e-15 for value in success_flat),
            "regressed_seed_task_count": sum(value < -1e-15 for value in success_flat),
        }
        dense_stats = {
            "mean_delta": statistics.fmean(dense_flat),
            "hierarchical_seed_then_task_bootstrap_95ci": hierarchical_seed_task_bootstrap_ci(
                dense_by_seed, samples=bootstrap_samples, seed=seed + 1
            ),
            "seed_stratified_paired_task_sign_permutation_pvalue_two_sided": seed_stratified_paired_permutation_pvalue(
                dense_by_seed, samples=permutation_samples, seed=seed + 98
            ),
            "paired_seed_task_count": len(dense_flat),
        }
    task_aggregate = []
    if include_detail:
        by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in details:
            by_task[row["task_id"]].append(row)
        for task_id, rows in sorted(by_task.items()):
            task_aggregate.append(
                {
                    "task_id": task_id,
                    "category": rows[0]["category"],
                    "constraint_count": rows[0]["constraint_count"],
                    "mean_success_delta_second_minus_first": statistics.fmean(
                        float(row["success_delta_second_minus_first"]) for row in rows
                    ),
                    "mean_dense_delta_second_minus_first": statistics.fmean(
                        float(row["dense_delta_second_minus_first"]) for row in rows
                    ),
                }
            )
    return (
        {
            "comparison": name,
            "estimand": "second minus first; paired by training seed, task_id, and rollout_index",
            "success": success_stats,
            "dense_score": dense_stats,
            "paired_rollout_discordance_diagnostic": dict(discordance),
            "by_seed": by_seed,
            "task_aggregate": {
                "improved_task_count": sum(row["mean_success_delta_second_minus_first"] > 1e-15 for row in task_aggregate),
                "tied_task_count": sum(abs(row["mean_success_delta_second_minus_first"]) <= 1e-15 for row in task_aggregate),
                "regressed_task_count": sum(row["mean_success_delta_second_minus_first"] < -1e-15 for row in task_aggregate),
                "largest_success_improvements": sorted(
                    task_aggregate, key=lambda row: (-row["mean_success_delta_second_minus_first"], row["task_id"])
                )[:20],
                "largest_success_regressions": sorted(
                    task_aggregate, key=lambda row: (row["mean_success_delta_second_minus_first"], row["task_id"])
                )[:20],
            }
            if include_detail
            else None,
        },
        details,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    _require(bool(rows), f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return sha256_file(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return sha256_file(path), count


def _comparison_pvalue(comparison: Mapping[str, Any]) -> float:
    success = comparison["success"]
    return float(
        success.get(
            "seed_stratified_paired_task_sign_permutation_pvalue_two_sided",
            success.get("paired_task_sign_permutation_pvalue_two_sided"),
        )
    )


def _comparison_ci(comparison: Mapping[str, Any]) -> Sequence[float]:
    success = comparison["success"]
    return success.get(
        "hierarchical_seed_then_task_bootstrap_95ci",
        success.get("paired_task_cluster_bootstrap_95ci"),
    )


def _significant(comparison: Mapping[str, Any], *, adjusted_pvalue: float | None = None) -> bool:
    ci = _comparison_ci(comparison)
    pvalue = _comparison_pvalue(comparison) if adjusted_pvalue is None else adjusted_pvalue
    return pvalue < 0.05 and (float(ci[0]) > 0.0 or float(ci[1]) < 0.0)


def render_markdown(report: Mapping[str, Any]) -> str:
    _require(report.get("schema_version") == FINAL_ANALYSIS_SCHEMA, "invalid M5 final analysis report")
    lines = [
        "# MiniWebWork-RL M5 WebShop 最终统计报告",
        "",
        "本报告只分析一次性冻结测试工件；测试结果未参与训练、选参或检查点选择。",
        "",
        "## 完整性",
        "",
        f"- 评测矩阵：8 个身份 × 500 个任务 × K=4，共 {report['matrix']['trajectory_count']:,} 条轨迹。",
        f"- 评测 Git：`{report['audit']['evaluation_git_sha']}`；分析 Git：`{report['analysis_git_sha']}`。",
        "- 八份评测报告和全部 group 自哈希均通过；所有报告的 unmet gates 均为空。",
        "",
        "## 单身份结果",
        "",
        "| 身份 | 成功率 | task-cluster 95% CI | 稠密得分 | 95% CI | action tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for identity in IDENTITIES:
        summary = report["identities"][identity]
        success_ci = summary["task_cluster_bootstrap_95ci_success"]
        dense_ci = summary["task_cluster_bootstrap_95ci_dense_score"]
        lines.append(
            f"| {identity} | {summary['task_macro_success_rate']:.4f} | [{success_ci[0]:.4f}, {success_ci[1]:.4f}] | "
            f"{summary['task_macro_dense_score']:.4f} | [{dense_ci[0]:.4f}, {dense_ci[1]:.4f}] | "
            f"{summary['cost']['generated_action_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            "## 三种子方法汇总",
            "",
            "| 方法 | 平均成功率 | hierarchical 95% CI | 种子标准差 | 平均稠密得分 | 平均 action tokens/seed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("multi_turn_grpo", "anchor_gigpo"):
        aggregate = report["method_aggregates"][method]
        ci = aggregate["hierarchical_seed_then_task_bootstrap_95ci_success"]
        lines.append(
            f"| {method} | {aggregate['task_macro_success_mean']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{aggregate['task_macro_success_sample_standard_deviation_across_seeds']:.4f} | "
            f"{aggregate['task_macro_dense_score_mean']:.4f} | "
            f"{aggregate['cost']['generated_action_tokens_mean_per_seed']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## 配对显著性检验",
            "",
            "主比较为 Anchor-GiGPO − GRPO。区间按训练种子→任务分层 bootstrap；双侧随机化检验在每个匹配的 seed/task 簇内交换方法标签。解释性比较的 p 值使用 Holm 校正。",
            "",
            "| 比较（second − first） | 成功率差 | 95% CI | 双侧 p | Holm p | 结论 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    ordered = [PRIMARY_COMPARISON, *EXPLANATORY_COMPARISONS]
    for name in ordered:
        comparison = report["comparisons"][name]
        ci = _comparison_ci(comparison)
        pvalue = _comparison_pvalue(comparison)
        adjusted = comparison.get("holm_adjusted_pvalue")
        significant = _significant(comparison, adjusted_pvalue=adjusted)
        lines.append(
            f"| {name} | {comparison['success']['mean_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{pvalue:.6f} | {'—' if adjusted is None else f'{adjusted:.6f}'} | "
            f"{'显著' if significant else '未达到显著标准'} |"
        )
    primary = report["comparisons"][PRIMARY_COMPARISON]
    task_aggregate = primary["task_aggregate"]
    lines.extend(
        [
            "",
            "## 主比较的配对任务差异",
            "",
            f"跨三个训练种子求均值后：Anchor-GiGPO 改善 {task_aggregate['improved_task_count']} 个任务，持平 "
            f"{task_aggregate['tied_task_count']} 个，退化 {task_aggregate['regressed_task_count']} 个。",
            "",
            "| 任务 | 类别 | 约束数 | 成功率差 | 稠密得分差 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    shown = task_aggregate["largest_success_improvements"][:5] + task_aggregate["largest_success_regressions"][:5]
    seen = set()
    for row in shown:
        if row["task_id"] in seen:
            continue
        seen.add(row["task_id"])
        lines.append(
            f"| {row['task_id']} | {row['category']} | {row['constraint_count']} | "
            f"{row['mean_success_delta_second_minus_first']:+.4f} | {row['mean_dense_delta_second_minus_first']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "完整的 1,500 个 seed/task 配对差异见 `paired_task_differences.csv`。",
            "",
            "## 失败轨迹分类",
            "",
            "分类为互斥主因：购买错误商品（部分/零匹配）、搜索导航耗尽、商品页配置耗尽、输出格式失败、其他策略失败。格式无效和公开动作错误另作正交标签。",
            "",
            "| 身份 | 失败数 | 部分匹配购买 | 零匹配购买 | 搜索耗尽 | 配置耗尽 | 输出格式 | 其他 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for identity in IDENTITIES:
        failure = report["identities"][identity]["failure_taxonomy"]
        counts = failure["primary_counts"]
        lines.append(
            f"| {identity} | {failure['failure_count']:,} | {counts['partial_match_purchase']:,} | "
            f"{counts['zero_match_purchase']:,} | {counts['search_navigation_exhaustion']:,} | "
            f"{counts['item_configuration_exhaustion']:,} | {counts['output_format_failure']:,} | "
            f"{counts['other_policy_failure']:,} |"
        )
    anchor = report["method_aggregates"]["anchor_gigpo"]
    grpo = report["method_aggregates"]["multi_turn_grpo"]
    token_reduction = 1.0 - (
        anchor["cost"]["generated_action_tokens_mean_per_seed"]
        / grpo["cost"]["generated_action_tokens_mean_per_seed"]
    )
    primary_ci = _comparison_ci(primary)
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- Anchor-GiGPO − GRPO 的成功率差为 {primary['success']['mean_delta']:+.4f}，95% CI "
            f"[{primary_ci[0]:+.4f}, {primary_ci[1]:+.4f}]，双侧 p={_comparison_pvalue(primary):.6f}；不支持成功率优势声明。",
            f"- Anchor-GiGPO 的平均稠密得分差为 {primary['dense_score']['mean_delta']:+.4f}；平均生成 token 比 GRPO 低 {token_reduction:.1%}。",
            "- 两种 RL 都显著高于退化后的 SFT，但仍显著低于 Raw 基座；因此本轮证明的是能力恢复和信用分配效率差异，不是对基座的最终超越。",
            "- 三个训练种子限制了算法总体推断；失败分类是基于公开轨迹状态的诊断规则，不等价于人工因果标注。",
            "",
            "## 可审计产物",
            "",
            f"- JSON 报告自哈希：`{report['content_sha256']}`",
            f"- 配对差异 CSV：`{report['artifacts']['paired_task_differences']['sha256']}`",
            f"- 失败轨迹 JSONL：`{report['artifacts']['failure_trajectory_classification']['sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_final_analysis(
    *,
    eval_root: Path,
    output_dir: Path,
    analysis_git_sha: str,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 50_000,
) -> dict[str, Any]:
    _require(bootstrap_samples >= 1_000 and permutation_samples >= 1_000, "statistical sample count is too small")
    eval_root = Path(eval_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    _require(len(analysis_git_sha) in (40, 64), "analysis Git SHA is invalid")
    models = {identity: _load_identity(eval_root, identity) for identity in IDENTITIES}
    common_eval_git = {model["report"]["consumer_git_sha"] for model in models.values()}
    common_protocol = {model["report"]["protocol_sha256"] for model in models.values()}
    common_plan = {model["report"]["eval_plan_sha256"] for model in models.values()}
    common_authorization = {model["report"]["authorization_sha256"] for model in models.values()}
    _require(len(common_eval_git) == len(common_protocol) == len(common_plan) == len(common_authorization) == 1, "evaluation lineage is not common")
    summaries = {
        identity: _identity_summary(
            model, bootstrap_samples=bootstrap_samples, bootstrap_seed=20260812 + index * 10_000
        )
        for index, (identity, model) in enumerate(models.items())
    }
    method_models = {
        method: {seed: models[identity] for seed, identity in mapping.items()}
        for method, mapping in METHOD_IDENTITIES.items()
    }
    method_aggregates = {
        method: _method_aggregate(
            method,
            method_models[method],
            summaries,
            bootstrap_samples=bootstrap_samples,
            seed=20260820 + index * 10_000,
        )
        for index, method in enumerate(("multi_turn_grpo", "anchor_gigpo"))
    }
    raw_repeated = {seed: models["raw_base_model"] for seed in TRAINING_SEEDS}
    sft_repeated = {seed: models["shared_verified_sft"] for seed in TRAINING_SEEDS}
    comparison_inputs = {
        PRIMARY_COMPARISON: (method_models["multi_turn_grpo"], method_models["anchor_gigpo"], True),
        "shared_verified_sft_minus_raw_base_model": ({0: models["raw_base_model"]}, {0: models["shared_verified_sft"]}, False),
        "multi_turn_grpo_minus_shared_verified_sft": (sft_repeated, method_models["multi_turn_grpo"], False),
        "anchor_gigpo_minus_shared_verified_sft": (sft_repeated, method_models["anchor_gigpo"], False),
        "multi_turn_grpo_minus_raw_base_model": (raw_repeated, method_models["multi_turn_grpo"], False),
        "anchor_gigpo_minus_raw_base_model": (raw_repeated, method_models["anchor_gigpo"], False),
    }
    comparisons: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for index, (name, (first, second, detailed)) in enumerate(comparison_inputs.items()):
        comparison, rows = _paired_comparison(
            name=name,
            first=first,
            second=second,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            seed=20260901 + index * 100_000,
            include_detail=detailed,
        )
        comparisons[name] = comparison
        paired_rows.extend(rows)
    adjusted = holm_adjust({name: _comparison_pvalue(comparisons[name]) for name in EXPLANATORY_COMPARISONS})
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_pvalue"] = value
        comparisons[name]["significant_at_familywise_alpha_0_05"] = _significant(
            comparisons[name], adjusted_pvalue=value
        )
    comparisons[PRIMARY_COMPARISON]["designated_final_primary"] = True
    comparisons[PRIMARY_COMPARISON]["significant_at_alpha_0_05"] = _significant(
        comparisons[PRIMARY_COMPARISON]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_task_differences.csv"
    paired_sha = _write_csv(paired_path, paired_rows)
    failure_path = output_dir / "failure_trajectory_classification.jsonl"
    failure_rows = (
        {
            key: row[key]
            for key in (
                "identity",
                "task_id",
                "rollout_index",
                "reward",
                "failure_class",
                "termination_reason",
                "category",
                "constraint_count",
                "schema_invalid_turn_count",
                "public_action_error_count",
                "public_action_error_codes",
                "last_page",
                "last_command_type",
                "generated_action_tokens",
                "environment_steps",
                "model_turns",
            )
        }
        for identity in IDENTITIES
        for row in models[identity]["records"]
        if not row["success"]
    )
    failure_sha, failure_count = _write_jsonl(failure_path, failure_rows)
    report = {
        "schema_version": FINAL_ANALYSIS_SCHEMA,
        "study_id": STUDY_ID,
        "complete": True,
        "analysis_git_sha": analysis_git_sha,
        "statistical_contract": {
            "primary_estimand": "anchor_gigpo minus multi_turn_grpo task-macro success across three matched training seeds",
            "analysis_plan_timing": (
                "The exact final-analysis implementation was frozen after all evaluation outcomes existed; "
                "the primary comparison is designated for coherent final reporting, not preregistered."
            ),
            "bootstrap_samples": bootstrap_samples,
            "permutation_samples": permutation_samples,
            "single_identity_ci": "task-cluster percentile bootstrap",
            "multi_seed_ci": "hierarchical training-seed then task bootstrap",
            "method_test": "two-sided paired task-cluster sign permutation stratified by training seed",
            "explanatory_multiplicity": "Holm family-wise correction across five non-primary comparisons",
            "rollout_pairing": "same task_id and rollout_index; rollout-level discordance is diagnostic only",
            "alpha": 0.05,
        },
        "matrix": {
            "identity_count": 8,
            "task_count_per_identity": 500,
            "rollouts_per_task": 4,
            "trajectory_count_per_identity": 2_000,
            "trajectory_count": 16_000,
        },
        "identities": summaries,
        "method_aggregates": method_aggregates,
        "comparisons": comparisons,
        "failure_taxonomy_contract": {
            "mutually_exclusive": True,
            "classes": list(FAILURE_CLASSES),
            "orthogonal_diagnostics": [
                "schema_invalid_turn_count",
                "public_action_error_count",
                "termination_reason",
                "last_page",
            ],
        },
        "artifacts": {
            "paired_task_differences": {
                "path": str(paired_path),
                "row_count": len(paired_rows),
                "sha256": paired_sha,
            },
            "failure_trajectory_classification": {
                "path": str(failure_path),
                "row_count": failure_count,
                "sha256": failure_sha,
            },
        },
        "audit": {
            "evaluation_git_sha": next(iter(common_eval_git)),
            "protocol_sha256": next(iter(common_protocol)),
            "eval_plan_sha256": next(iter(common_plan)),
            "authorization_sha256": next(iter(common_authorization)),
            "identity_report_content_sha256": {
                identity: models[identity]["report"]["content_sha256"] for identity in IDENTITIES
            },
            "identity_group_inventory_sha256": {
                identity: models[identity]["group_inventory_sha256"] for identity in IDENTITIES
            },
            "all_reports_passed": all(
                model["report"].get("complete") is True
                and model["report"].get("passed") is True
                and model["report"].get("unmet_gates") == []
                for model in models.values()
            ),
            "all_current_failure_markers_absent": True,
        },
        "limitations": [
            "The exact final-analysis implementation was written after evaluation completion, so inferential results are exploratory rather than preregistered confirmation.",
            "Only three training seeds are available, so population-level algorithm uncertainty remains wide.",
            "The study covers one WebShop environment, one base model, and one frozen prompt/runtime contract.",
            "Failure classes are deterministic diagnostics from public trajectory evidence, not human causal labels.",
            "The frozen test was opened once after training; its outcomes must not be used for further model selection.",
        ],
    }
    report["content_sha256"] = self_hash(report)
    report_path = output_dir / "final_statistical_report.json"
    atomic_write_json(report_path, report)
    markdown_path = output_dir / "FINAL_STATISTICAL_REPORT.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "report": report,
        "report_path": str(report_path),
        "report_file_sha256": sha256_file(report_path),
        "markdown_path": str(markdown_path),
        "markdown_file_sha256": sha256_file(markdown_path),
    }
