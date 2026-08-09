"""Auditable task-clustered statistics for the seven frozen formal models."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..m4_long_horizon_protocol import (
    DATASET_ROOT,
    ONLINE_METHODS,
    ONLINE_SEEDS,
    PROJECT_ROOT,
    SFT_SEED,
    STUDY_ID,
)
from .contracts import RunIdentity, atomic_write_json, sha256_file, sha256_json
from .formal_contract import assert_clean_tracked_worktree, current_git_sha, formal_output_path, load_formal_authorization
from .formal_eval import expected_eval_root, validate_formal_eval_manifest
from .formal_invocation import collect_slurm_accounting
from .formal_online import expected_online_root, validate_formal_online_manifest
from .formal_sft import FORMAL_SFT_ROOT, validate_formal_sft_manifest
from .iteration import IterationStore
from .journal import CollectionStore
from .learner import prepare_group_training_examples, summarize_credit_assignment

FORMAL_ANALYSIS_SCHEMA = "m4_long_horizon_formal_analysis_v1"
FORMAL_ANALYSIS_ROOT = PROJECT_ROOT / "outputs" / STUDY_ID / "formal" / "analysis"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON payload must be an object: {path}")
    return value


def _content_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot take a quantile of an empty sequence")
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]


def _bootstrap_mean(values: Sequence[float], *, samples: int, seed: int) -> list[float]:
    _require(values and samples >= 1000, "bootstrap requires values and at least 1000 samples")
    rng = random.Random(seed)
    n = len(values)
    draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def _paired_sign_permutation_pvalue(differences: Sequence[float], *, samples: int, seed: int) -> float:
    _require(differences and samples >= 1000, "paired permutation requires data")
    observed = abs(sum(differences) / len(differences))
    rng = random.Random(seed)
    extreme = 1
    for _ in range(samples):
        candidate = abs(sum(value if rng.randrange(2) else -value for value in differences) / len(differences))
        extreme += candidate >= observed
    return extreme / (samples + 1)


def _hierarchical_seed_task_bootstrap(
    seed_task_differences: Mapping[int, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    """Resample seeds, then paired tasks within each sampled seed."""

    _require(seed_task_differences and samples >= 1000, "hierarchical bootstrap requires data")
    seeds = sorted(seed_task_differences)
    task_count = len(seed_task_differences[seeds[0]])
    _require(task_count > 0, "hierarchical bootstrap task roster is empty")
    _require(all(len(seed_task_differences[item]) == task_count for item in seeds), "hierarchical bootstrap roster drift")
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        total = 0.0
        count = 0
        for _seed_slot in seeds:
            sampled_seed = seeds[rng.randrange(len(seeds))]
            differences = seed_task_differences[sampled_seed]
            total += sum(differences[rng.randrange(task_count)] for _ in range(task_count))
            count += task_count
        draws.append(total / count)
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def _seed_stratified_task_sign_permutation_pvalue(
    seed_task_differences: Mapping[int, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> float:
    """Randomize paired method labels within task pairs, stratified by seed."""

    _require(seed_task_differences and samples >= 1000, "stratified permutation requires data")
    flattened = [value for item in sorted(seed_task_differences) for value in seed_task_differences[item]]
    _require(bool(flattened), "stratified permutation roster is empty")
    observed = abs(sum(flattened) / len(flattened))
    rng = random.Random(seed)
    extreme = 1
    for _ in range(samples):
        candidate = abs(sum(value if rng.randrange(2) else -value for value in flattened) / len(flattened))
        extreme += candidate >= observed
    return extreme / (samples + 1)


def _task_metadata() -> dict[str, dict[str, Any]]:
    path = DATASET_ROOT / "test" / "test_public.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(len(rows) == len({row["task_id"] for row in rows}) == 120, "analysis test roster drift")
    return {row["task_id"]: row for row in rows}


def _load_model(method: str, seed: int, metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    root = expected_eval_root(method, seed)
    evaluated = validate_formal_eval_manifest(root)
    config = _json(root / "run_config.json")
    identity = RunIdentity.from_mapping(config["identity"])
    store = CollectionStore(root / "collection", identity)
    groups = store.load_committed_groups()
    records = []
    for group in groups:
        task = metadata[group["task_id"]]
        for trajectory in group["trajectories"]:
            turns = trajectory["turns"]
            records.append({
                "task_id": group["task_id"],
                "task_family": task["task_family"],
                "horizon_stratum": task["horizon_stratum"],
                "oracle_min_env_actions": task["oracle_min_env_actions"],
                "rollout_index": trajectory["rollout_index"],
                "success": float(trajectory["success"]),
                "termination_reason": trajectory["termination_reason"] or "unspecified_policy_failure",
                "failure_reasons": list(trajectory["failure_reasons"]),
                "elapsed_s": float(trajectory.get("elapsed_s", 0.0)),
                "action_tokens": trajectory["generated_action_tokens"],
                "model_turns": trajectory["model_turns"],
                "environment_steps": trajectory["environment_steps"],
                "strict_json_turns": sum(turn["strict_json_success"] for turn in turns),
                "schema_valid_turns": sum(turn["schema_valid"] for turn in turns),
                "fallback_turns": sum(turn["fallback_used"] for turn in turns),
                "turn_count": len(turns),
            })
    _require(len(records) == 480, "analysis model lacks 120xK4 trajectories")
    task_values = defaultdict(list)
    for row in records:
        task_values[row["task_id"]].append(row["success"])
    task_success = {task_id: sum(values) / 4 for task_id, values in task_values.items()}
    _require(set(task_success) == set(metadata), "analysis model task roster mismatch")
    return {
        "method": method,
        "seed": seed,
        "eval_manifest_path": evaluated["path"],
        "eval_manifest_sha256": evaluated["sha256"],
        "training_manifest_sha256": evaluated["payload"]["training_manifest_sha256"],
        "records": records,
        "task_success": task_success,
        "eval_summary": evaluated["payload"]["summary"],
        "collection_cost": evaluated["payload"]["collection"],
    }


def _stratified_task_success(model: Mapping[str, Any], metadata: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, float]:
    values = defaultdict(list)
    for task_id, success in model["task_success"].items():
        values[str(metadata[task_id][field])].append(success)
    return {key: sum(rows) / len(rows) for key, rows in sorted(values.items())}


def _model_summary(model: Mapping[str, Any], metadata: Mapping[str, Mapping[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    task_values = [model["task_success"][task_id] for task_id in sorted(metadata)]
    records = model["records"]
    turns = sum(row["turn_count"] for row in records)
    failures = Counter(row["termination_reason"] for row in records if not row["success"])
    failure_reasons = Counter(reason for row in records if not row["success"] for reason in row["failure_reasons"])
    return {
        "method": model["method"],
        "seed": model["seed"],
        "task_macro_success": sum(task_values) / len(task_values),
        "task_cluster_bootstrap_95ci": _bootstrap_mean(task_values, samples=bootstrap_samples, seed=model["seed"]),
        "trajectory_success": sum(row["success"] for row in records) / len(records),
        "task_pass_at_4": sum(value > 0 for value in task_values) / len(task_values),
        "task_all_four_success": sum(value == 1.0 for value in task_values) / len(task_values),
        "by_horizon": _stratified_task_success(model, metadata, "horizon_stratum"),
        "by_task_family": _stratified_task_success(model, metadata, "task_family"),
        "failure_categories": dict(sorted(failures.items())),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "cost": {
            "generated_action_tokens": sum(row["action_tokens"] for row in records),
            "invalidated_action_tokens": model["collection_cost"]["invalidated_action_tokens"],
            "model_turns": sum(row["model_turns"] for row in records),
            "environment_steps": sum(row["environment_steps"] for row in records),
            "sum_trajectory_elapsed_seconds": model["eval_summary"]["sum_trajectory_elapsed_seconds"],
            "finalization_process_elapsed_seconds": model["eval_summary"]["finalization_process_elapsed_seconds"],
        },
        "trajectory_json_analysis": {
            "turn_count": turns,
            "strict_json_success_fraction": sum(row["strict_json_turns"] for row in records) / turns,
            "schema_valid_fraction": sum(row["schema_valid_turns"] for row in records) / turns,
            "fallback_fraction": sum(row["fallback_turns"] for row in records) / turns,
        },
        "eval_manifest_path": model["eval_manifest_path"],
        "eval_manifest_sha256": model["eval_manifest_sha256"],
        "training_manifest_sha256": model["training_manifest_sha256"],
    }


def _formal_training_dynamics(method: str, seed: int) -> dict[str, Any]:
    root = expected_online_root(method, seed).resolve()
    validated = validate_formal_online_manifest(root)
    manifests = IterationStore(root / "state").load_committed_iteration_manifests()
    iteration_rows = []
    for manifest in manifests:
        index = manifest["iteration_index"]
        collection_root = root / "collections" / f"iteration-{index:04d}"
        identity = RunIdentity.from_mapping(_json(collection_root / "run_identity.json"))
        groups = CollectionStore(collection_root, identity).load_committed_groups()
        prepared_groups = [prepare_group_training_examples(group, method, identity=identity) for group in groups]
        recomputed_credit = summarize_credit_assignment(prepared_groups)
        learner = _json(root / "state" / "iterations" / f"iteration-{index:04d}" / "learner_report.json")
        _require(learner.get("credit_assignment") == recomputed_credit, "formal analysis credit-assignment audit drift")
        rewards = [float(trajectory["reward"]) for group in groups for trajectory in group["trajectories"]]
        iteration_rows.append({
            "iteration_index": index,
            "trajectory_success": sum(rewards) / len(rewards),
            "group_count": len(groups),
            "generated_action_tokens": manifest["iteration_generated_action_tokens"],
            "optimizer_updates": learner["optimizer_updates"],
            "effective_optimizer_action_tokens": learner["effective_optimizer_action_tokens"],
            "zero_advantage_group_count": learner["zero_advantage_group_count"],
            "credit_assignment": recomputed_credit,
            "mean_ratio": learner["mean_ratio"],
            "clip_fraction": learner["clip_fraction"],
            "approx_kl": learner["approx_kl"],
            "entropy": learner["entropy"],
            "gradient_norm": learner["gradient_norm"],
            "parameter_change_norm": learner["parameter_change_norm"],
            "learner_elapsed_seconds": learner["learner_elapsed_seconds"],
            "initial_replay_parity": learner["initial_replay_parity"],
        })
    return {
        "method": method,
        "seed": seed,
        "iteration_count": len(iteration_rows),
        "iterations": iteration_rows,
        "total_groups": sum(row["group_count"] for row in iteration_rows),
        "total_optimizer_updates": sum(row["optimizer_updates"] for row in iteration_rows),
        "total_effective_optimizer_action_tokens": sum(row["effective_optimizer_action_tokens"] for row in iteration_rows),
        "total_zero_advantage_groups": sum(row["zero_advantage_group_count"] for row in iteration_rows),
        "total_mixed_reward_groups": sum(row["credit_assignment"]["mixed_reward_group_count"] for row in iteration_rows),
        "total_informative_anchors": sum(row["credit_assignment"]["informative_anchor_count"] for row in iteration_rows),
        "total_learner_elapsed_seconds": sum(row["learner_elapsed_seconds"] for row in iteration_rows),
        "action_token_budget": validated["payload"]["action_token_budget"],
        "training_manifest_sha256": validated["sha256"],
    }


def _paired_comparison(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    task_ids = sorted(first["task_success"])
    _require(task_ids == sorted(second["task_success"]), "paired comparison roster mismatch")
    differences = [second["task_success"][task_id] - first["task_success"][task_id] for task_id in task_ids]
    return {
        "second_minus_first_task_macro_delta": sum(differences) / len(differences),
        "paired_task_bootstrap_95ci": _bootstrap_mean(differences, samples=bootstrap_samples, seed=seed),
        "paired_task_sign_permutation_pvalue_two_sided": _paired_sign_permutation_pvalue(differences, samples=permutation_samples, seed=seed + 7919),
        "task_count": len(task_ids),
    }


def build_formal_analysis(
    *,
    output_dir: Path = FORMAL_ANALYSIS_ROOT,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 20_000,
    sacct_path: Path = Path("/opt/slurm/slurm.25.05/bin/sacct"),
) -> dict[str, Any]:
    _require(bootstrap_samples >= 1000 and permutation_samples >= 1000, "formal statistics sample count is too small")
    assert_clean_tracked_worktree()
    git_sha = current_git_sha()
    authorization = load_formal_authorization()
    root = formal_output_path(output_dir)
    _require(root == FORMAL_ANALYSIS_ROOT.resolve(), "formal analysis root drift")
    metadata = _task_metadata()
    keys = [("verified_sft", SFT_SEED)] + [
        (method, seed) for method in ONLINE_METHODS for seed in ONLINE_SEEDS
    ]
    models = {key: _load_model(*key, metadata) for key in keys}
    summaries = {f"{method}:{seed}": _model_summary(model, metadata, bootstrap_samples) for (method, seed), model in models.items()}
    common_eval_git = {_json(expected_eval_root(method, seed) / "run_config.json")["git_sha"] for method, seed in keys}
    common_readiness = {_json(expected_eval_root(method, seed) / "run_config.json")["readiness_sha256"] for method, seed in keys}
    common_test = {_json(expected_eval_root(method, seed) / "run_config.json")["test_public_sha256"] for method, seed in keys}
    common_authorization = {_json(expected_eval_root(method, seed) / "run_config.json")["authorization_sha256"] for method, seed in keys}
    _require(common_eval_git == {git_sha}, "formal analysis/evaluation Git lineage drift")
    _require(len(common_readiness) == 1, "formal analysis readiness lineage drift")
    _require(common_test == {sha256_file(DATASET_ROOT / "test" / "test_public.jsonl")}, "formal analysis test-file lineage drift")
    _require(common_authorization == {authorization["sha256"]}, "formal analysis evaluation authorization drift")
    shared = validate_formal_sft_manifest(FORMAL_SFT_ROOT)
    _require(shared["payload"]["git_sha"] == git_sha, "formal analysis shared-SFT Git drift")
    _require(shared["payload"]["authorization_sha256"] == authorization["sha256"], "formal analysis authorization drift")
    training_dynamics = {
        f"{method}:{seed}": _formal_training_dynamics(method, seed)
        for method in ONLINE_METHODS for seed in ONLINE_SEEDS
    }
    method_aggregates = {}
    for method in ONLINE_METHODS:
        values = [summaries[f"{method}:{seed}"]["task_macro_success"] for seed in ONLINE_SEEDS]
        method_aggregates[method] = {
            "seed_count": len(values),
            "task_macro_success_mean": statistics.fmean(values),
            "task_macro_success_sample_standard_deviation": statistics.stdev(values),
            "per_seed": {str(seed): summaries[f"{method}:{seed}"]["task_macro_success"] for seed in ONLINE_SEEDS},
        }
    paired_by_seed = {
        str(seed): _paired_comparison(
            models[("multi_turn_grpo", seed)],
            models[("step_aware_gpo", seed)],
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            seed=seed,
        )
        for seed in ONLINE_SEEDS
    }
    differences_by_seed = {}
    for seed in ONLINE_SEEDS:
        first = models[("multi_turn_grpo", seed)]["task_success"]
        second = models[("step_aware_gpo", seed)]["task_success"]
        differences_by_seed[seed] = [second[task_id] - first[task_id] for task_id in sorted(metadata)]
    pooled_differences = [value for seed in ONLINE_SEEDS for value in differences_by_seed[seed]]
    primary = {
        "estimand": "step_aware_gpo_minus_multi_turn_grpo paired task-macro success across three matched seeds",
        "mean_delta": sum(pooled_differences) / len(pooled_differences),
        "hierarchical_seed_then_task_bootstrap_95ci": _hierarchical_seed_task_bootstrap(differences_by_seed, samples=bootstrap_samples, seed=20260809),
        "seed_stratified_paired_task_sign_permutation_pvalue_two_sided": _seed_stratified_task_sign_permutation_pvalue(differences_by_seed, samples=permutation_samples, seed=20260810),
        "paired_seed_task_count": len(pooled_differences),
        "by_seed": paired_by_seed,
        "interpretation_boundary": "Three seeds limit population-level certainty; the interval hierarchically resamples seeds then paired tasks, and the randomization test is paired within seed without claiming universal superiority.",
    }
    accounting_roots = [FORMAL_SFT_ROOT]
    accounting_roots.extend(expected_online_root(method, seed) for method in ONLINE_METHODS for seed in ONLINE_SEEDS)
    accounting_roots.extend(expected_eval_root(method, seed) for method, seed in keys)
    slurm_cost = collect_slurm_accounting(accounting_roots, repo_root=PROJECT_ROOT, sacct_path=sacct_path)
    _require(slurm_cost["job_count"] >= 14 and slurm_cost["gpu_hours"] > 0, "formal Slurm cost matrix is incomplete")
    report = {
        "schema_version": FORMAL_ANALYSIS_SCHEMA,
        "study_id": STUDY_ID,
        "complete": True,
        "authorization_sha256": authorization["sha256"],
        "matrix": {"model_count": 7, "test_tasks_per_model": 120, "rollouts_per_task": 4, "trajectory_count": 3360},
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
        "models": summaries,
        "online_method_seed_aggregates": method_aggregates,
        "training_dynamics": training_dynamics,
        "primary_credit_assignment_comparison": primary,
        "slurm_resource_cost": slurm_cost,
        "audit": {
            "all_seven_eval_manifests_validated": True,
            "all_training_lineage_hashes_validated": True,
            "common_git_sha": git_sha,
            "common_readiness_sha256": next(iter(common_readiness)),
            "common_test_roster_sha256": sha256_file(DATASET_ROOT / "test" / "test_public.jsonl"),
            "excluded_roots": ["outputs/m4_invalidated", "outputs/m4_v2_runs", "outputs/m4_v3_runs"],
            "no_excluded_artifact_loaded": True,
        },
        "reporting": {
            "success": "task-macro and trajectory success with task-clustered confidence intervals",
            "failures": "policy termination categories; infrastructure-invalid attempts excluded from outcomes but retained in cost",
            "cost": "generated and invalidated action tokens plus Slurm GPU-hours, CPU-hours, elapsed time, requested memory, allocation TRES, log hashes, and per-job single-device GPU utilization, VRAM, and power telemetry",
            "trajectory_json": "strict JSON, schema validity, fallback use, horizon, and task-family slices",
            "training_dynamics": "per-iteration success, mixed/zero-signal groups, macro/micro credit, public-anchor coverage, early/middle/late token signal, KL, clipping, entropy, gradients, and parameter change",
        },
    }
    report["report_content_sha256"] = _content_sha(report, "report_content_sha256")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "formal_analysis_report.json"
    atomic_write_json(report_path, report)
    rows = []
    for key, summary in sorted(summaries.items()):
        rows.append({
            "model": key,
            "method": summary["method"],
            "seed": summary["seed"],
            "task_macro_success": summary["task_macro_success"],
            "task_pass_at_4": summary["task_pass_at_4"],
            "ci_low": summary["task_cluster_bootstrap_95ci"][0],
            "ci_high": summary["task_cluster_bootstrap_95ci"][1],
            "generated_action_tokens": summary["cost"]["generated_action_tokens"],
            "invalidated_action_tokens": summary["cost"]["invalidated_action_tokens"],
            "model_turns": summary["cost"]["model_turns"],
            "environment_steps": summary["cost"]["environment_steps"],
        })
    csv_path = root / "formal_analysis_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "m4_long_horizon_formal_analysis_artifacts_v1",
        "complete": True,
        "report_sha256": sha256_file(report_path),
        "results_csv_sha256": sha256_file(csv_path),
    }
    manifest["manifest_content_sha256"] = _content_sha(manifest, "manifest_content_sha256")
    atomic_write_json(root / "analysis_manifest.json", manifest)
    return {"report": str(report_path), "report_sha256": sha256_file(report_path), "csv": str(csv_path), "manifest": str(root / "analysis_manifest.json")}
