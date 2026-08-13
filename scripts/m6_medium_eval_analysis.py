#!/usr/bin/env python3
"""Paired task and training-seed statistics for M6 medium formal-dev evaluation."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json, sha256_json  # noqa: E402
from miniwebwork.m6_mini import validate_closed_loop_identity, validate_mini_rl_audit  # noqa: E402
from miniwebwork.webshop_rl.final_analysis import (  # noqa: E402
    bootstrap_mean_ci,
    crossed_seed_task_bootstrap_ci,
    exact_paired_seed_sign_permutation_pvalue,
    paired_sign_permutation_pvalue,
)


SEEDS = (20260812, 20260813, 20260814)
METHODS = ("multi_turn_grpo", "anchor_gigpo")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_plan(plan: dict) -> dict:
    expected_identities = [
        "raw", "sft",
        "multi_turn_grpo_seed_20260812", "anchor_gigpo_seed_20260812",
        "multi_turn_grpo_seed_20260813", "anchor_gigpo_seed_20260813",
        "multi_turn_grpo_seed_20260814", "anchor_gigpo_seed_20260814",
    ]
    if plan.get("schema_version") != "m6_medium_eval_plan_v1":
        raise ValueError("M6 medium evaluation plan schema drift")
    if plan.get("role") != "formal_dev" or plan.get("task_count") != 500 or plan.get("group_size") != 4:
        raise ValueError("M6 medium evaluation role/size drift")
    if plan.get("rollout_seed") != 20260815 or plan.get("identities") != expected_identities:
        raise ValueError("M6 medium evaluation identity/seed drift")
    if plan.get("paired_training_seeds") != list(SEEDS):
        raise ValueError("M6 medium paired training-seed drift")
    if plan.get("formal_training") is not False or plan.get("isolation") != {
        "mini_dev_reuse_for_selection": False,
        "promotion_opened": False,
        "holdout_opened": False,
    }:
        raise ValueError("M6 medium evaluation isolation drift")
    if plan.get("success_rule") != {
        "mean_rl_minus_sft_pp": 3.0,
        "minimum_positive_seeds": 2,
        "minimum_bootstrap_positive_fraction": 0.8,
        "all_rl_audits_must_pass": True,
    }:
        raise ValueError("M6 medium evaluation success-rule drift")
    statistics_contract = plan.get("statistics", {})
    if statistics_contract.get("bootstrap_samples") != 20000 or statistics_contract.get("permutation_samples") != 20000:
        raise ValueError("M6 medium evaluation statistics-budget drift")
    return plan


def metrics(identity: dict) -> dict:
    return {
        field: identity[field]
        for field in (
            "strict_success_rate", "dense_score", "search_exhaustion_rate",
            "partial_match_purchase_rate", "schema_error_rate", "action_error_rate",
            "mean_generated_action_tokens", "mean_environment_steps",
        )
    }


def differences(first: dict, second: dict) -> list[float]:
    if first["evaluation_contract_sha256"] != second["evaluation_contract_sha256"]:
        raise ValueError("M6 medium paired evaluation contract drift")
    if tuple(first["task_rows"]) != tuple(second["task_rows"]):
        raise ValueError("M6 medium paired task roster drift")
    result = []
    for task_id, row in first["task_rows"].items():
        other = second["task_rows"][task_id]
        if row["rollout_keys"] != other["rollout_keys"]:
            raise ValueError("M6 medium paired rollout-key drift")
        result.append(float(other["strict_success_rate"]) - float(row["strict_success_rate"]))
    return result


def bootstrap_positive(values: dict[int, list[float]], *, samples: int, seed: int) -> float:
    seeds = sorted(values)
    task_count = len(values[seeds[0]])
    rng = random.Random(seed)
    positive = 0
    for _ in range(samples):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        sampled_tasks = [rng.randrange(task_count) for _ in range(task_count)]
        mean = statistics.fmean(values[s][task] for s in sampled_seeds for task in sampled_tasks)
        positive += mean > 0.0
    return positive / samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = validate_plan(load(args.plan))
    samples = int(plan["statistics"]["bootstrap_samples"])
    permutations = int(plan["statistics"]["permutation_samples"])
    identities = {
        label: validate_closed_loop_identity(load(args.eval_root / label / "identity_report.json"))
        for label in plan["identities"]
    }
    if any(item.get("evaluation_role") != "formal_dev" for item in identities.values()):
        raise ValueError("M6 medium evaluation escaped formal_dev")
    raw, sft = identities["raw"], identities["sft"]
    raw_sft = differences(raw, sft)
    methods = {}
    for method in METHODS:
        by_seed: dict[int, list[float]] = {}
        seed_rows = {}
        audit_passes = []
        for seed in SEEDS:
            label = f"{method}_seed_{seed}"
            audit = validate_mini_rl_audit(load(args.online_root / f"seed_{seed}" / method / "rl_audit.json"))
            if audit.get("passed") is not True:
                raise ValueError("M6 medium RL audit failed before evaluation analysis")
            audit_passes.append(audit["passed"] is True)
            values = differences(sft, identities[label])
            by_seed[seed] = values
            seed_rows[str(seed)] = {
                "sft_strict_success_rate": sft["strict_success_rate"],
                "rl_strict_success_rate": identities[label]["strict_success_rate"],
                "rl_minus_sft_pp": statistics.fmean(values) * 100.0,
                "task_bootstrap_95_ci_pp": [value * 100.0 for value in bootstrap_mean_ci(values, samples=samples, seed=seed)],
                "paired_task_permutation_pvalue": paired_sign_permutation_pvalue(values, samples=permutations, seed=seed),
                "rl_audit_content_sha256": audit["content_sha256"],
                "identity_content_sha256": identities[label]["content_sha256"],
                "evaluation_metrics": metrics(identities[label]),
            }
        seed_means = [statistics.fmean(by_seed[seed]) for seed in SEEDS]
        task_means = [statistics.fmean(by_seed[seed][index] for seed in SEEDS) for index in range(len(raw_sft))]
        positive_fraction = bootstrap_positive(by_seed, samples=samples, seed=20260815)
        aggregate = {
            "mean_rl_minus_sft_pp": statistics.fmean(seed_means) * 100.0,
            "positive_seed_count": sum(value > 0.0 for value in seed_means),
            "crossed_seed_task_bootstrap_95_ci_pp": [value * 100.0 for value in crossed_seed_task_bootstrap_ci(by_seed, samples=samples, seed=20260815)],
            "crossed_seed_task_bootstrap_positive_fraction": positive_fraction,
            "paired_task_permutation_pvalue_on_seed_mean": paired_sign_permutation_pvalue(task_means, samples=permutations, seed=20260816),
            "exact_paired_seed_sign_permutation_pvalue": exact_paired_seed_sign_permutation_pvalue(seed_means),
        }
        rule = plan["success_rule"]
        checks = {
            "mean_improvement": aggregate["mean_rl_minus_sft_pp"] >= float(rule["mean_rl_minus_sft_pp"]),
            "positive_seeds": aggregate["positive_seed_count"] >= int(rule["minimum_positive_seeds"]),
            "bootstrap_direction": positive_fraction >= float(rule["minimum_bootstrap_positive_fraction"]),
            "all_rl_audits": all(audit_passes),
        }
        aggregate["checks"] = checks
        aggregate["passed"] = all(checks.values())
        methods[method] = {"seeds": seed_rows, "aggregate": aggregate}

    anchor_grpo = {
        seed: differences(
            identities[f"multi_turn_grpo_seed_{seed}"],
            identities[f"anchor_gigpo_seed_{seed}"],
        )
        for seed in SEEDS
    }
    report = {
        "schema_version": "m6_medium_formal_dev_analysis_v1",
        "study_id": plan["study_id"],
        "development_only": True,
        "formal_training": False,
        "evaluation_role": "formal_dev",
        "task_count": 500,
        "K": 4,
        "rollout_seed": plan["rollout_seed"],
        "raw": {"metrics": metrics(raw), "content_sha256": raw["content_sha256"]},
        "sft": {"metrics": metrics(sft), "content_sha256": sft["content_sha256"]},
        "sft_minus_raw": {
            "delta_pp": statistics.fmean(raw_sft) * 100.0,
            "task_bootstrap_95_ci_pp": [value * 100.0 for value in bootstrap_mean_ci(raw_sft, samples=samples, seed=20260815)],
            "paired_task_permutation_pvalue": paired_sign_permutation_pvalue(raw_sft, samples=permutations, seed=20260815),
        },
        "methods": methods,
        "anchor_minus_grpo": {
            "mean_pp": statistics.fmean(value for rows in anchor_grpo.values() for value in rows) * 100.0,
            "crossed_seed_task_bootstrap_95_ci_pp": [value * 100.0 for value in crossed_seed_task_bootstrap_ci(anchor_grpo, samples=samples, seed=20260817)],
        },
        "decision": "ALLOW_PROMOTION_APPROVAL_REQUEST" if any(item["aggregate"]["passed"] for item in methods.values()) else "STOP_MEDIUM_RL",
        "promotion_opened": False,
        "holdout_opened": False,
        "plan_file_sha256": __import__("hashlib").sha256(args.plan.read_bytes()).hexdigest(),
    }
    report["content_sha256"] = sha256_json(report)
    publish_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
