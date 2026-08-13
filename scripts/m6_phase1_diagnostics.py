#!/usr/bin/env python3
"""Offline M6 phase-one diagnostics over immutable evaluation and K8 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402
from miniwebwork.webshop_rl.verifier_td import (  # noqa: E402
    ANCHOR_METHOD,
    BASELINE_METHOD,
    assign_group_credit,
)

SEEDS = (20260812, 20260813, 20260814)
METHODS = (BASELINE_METHOD, ANCHOR_METHOD)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def _iter_groups(root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(root.glob("seed_*/**/iteration_*/collection/groups/g*.json")):
        iteration_root = _iteration_root(path)
        if (iteration_root / "learner" / "learner_report.json").is_file():
            yield path, validate_committed_group(_load(path), require_k=8)


def _iter_evaluation_groups(root: Path) -> Iterable[tuple[str, Path, dict[str, Any]]]:
    for identity_root in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted((identity_root / "groups").glob("g*.json")):
            yield identity_root.name, path, validate_committed_group(_load(path), require_k=4)


def _iteration_root(group_path: Path) -> Path:
    return group_path.parents[2]


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    wins = sum(left > right for left in positive for right in negative)
    ties = sum(left == right for left in positive for right in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 1.0


def _mean_task_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    tasks: set[str],
) -> dict[str, Any]:
    shared = sorted(tasks & set(baseline["task_rows"]) & set(candidate["task_rows"]))
    values = [
        float(candidate["task_rows"][task]["strict_success_rate"])
        - float(baseline["task_rows"][task]["strict_success_rate"])
        for task in shared
    ]
    return {
        "paired_task_count": len(shared),
        "task_ids": shared,
        "mean_strict_delta_pp": statistics.fmean(values) * 100.0 if values else None,
        "positive_task_count": sum(value > 0 for value in values),
        "negative_task_count": sum(value < 0 for value in values),
    }


def _pairing_contract(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "same_task_roster": tuple(baseline["task_rows"]) == tuple(candidate["task_rows"]),
        "same_rollout_seed": baseline.get("rollout_seed") == candidate.get("rollout_seed"),
        "same_K": baseline.get("K") == candidate.get("K") == 8,
    }


def _validate_seen_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == "m6_phase1_seen_task_identity_v1", "M6 seen identity schema drift")
    _require(value.get("content_sha256") == _self_hash(value), "M6 seen identity self-hash drift")
    _require(value.get("development_only") is True and value.get("formal_training") is False, "M6 seen identity scope drift")
    _require(value.get("K") == 8 and value.get("trajectory_count") == value.get("task_count") * 8, "M6 seen identity size drift")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--seen-eval-root", type=Path)
    parser.add_argument("--gpu-probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    online_root = args.online_root.expanduser().resolve()
    eval_root = args.eval_root.expanduser().resolve()
    groups = list(_iter_groups(online_root))
    _require(groups, "M6 phase-one diagnostics found no K8 update groups")

    accepted_by_identity: dict[str, set[str]] = {}
    credit_rows: list[dict[str, Any]] = []
    for path, group in groups:
        parts = path.parts
        seed = next(part for part in parts if part.startswith("seed_"))
        method = next(part for part in parts if part in METHODS)
        identity = f"{method}_{seed}"
        accepted_by_identity.setdefault(identity, set()).add(group["task_id"])
        grpo = assign_group_credit(group["trajectories"], method=BASELINE_METHOD)
        anchor = assign_group_credit(group["trajectories"], method=ANCHOR_METHOD)
        left = [item["turn_advantage"] for trajectory in grpo["turn_credit"] for item in trajectory]
        right = [item["turn_advantage"] for trajectory in anchor["turn_credit"] for item in trajectory]
        difference_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
        base_norm = math.sqrt(sum(value * value for value in left))
        credit_rows.append(
            {
                "group_content_sha256": group["content_sha256"],
                "task_id": group["task_id"],
                "advantage_cosine": _cosine(left, right),
                "relative_advantage_norm_difference": difference_norm / base_norm if base_norm else 0.0,
                "sign_flip_count": sum((a > 0) != (b > 0) for a, b in zip(left, right) if a != 0 and b != 0),
            }
        )
    expected_identities = {f"{method}_seed_{seed}" for method in METHODS for seed in SEEDS}
    _require(set(accepted_by_identity) == expected_identities, "M6 phase-one update identity matrix is incomplete")
    accepted_union = set().union(*accepted_by_identity.values())
    accepted_intersection = set.intersection(*accepted_by_identity.values())
    seen_eval: dict[str, Any] = {}
    learning_rate_behavior: dict[str, Any] = {}
    if args.seen_eval_root is not None:
        seen_root = args.seen_eval_root.expanduser().resolve()
        sft = _validate_seen_report(_load(seen_root / "sft" / "seen_task_report.json"))
        for method in METHODS:
            for seed in SEEDS:
                label = f"{method}_seed_{seed}"
                candidate = _validate_seen_report(_load(seen_root / label / "seen_task_report.json"))
                pairing = _pairing_contract(sft, candidate)
                _require(all(pairing.values()), "M6 seen RL paired evaluation contract drift")
                seen_eval[label] = {**_mean_task_delta(sft, candidate, accepted_union), "pairing_checks": pairing}
        for label in ("lr_1e_6", "lr_3e_6", "lr_1e_5"):
            candidate = _validate_seen_report(_load(seen_root / label / "seen_task_report.json"))
            pairing = _pairing_contract(sft, candidate)
            _require(all(pairing.values()), "M6 seen LR paired evaluation contract drift")
            learning_rate_behavior[label] = {**_mean_task_delta(sft, candidate, accepted_union), "pairing_checks": pairing}
    else:
        seen_eval = {"status": "PENDING_FRESH_PAIRED_K8_SEEN_TASK_EVALUATION"}
        learning_rate_behavior = {"status": "PENDING_FRESH_PAIRED_K8_SEEN_TASK_EVALUATION"}

    labels: list[int] = []
    terminal_scores: list[float] = []
    maximum_preterminal: list[float] = []
    final_preterminal: list[float] = []
    failure_high = 0
    failure_count = 0
    failure_classes: dict[str, int] = {}
    evaluation_group_count = 0
    evaluation_identities: set[str] = set()
    for identity, _, group in _iter_evaluation_groups(eval_root):
        evaluation_group_count += 1
        evaluation_identities.add(identity)
        for trajectory in group["trajectories"]:
            strict = int(bool(trajectory["success"]))
            preterminal = [float(turn["verifier_progress_after_action"]) for turn in trajectory["turns"][:-1]]
            maximum = max(preterminal, default=0.0)
            final = preterminal[-1] if preterminal else 0.0
            labels.append(strict)
            terminal_scores.append(float(trajectory["task_score"]))
            maximum_preterminal.append(maximum)
            final_preterminal.append(final)
            if not strict:
                failure_count += 1
                failure_high += int(maximum >= 0.5)
                if trajectory.get("termination_reason") == "purchase":
                    label = "partial_purchase" if float(trajectory["task_score"]) > 0 else "zero_match_purchase"
                elif trajectory.get("termination_reason") in {"max_environment_steps", "max_model_turns"}:
                    label = "horizon_exhaustion"
                elif any(turn.get("schema_valid") is not True for turn in trajectory["turns"]):
                    label = "schema_failure"
                else:
                    label = "other_failure"
                failure_classes[label] = failure_classes.get(label, 0) + 1

    maximum_auc = _auc(labels, maximum_preterminal)
    final_auc = _auc(labels, final_preterminal)
    dense_auc = _auc(labels, terminal_scores)
    phi_stop = (
        maximum_auc is None
        or maximum_auc <= 0.60
        or (failure_count > 0 and failure_high / failure_count > 0.80)
    )
    mean_cosine = statistics.fmean(row["advantage_cosine"] for row in credit_rows)
    mean_norm_difference = statistics.fmean(row["relative_advantage_norm_difference"] for row in credit_rows)
    credit_stop = mean_cosine >= 0.98 and mean_norm_difference < 0.10
    gpu_probe = None
    if args.gpu_probe is not None:
        gpu_probe = _load(args.gpu_probe.expanduser().resolve())
        _require(gpu_probe.get("schema_version") == "m6_phase1_gpu_probe_v1", "M6 phase-one GPU probe schema drift")
        _require(gpu_probe.get("content_sha256") == _self_hash(gpu_probe), "M6 phase-one GPU probe self-hash drift")

    report = {
        "schema_version": "m6_phase1_offline_diagnostics_v1",
        "development_only": True,
        "formal_training": False,
        "source_online_root": str(online_root),
        "source_eval_root": str(eval_root),
        "source_group_count": len(groups),
        "accepted_task_union_count": len(accepted_union),
        "accepted_task_intersection_count": len(accepted_intersection),
        "accepted_task_ids": sorted(accepted_union),
        "seen_task_diagnostics": seen_eval,
        "phi_diagnostics": {
            "evaluation_identity_count": len(evaluation_identities),
            "evaluation_group_count": evaluation_group_count,
            "trajectory_count": len(labels),
            "strict_success_count": sum(labels),
            "maximum_preterminal_phi_auc_for_strict_success": maximum_auc,
            "final_preterminal_phi_auc_for_strict_success": final_auc,
            "official_terminal_dense_score_auc_for_strict_success": dense_auc,
            "failure_count": failure_count,
            "failure_with_maximum_preterminal_phi_at_least_0_5_count": failure_high,
            "failure_with_high_phi_fraction": failure_high / failure_count if failure_count else 0.0,
            "failure_class_counts": dict(sorted(failure_classes.items())),
            "anchor_stop_thresholds": {"auc_maximum": 0.60, "high_phi_failure_fraction_maximum": 0.80},
            "anchor_stop_recommended": phi_stop,
        },
        "credit_counterfactual": {
            "group_count": len(credit_rows),
            "mean_advantage_cosine": mean_cosine,
            "minimum_advantage_cosine": min(row["advantage_cosine"] for row in credit_rows),
            "mean_relative_advantage_norm_difference": mean_norm_difference,
            "total_turn_sign_flips": sum(row["sign_flip_count"] for row in credit_rows),
            "parameter_gradient_probe_required": True,
            "redundancy_thresholds": {"gradient_cosine_minimum": 0.98, "gradient_norm_difference_maximum": 0.10},
            "advantage_level_redundancy_recommended": credit_stop,
            "parameter_gradient_probe": gpu_probe["gradient_counterfactual"] if gpu_probe else None,
        },
        "learning_rate_probe": {
            "parameter_and_fixed_state": gpu_probe["learning_rate_probe"] if gpu_probe else None,
            "seen_task_behavior": learning_rate_behavior,
        },
        "decisions": {
            "do_not_open_promotion_or_holdout": True,
            "do_not_scale_medium_rl": True,
            "run_parameter_gradient_probe": gpu_probe is None,
            "run_learning_rate_probe": gpu_probe is None,
            "stop_anchor_as_redundant": (
                gpu_probe["gradient_counterfactual"]["anchor_redundant"] if gpu_probe else None
            ),
        },
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
