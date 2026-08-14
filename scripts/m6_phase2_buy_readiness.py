#!/usr/bin/env python3
"""Calibrate a public item/option buy-readiness score on frozen M6 trajectories."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402

ITEM_PAGES = {"item", "product", "subpage", "item_subpage", "item_options", "options", "product_options", "item_with_options"}
LEGACY_FORMULA = "item_80_option_20_v1"
EQUAL_BLOCK_FORMULA = "equal_item_option_blocks_v2"
FORMULAS = (LEGACY_FORMULA, EQUAL_BLOCK_FORMULA)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    wins = sum(left > right for left in positive for right in negative)
    ties = sum(left == right for left in positive for right in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _readiness_from_evidence(
    evidence: Mapping[str, Any],
    *,
    formula_version: str = LEGACY_FORMULA,
) -> float:
    _require(evidence.get("policy_visible_input_only") is True, "M6 Phase2 buy-readiness found non-public evidence")
    _require(evidence.get("gradient_attached") is False, "M6 Phase2 buy-readiness evidence is gradient-attached")
    _require(evidence.get("content_sha256") == _self_hash(evidence), "M6 Phase2 verifier evidence self-hash drift")
    if str(evidence.get("boundary")) != "nonterminal_public_state":
        return 0.0
    if str(evidence.get("page_type", "")).casefold() not in ITEM_PAGES:
        return 0.0
    item = float(evidence.get("item_attribute_match_fraction", 0.0))
    option = float(evidence.get("selected_option_fraction", 0.0))
    option_count = int(evidence.get("selected_option_count", 0))
    _require(0.0 <= item <= 1.0 and 0.0 <= option <= 1.0, "M6 Phase2 readiness component left [0, 1]")
    _require(formula_version in FORMULAS, "unsupported M6 Phase2 readiness formula")
    if option_count == 0:
        # If the task has no option constraint, the public item block (which
        # already includes public price evidence) owns the full score.
        return item
    if formula_version == LEGACY_FORMULA:
        return 0.8 * item + 0.2 * option
    # Product constraints and selected options are two semantically necessary
    # blocks: choosing the right item with the wrong option is still a failed
    # purchase.  Equal block weight is fixed by that contract, not fitted from
    # task outcomes, and uses only detached public evidence.
    return 0.5 * item + 0.5 * option


def _trajectory_row(
    trajectory: Mapping[str, Any],
    *,
    formula_version: str = LEGACY_FORMULA,
) -> dict[str, Any]:
    turns = trajectory.get("turns")
    _require(isinstance(turns, list) and turns, "M6 Phase2 trajectory has no turns")
    preterminal_scores = [
        _readiness_from_evidence(
            turn["verifier_progress_evidence"],
            formula_version=formula_version,
        )
        for turn in turns[:-1]
    ]
    success = bool(trajectory.get("success"))
    task_score = float(trajectory.get("task_score", 0.0))
    termination = str(trajectory.get("termination_reason", ""))
    if success:
        failure_class = "strict_success"
    elif termination == "purchase" and task_score > 0.0:
        failure_class = "partial_purchase"
    elif termination == "purchase":
        failure_class = "zero_match_purchase"
    elif termination in {"max_environment_steps", "max_model_turns"}:
        failure_class = "horizon_exhaustion"
    elif any(turn.get("schema_valid") is not True for turn in turns):
        failure_class = "schema_failure"
    else:
        failure_class = "other_failure"
    return {
        "task_id": str(trajectory.get("task_id", "")),
        "success": success,
        "failure_class": failure_class,
        "final_preterminal_buy_readiness": preterminal_scores[-1] if preterminal_scores else 0.0,
        "maximum_preterminal_buy_readiness": max(preterminal_scores, default=0.0),
    }


def _iter_trajectories(root: Path, identity: str) -> Iterable[Mapping[str, Any]]:
    for path in sorted((root / identity / "groups").glob("g*.json")):
        group = validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=4)
        yield from group["trajectories"]


def _dataset_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    _require(rows, "M6 Phase2 buy-readiness found no frozen trajectories")
    labels = [int(row["success"]) for row in rows]
    final_scores = [float(row["final_preterminal_buy_readiness"]) for row in rows]
    maximum_scores = [float(row["maximum_preterminal_buy_readiness"]) for row in rows]
    strict_partial = [row for row in rows if row["success"] or row["failure_class"] == "partial_purchase"]
    strict_partial_labels = [int(row["success"]) for row in strict_partial]
    strict_partial_final = [float(row["final_preterminal_buy_readiness"]) for row in strict_partial]
    failures = [row for row in rows if not row["success"]]
    high_failure_count = sum(float(row["final_preterminal_buy_readiness"]) >= 0.8 for row in failures)
    classes: dict[str, int] = {}
    for row in rows:
        label = str(row["failure_class"])
        classes[label] = classes.get(label, 0) + 1
    final_auc = _auc(labels, final_scores)
    maximum_auc = _auc(labels, maximum_scores)
    strict_partial_auc = _auc(strict_partial_labels, strict_partial_final)
    high_failure_fraction = high_failure_count / len(failures) if failures else 0.0
    gates = {
        "minimum_final_strict_vs_all_auc": 0.80,
        "minimum_final_strict_vs_partial_auc": 0.70,
        "maximum_high_readiness_failure_fraction": 0.30,
        "final_strict_vs_all_auc_passed": final_auc is not None and final_auc >= 0.80,
        "final_strict_vs_partial_auc_passed": strict_partial_auc is not None and strict_partial_auc >= 0.70,
        "high_readiness_failure_fraction_passed": high_failure_fraction <= 0.30,
    }
    gates["calibration_passed"] = all(
        gates[key]
        for key in (
            "final_strict_vs_all_auc_passed",
            "final_strict_vs_partial_auc_passed",
            "high_readiness_failure_fraction_passed",
        )
    )
    return {
        "trajectory_count": len(rows),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "strict_success_count": sum(labels),
        "failure_class_counts": dict(sorted(classes.items())),
        "metrics": {
            "final_preterminal_strict_vs_all_failure_auc": final_auc,
            "maximum_preterminal_strict_vs_all_failure_auc": maximum_auc,
            "final_preterminal_strict_vs_partial_purchase_auc": strict_partial_auc,
            "high_readiness_threshold": 0.8,
            "high_readiness_failure_count": high_failure_count,
            "high_readiness_failure_fraction": high_failure_fraction,
            "mean_final_readiness_strict": statistics.fmean(score for label, score in zip(labels, final_scores) if label),
            "mean_final_readiness_failure": statistics.fmean(score for label, score in zip(labels, final_scores) if not label),
        },
        "decision": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--identity", default="sft")
    parser.add_argument("--formula-version", choices=FORMULAS, default=LEGACY_FORMULA)
    parser.add_argument("--validation-eval-root", type=Path)
    parser.add_argument("--validation-identity", default="sft")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    _require(not output.exists(), "M6 Phase2 buy-readiness report already exists")
    _require(args.identity in {"sft", "full_18_15_prefix_replay_r2"}, "M6 Phase2 readiness calibration identity drift")
    rows = [
        _trajectory_row(item, formula_version=args.formula_version)
        for item in _iter_trajectories(args.eval_root.expanduser().resolve(), args.identity)
    ]
    calibration = _dataset_metrics(rows)
    validation = None
    if args.validation_eval_root is not None:
        validation_rows = [
            _trajectory_row(item, formula_version=args.formula_version)
            for item in _iter_trajectories(
                args.validation_eval_root.expanduser().resolve(),
                args.validation_identity,
            )
        ]
        validation = _dataset_metrics(validation_rows)
    process_reward_calibration_passed = bool(calibration["decision"]["calibration_passed"]) and (
        validation is None or bool(validation["decision"]["calibration_passed"])
    )
    report = {
        "schema_version": "m6_phase2_buy_readiness_calibration_v2",
        "development_only": True,
        "training_performed": False,
        "source_eval_root": str(args.eval_root.expanduser().resolve()),
        "source_identity": args.identity,
        "formula_version": args.formula_version,
        "score_contract": {
            "public_evidence_only": True,
            "search_result_candidate_score_used": False,
            "item_weight_when_options_required": 0.5 if args.formula_version == EQUAL_BLOCK_FORMULA else 0.8,
            "selected_option_weight_when_options_required": 0.5 if args.formula_version == EQUAL_BLOCK_FORMULA else 0.2,
            "price_is_included_in_existing_item_constraint_fraction": True,
            "target_asin_or_hidden_answer_used": False,
        },
        "calibration": calibration,
        "historical_burned_external_validation": None if validation is None else {
            "source_eval_root": str(args.validation_eval_root.expanduser().resolve()),
            "source_identity": args.validation_identity,
            "used_for_formula_selection": False,
            "result": validation,
        },
        "decision": {
            "process_reward_calibration_passed": process_reward_calibration_passed,
            "requires_calibration_and_external_validation_when_provided": True,
        },
    }
    report["content_sha256"] = _self_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
