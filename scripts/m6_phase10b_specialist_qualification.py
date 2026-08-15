#!/usr/bin/env python3
"""Audit paired Student/Specialist K4 qualification and freeze the OPD admission gate."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m6_phase7_terminal_contrast_diagnostic import _prebuy_public_choice  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402

IDENTITIES = ("S_nav", "S_match", "S_finish")
TASK_COUNT = 24
K = 4
SEED = 20260851
MINIMUM_GAIN_PP = 5.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("content_sha256") == _self_hash(value), f"Phase10-B self-hash drift: {path}")
    return value


def _failure_class(trajectory: Mapping[str, Any]) -> str:
    if bool(trajectory.get("success")):
        return "strict_success"
    termination = str(trajectory.get("termination_reason", ""))
    if termination == "purchase":
        return "partial_purchase" if float(trajectory.get("task_score", 0.0)) > 0 else "zero_score_purchase"
    if termination in {"max_model_turns", "max_environment_steps"}:
        return "horizon_exhaustion"
    turns = trajectory.get("turns") or []
    if any(turn.get("schema_valid") is not True for turn in turns):
        return "schema_failure"
    if any((turn.get("action_result") or {}).get("success") is False for turn in turns):
        return "action_failure"
    if termination == "premature_finish":
        return "premature_finish"
    return "other_failure"


def _load_arm(root: Path, *, expected_identity: str) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    _require(not ({"promotion", "holdout"} & {part.casefold() for part in resolved.parts}),
             "Phase10-B qualification touches promotion/holdout")
    report = _load_hashed(resolved / "collection_report.json")
    invocation = _load_hashed(resolved / "invocation.json")
    _require(
        report.get("complete") is True
        and report.get("development_only") is True
        and report.get("mode") == invocation.get("mode") == "phase10b_specialist_qualification"
        and report.get("role") == invocation.get("role") == "train"
        and report.get("K") == invocation.get("K") == K
        and report.get("task_count") == invocation.get("task_count") == TASK_COUNT
        and report.get("trajectory_count") == TASK_COUNT * K
        and report.get("training_updates_allowed") is False
        and invocation.get("training_updates_allowed") is False,
        "Phase10-B qualification collection contract drift",
    )
    _require(
        report.get("phase10b_qualification_identity")
        == invocation.get("phase10b_qualification_identity")
        == expected_identity,
        "Phase10-B qualification identity drift",
    )
    _require(
        invocation.get("seed") == SEED
        and (invocation.get("max_model_turns"), invocation.get("max_environment_steps")) == (18, 15),
        "Phase10-B qualification seed/horizon drift",
    )
    groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=K)
        for path in sorted((resolved / "groups").glob("g*.json"))
    ]
    _require(len(groups) == TASK_COUNT, "Phase10-B qualification group count drift")
    _require(report.get("group_content_sha256") == [group["content_sha256"] for group in groups],
             "Phase10-B qualification report/group drift")
    _require(len({group["task_id"] for group in groups}) == TASK_COUNT, "Phase10-B qualification duplicate task")
    return {"root": str(resolved), "report": report, "invocation": invocation, "groups": groups}


def _task_rates(groups: list[dict[str, Any]]) -> dict[str, float]:
    return {
        group["task_id"]: statistics.fmean(float(row["success"]) for row in group["trajectories"])
        for group in groups
    }


def qualification_decision(
    *,
    identity: str,
    delta_pp: float,
    positive_tasks: int,
    negative_tasks: int,
    classes: Mapping[str, Mapping[str, int]],
    same_item_option_positive_flips: int,
    finish_purchase_positive_flips: int,
    finish_recovery_positive_flips: int,
) -> dict[str, bool]:
    """Apply the frozen per-Specialist performance and safety gates."""

    _require(identity in IDENTITIES, "Phase10-B Specialist identity drift")
    common = {
        "strict_point_gain_at_least_5pp": delta_pp >= MINIMUM_GAIN_PP,
        "paired_task_net_flips_positive": positive_tasks > negative_tasks,
    }
    student_classes = classes["student"]
    specialist_classes = classes["specialist"]
    if identity == "S_nav":
        specific = {
            "schema_action_failures_not_increased": (
                specialist_classes.get("schema_failure", 0)
                + specialist_classes.get("action_failure", 0)
                <= student_classes.get("schema_failure", 0)
                + student_classes.get("action_failure", 0)
            ),
        }
    elif identity == "S_match":
        specific = {
            "same_item_option_positive_flips_present": same_item_option_positive_flips > 0,
            "zero_score_purchase_not_increased": (
                specialist_classes.get("zero_score_purchase", 0)
                <= student_classes.get("zero_score_purchase", 0)
            ),
        }
    else:
        specific = {
            "recovery_or_purchase_positive_flips_present": (
                finish_recovery_positive_flips + finish_purchase_positive_flips > 0
            ),
            "nonstrict_purchase_not_increased": (
                specialist_classes.get("partial_purchase", 0)
                + specialist_classes.get("zero_score_purchase", 0)
                <= student_classes.get("partial_purchase", 0)
                + student_classes.get("zero_score_purchase", 0)
            ),
        }
    decision = {**common, **specific}
    decision["specialist_qualified"] = all(decision.values())
    return decision


def pair_report(*, student_root: Path, specialist_root: Path, identity: str) -> dict[str, Any]:
    _require(identity in IDENTITIES, "Phase10-B Specialist identity drift")
    student = _load_arm(student_root, expected_identity="student")
    specialist = _load_arm(specialist_root, expected_identity=identity)
    for field in (
        "task_order_sha256",
        "split_lock_content_sha256",
        "protocol_sha256",
    ):
        _require(student["report"][field] == specialist["report"][field],
                 f"Phase10-B qualification pairing drift: {field}")
    _require(student["invocation"]["adapter"] is not None, "Phase10-B Student arm is not pi_0")
    _require(specialist["invocation"]["adapter"] is None, "Phase10-B Specialist arm unexpectedly has adapter")

    rows = []
    seed_mismatches = 0
    transition_counts: Counter[str] = Counter()
    same_item_option_positive_flips = 0
    finish_purchase_positive_flips = 0
    finish_recovery_positive_flips = 0
    for student_group, specialist_group in zip(student["groups"], specialist["groups"]):
        _require(student_group["task_id"] == specialist_group["task_id"], "Phase10-B paired task drift")
        for left, right in zip(student_group["trajectories"], specialist_group["trajectories"]):
            _require(left["rollout_index"] == right["rollout_index"], "Phase10-B paired rollout-index drift")
            for left_turn, right_turn in zip(left["turns"], right["turns"]):
                seed_mismatches += int(left_turn["sampling_seed"] != right_turn["sampling_seed"])
            left_class = _failure_class(left)
            right_class = _failure_class(right)
            transition_counts[f"{left_class}->{right_class}"] += 1
            specialist_only = bool(right["success"]) and not bool(left["success"])
            if specialist_only:
                left_choice = _prebuy_public_choice(left)
                right_choice = _prebuy_public_choice(right)
                if (
                    left_class in {"partial_purchase", "zero_score_purchase"}
                    and left_choice is not None
                    and right_choice is not None
                    and left_choice["asin"] == right_choice["asin"]
                    and left_choice["selected_options"] != right_choice["selected_options"]
                ):
                    same_item_option_positive_flips += 1
                if left_class in {"partial_purchase", "zero_score_purchase"}:
                    finish_purchase_positive_flips += 1
                elif left_class in {
                    "horizon_exhaustion", "schema_failure", "action_failure", "premature_finish", "other_failure"
                }:
                    finish_recovery_positive_flips += 1
            rows.append({
                "task_id": student_group["task_id"],
                "rollout_index": int(left["rollout_index"]),
                "student_success": bool(left["success"]),
                "specialist_success": bool(right["success"]),
                "student_failure_class": left_class,
                "specialist_failure_class": right_class,
                "student_tokens": int(left["generated_action_tokens"]),
                "specialist_tokens": int(right["generated_action_tokens"]),
                "student_steps": int(left["environment_steps"]),
                "specialist_steps": int(right["environment_steps"]),
            })
    _require(len(rows) == TASK_COUNT * K and seed_mismatches == 0,
             "Phase10-B qualification rollout pairing/seed drift")
    student_rate = statistics.fmean(float(row["student_success"]) for row in rows)
    specialist_rate = statistics.fmean(float(row["specialist_success"]) for row in rows)
    delta_pp = (specialist_rate - student_rate) * 100.0
    specialist_only = sum(row["specialist_success"] and not row["student_success"] for row in rows)
    student_only = sum(row["student_success"] and not row["specialist_success"] for row in rows)
    student_task_rates = _task_rates(student["groups"])
    specialist_task_rates = _task_rates(specialist["groups"])
    positive_tasks = sum(specialist_task_rates[task] > student_task_rates[task] for task in student_task_rates)
    negative_tasks = sum(specialist_task_rates[task] < student_task_rates[task] for task in student_task_rates)
    classes = {
        "student": dict(sorted(Counter(row["student_failure_class"] for row in rows).items())),
        "specialist": dict(sorted(Counter(row["specialist_failure_class"] for row in rows).items())),
    }
    decision = qualification_decision(
        identity=identity,
        delta_pp=delta_pp,
        positive_tasks=positive_tasks,
        negative_tasks=negative_tasks,
        classes=classes,
        same_item_option_positive_flips=same_item_option_positive_flips,
        finish_purchase_positive_flips=finish_purchase_positive_flips,
        finish_recovery_positive_flips=finish_recovery_positive_flips,
    )
    return {
        "schema_version": "m6_phase10b_specialist_qualification_pair_v1",
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "identity": identity,
        "paired_task_count": TASK_COUNT,
        "paired_trajectory_count": TASK_COUNT * K,
        "K": K,
        "seed": SEED,
        "sampling_seed_mismatch_count": seed_mismatches,
        "student_strict_rate": student_rate,
        "specialist_strict_rate": specialist_rate,
        "specialist_minus_student_strict_delta_pp": delta_pp,
        "pass_at_1": {
            "student": statistics.fmean(float(row["student_success"]) for row in rows if row["rollout_index"] == 0),
            "specialist": statistics.fmean(float(row["specialist_success"]) for row in rows if row["rollout_index"] == 0),
        },
        "pass_at_4_task_fraction": {
            "student": sum(rate > 0 for rate in student_task_rates.values()) / TASK_COUNT,
            "specialist": sum(rate > 0 for rate in specialist_task_rates.values()) / TASK_COUNT,
        },
        "trajectory_flips": {
            "specialist_only": specialist_only,
            "student_only": student_only,
            "net": specialist_only - student_only,
        },
        "task_rate_flips": {"positive": positive_tasks, "negative": negative_tasks, "net": positive_tasks - negative_tasks},
        "failure_transition_counts": dict(sorted(transition_counts.items())),
        "failure_class_counts": classes,
        "specialized_positive_flips": {
            "same_item_option": same_item_option_positive_flips,
            "finish_purchase": finish_purchase_positive_flips,
            "finish_recovery": finish_recovery_positive_flips,
        },
        "cost": {
            "student_mean_tokens": statistics.fmean(row["student_tokens"] for row in rows),
            "specialist_mean_tokens": statistics.fmean(row["specialist_tokens"] for row in rows),
            "student_mean_steps": statistics.fmean(row["student_steps"] for row in rows),
            "specialist_mean_steps": statistics.fmean(row["specialist_steps"] for row in rows),
        },
        "source": {
            "student_root": student["root"],
            "specialist_root": specialist["root"],
            "student_collection_report_sha256": student["report"]["content_sha256"],
            "specialist_collection_report_sha256": specialist["report"]["content_sha256"],
            "student_task_roster_content_sha256": student["report"]["task_roster_content_sha256"],
            "specialist_task_roster_content_sha256": specialist["report"]["task_roster_content_sha256"],
            "student_policy_lineage": student["report"]["policy_lineage"],
            "specialist_policy_lineage": specialist["report"]["policy_lineage"],
        },
        "decision": decision,
    }


def aggregate_reports(paths: list[Path], *, alignment_report: Path) -> dict[str, Any]:
    alignment = _load_hashed(alignment_report.expanduser().resolve())
    _require(
        alignment.get("all_models_pass") is True
        or alignment.get("decision", {}).get("all_models_pass") is True,
             "Phase10-B logits alignment did not pass")
    reports = [_load_hashed(path.expanduser().resolve()) for path in paths]
    _require([report.get("identity") for report in reports] == list(IDENTITIES),
             "Phase10-B qualification aggregate identity order drift")
    qualified = [report["identity"] for report in reports if report["decision"]["specialist_qualified"]]
    return {
        "schema_version": "m6_phase10b_specialist_qualification_v1",
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "alignment_report_content_sha256": alignment["content_sha256"],
        "pair_report_content_sha256": [report["content_sha256"] for report in reports],
        "qualified_specialists": qualified,
        "qualified_specialist_count": len(qualified),
        "decision": {
            "multi_specialist_opd_feasible": len(qualified) >= 2,
            "single_specialist_opd_feasible": len(qualified) == 1,
            "stop_opd": len(qualified) == 0,
        },
    }


def _write(output: Path, report: dict[str, Any]) -> None:
    destination = output.expanduser().resolve()
    _require(not destination.exists(), "Phase10-B qualification report already exists")
    report["content_sha256"] = _self_hash(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pair = subparsers.add_parser("pair")
    pair.add_argument("--identity", choices=IDENTITIES, required=True)
    pair.add_argument("--student-root", type=Path, required=True)
    pair.add_argument("--specialist-root", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--pair-report", type=Path, action="append", required=True)
    aggregate.add_argument("--alignment-report", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pair":
        _write(args.output, pair_report(
            student_root=args.student_root,
            specialist_root=args.specialist_root,
            identity=args.identity,
        ))
    else:
        _require(len(args.pair_report) == len(IDENTITIES), "Phase10-B aggregate requires three pair reports")
        _write(args.output, aggregate_reports(args.pair_report, alignment_report=args.alignment_report))


if __name__ == "__main__":
    main()
