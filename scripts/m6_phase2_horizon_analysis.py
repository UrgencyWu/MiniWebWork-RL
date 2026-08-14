#!/usr/bin/env python3
"""Audit and compare paired short/full-horizon SFT rollouts for M6 Phase2."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _failure_class(trajectory: Mapping[str, Any]) -> str:
    if trajectory["success"]:
        return "strict_success"
    termination = str(trajectory.get("termination_reason", ""))
    score = float(trajectory.get("task_score", 0.0))
    if termination == "purchase" and score > 0:
        return "partial_purchase"
    if termination == "purchase":
        return "zero_match_purchase"
    if termination in {"max_environment_steps", "max_model_turns"}:
        return "horizon_exhaustion"
    if any(turn.get("schema_valid") is not True for turn in trajectory["turns"]):
        return "schema_failure"
    return "other_failure"


def _identity(root: Path, expected_horizon: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    report = _load(root / "collection_report.json")
    invocation = _load(root / "invocation.json")
    _require(report["content_sha256"] == _self_hash(report), "M6 Phase2 horizon collection self-hash drift")
    _require(invocation["content_sha256"] == _self_hash(invocation), "M6 Phase2 horizon invocation self-hash drift")
    _require(report["invocation_content_sha256"] == invocation["content_sha256"], "M6 Phase2 horizon invocation/report drift")
    _require(report["mode"] == invocation["mode"] == "phase2_horizon_evaluation", "M6 Phase2 horizon mode drift")
    _require(report["role"] == invocation["role"] == "train", "M6 Phase2 horizon role drift")
    _require(report["K"] == invocation["K"] == 4 and report["task_count"] == invocation["task_count"] == 64, "M6 Phase2 horizon size drift")
    _require(
        (invocation["max_model_turns"], invocation["max_environment_steps"]) == expected_horizon,
        "M6 Phase2 horizon arm mismatch",
    )
    groups = [
        validate_committed_group(_load(path), require_k=4)
        for path in sorted((root / "groups").glob("g*.json"))
    ]
    _require(len(groups) == 64, "M6 Phase2 horizon group matrix incomplete")
    _require(report["group_content_sha256"] == [group["content_sha256"] for group in groups], "M6 Phase2 horizon group/report drift")
    return report, invocation, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    _require(not output.exists(), "M6 Phase2 horizon analysis already exists")
    short_report, short_invocation, short_groups = _identity(args.short_root.expanduser().resolve(), (6, 6))
    full_report, full_invocation, full_groups = _identity(args.full_root.expanduser().resolve(), (18, 15))
    for field in (
        "task_order_sha256",
        "task_roster_content_sha256",
        "task_roster_producer_git_sha",
        "task_roster_consumer_git_sha",
        "split_lock_content_sha256",
        "protocol_sha256",
        "policy_lineage",
    ):
        _require(short_report[field] == full_report[field], f"M6 Phase2 horizon pairing drift: {field}")
    _require(short_invocation["seed"] == full_invocation["seed"] == 20260821, "M6 Phase2 horizon seed drift")
    rows = []
    shared_turn_seed_mismatches = 0
    shared_turn_token_mismatches = 0
    for short_group, full_group in zip(short_groups, full_groups):
        _require(short_group["task_id"] == full_group["task_id"], "M6 Phase2 horizon task pairing drift")
        for short, full in zip(short_group["trajectories"], full_group["trajectories"]):
            _require(short["rollout_index"] == full["rollout_index"], "M6 Phase2 horizon rollout pairing drift")
            for left, right in zip(short["turns"], full["turns"]):
                shared_turn_seed_mismatches += int(left["sampling_seed"] != right["sampling_seed"])
                shared_turn_token_mismatches += int(left["generated_token_sha256"] != right["generated_token_sha256"])
            rows.append({
                "task_id": short_group["task_id"],
                "rollout_index": short["rollout_index"],
                "short_success": bool(short["success"]),
                "full_success": bool(full["success"]),
                "short_failure_class": _failure_class(short),
                "full_failure_class": _failure_class(full),
                "short_tokens": int(short["generated_action_tokens"]),
                "full_tokens": int(full["generated_action_tokens"]),
                "short_steps": int(short["environment_steps"]),
                "full_steps": int(full["environment_steps"]),
            })
    _require(len(rows) == 256, "M6 Phase2 horizon paired trajectory matrix incomplete")
    _require(shared_turn_seed_mismatches == 0, "M6 Phase2 horizon shared-turn sampling seeds diverged")
    _require(shared_turn_token_mismatches == 0, "M6 Phase2 horizon shared-prefix actions diverged")
    short_rate = statistics.fmean(float(row["short_success"]) for row in rows)
    full_rate = statistics.fmean(float(row["full_success"]) for row in rows)
    short_horizon = [row for row in rows if row["short_failure_class"] == "horizon_exhaustion"]
    converted = sum(bool(row["full_success"]) for row in short_horizon)
    conversion = converted / len(short_horizon) if short_horizon else 0.0
    delta_pp = (full_rate - short_rate) * 100.0
    if delta_pp >= 1.0 or conversion >= 0.20:
        decision = "USE_FULL_HORIZON"
    elif delta_pp < 0.5 and conversion < 0.10:
        decision = "HORIZON_SECONDARY"
    else:
        decision = "INCONCLUSIVE_RETAIN_FULL_ARM"
    classes = {}
    for arm in ("short", "full"):
        values = {}
        for row in rows:
            label = row[f"{arm}_failure_class"]
            values[label] = values.get(label, 0) + 1
        classes[arm] = dict(sorted(values.items()))
    report = {
        "schema_version": "m6_phase2_horizon_analysis_v1",
        "development_only": True,
        "training_performed": False,
        "paired_task_count": 64,
        "paired_trajectory_count": 256,
        "rollout_seed": 20260821,
        "shared_turn_sampling_seed_mismatch_count": shared_turn_seed_mismatches,
        "shared_turn_generated_token_mismatch_count": shared_turn_token_mismatches,
        "short_strict_success_rate": short_rate,
        "full_strict_success_rate": full_rate,
        "full_minus_short_strict_delta_pp": delta_pp,
        "short_horizon_failure_count": len(short_horizon),
        "short_horizon_to_full_strict_count": converted,
        "short_horizon_to_full_strict_fraction": conversion,
        "failure_class_counts": classes,
        "cost": {
            "short_mean_generated_action_tokens": statistics.fmean(row["short_tokens"] for row in rows),
            "full_mean_generated_action_tokens": statistics.fmean(row["full_tokens"] for row in rows),
            "short_mean_environment_steps": statistics.fmean(row["short_steps"] for row in rows),
            "full_mean_environment_steps": statistics.fmean(row["full_steps"] for row in rows),
        },
        "decision_thresholds": {
            "use_full_minimum_strict_delta_pp": 1.0,
            "use_full_minimum_horizon_conversion_fraction": 0.20,
            "secondary_maximum_strict_delta_pp": 0.5,
            "secondary_maximum_horizon_conversion_fraction": 0.10,
        },
        "decision": decision,
        "source": {
            "short_collection_report_content_sha256": short_report["content_sha256"],
            "full_collection_report_content_sha256": full_report["content_sha256"],
            "task_roster_content_sha256": short_report["task_roster_content_sha256"],
            "policy_lineage": short_report["policy_lineage"],
        },
    }
    report["content_sha256"] = _self_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
