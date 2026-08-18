#!/usr/bin/env python3
"""Pair and summarize Phase10-C Raw35/SFT35/SFT4 environment evaluations."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402

TASK_COUNT = 96
K = 4
PAIR_BY_STAGE = {
    "dev": ("raw35", "sft35"),
    "qualification": ("sft4", "sft35"),
    "same_corpus": ("sft4", "sft35_d4"),
    "same_corpus_d35": ("sft4_d35", "sft35_d35"),
}
EXPECTED_SEEDS = {
    "dev": 20260864,
    "qualification": 20260865,
    "same_corpus": 20260867,
    "same_corpus_d35": 20260868,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("content_sha256") == _self_hash(value), f"self-hash drift: {path}")
    return value


def _failure_class(row: Mapping[str, Any]) -> str:
    if bool(row.get("success")):
        return "strict_success"
    termination = str(row.get("termination_reason", ""))
    if termination == "purchase":
        return "partial_purchase" if float(row.get("task_score", 0.0)) > 0 else "zero_score_purchase"
    if termination in {"max_model_turns", "max_environment_steps"}:
        return "horizon_exhaustion"
    turns = row.get("turns") or []
    if any(turn.get("schema_valid") is not True for turn in turns):
        return "schema_failure"
    if any((turn.get("action_result") or {}).get("success") is False for turn in turns):
        return "action_failure"
    if termination == "premature_finish":
        return "premature_finish"
    return "other_failure"


def _load_arm(root: Path, *, identity: str, stage: str) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    _require(not ({"promotion", "holdout"} & {part.casefold() for part in resolved.parts}),
             "Phase10-C evaluation touches promotion/holdout")
    report = _load_hashed(resolved / "collection_report.json")
    invocation = _load_hashed(resolved / "invocation.json")
    _require(
        report.get("complete") is True
        and report.get("development_only") is True
        and report.get("mode") == invocation.get("mode") == "phase10c_teacher_stage_evaluation"
        and report.get("phase10c_evaluation_identity")
        == invocation.get("phase10c_evaluation_identity") == identity
        and report.get("task_count") == invocation.get("task_count") == TASK_COUNT
        and report.get("trajectory_count") == TASK_COUNT * K
        and report.get("K") == invocation.get("K") == K
        and report.get("training_updates_allowed") is False
        and invocation.get("training_updates_allowed") is False,
        "Phase10-C evaluation collection contract drift",
    )
    expected_seed = EXPECTED_SEEDS[stage]
    _require(invocation.get("seed") == expected_seed, "Phase10-C evaluation seed drift")
    groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=K)
        for path in sorted((resolved / "groups").glob("g*.json"))
    ]
    _require(len(groups) == TASK_COUNT, "Phase10-C evaluation group count drift")
    _require(report.get("group_content_sha256") == [row["content_sha256"] for row in groups],
             "Phase10-C evaluation group/report drift")
    return {"root": str(resolved), "report": report, "invocation": invocation, "groups": groups}


def _bootstrap_ci(task_differences: Sequence[float], *, seed: int, draws: int = 10000) -> list[float]:
    rng = random.Random(seed)
    n = len(task_differences)
    values = sorted(
        statistics.fmean(task_differences[rng.randrange(n)] for _ in range(n)) * 100.0
        for _ in range(draws)
    )
    return [values[int(draws * 0.025)], values[int(draws * 0.975)]]


def pair_stage(
    *,
    baseline_root: Path,
    candidate_root: Path,
    baseline_identity: str,
    candidate_identity: str,
    stage: str,
) -> dict[str, Any]:
    _require(stage in PAIR_BY_STAGE, "Phase10-C evaluation stage drift")
    _require((baseline_identity, candidate_identity) == PAIR_BY_STAGE[stage],
             "Phase10-C paired identity order drift")
    baseline = _load_arm(baseline_root, identity=baseline_identity, stage=stage)
    candidate = _load_arm(candidate_root, identity=candidate_identity, stage=stage)
    for field in ("task_order_sha256", "task_roster_content_sha256", "split_lock_content_sha256", "protocol_sha256"):
        _require(baseline["report"][field] == candidate["report"][field],
                 f"Phase10-C pairing drift: {field}")
    rows = []
    seed_mismatches = 0
    task_differences = []
    positive_tasks = negative_tasks = 0
    for left_group, right_group in zip(baseline["groups"], candidate["groups"]):
        _require(left_group["task_id"] == right_group["task_id"], "Phase10-C paired task drift")
        left_rate = statistics.fmean(float(row["success"]) for row in left_group["trajectories"])
        right_rate = statistics.fmean(float(row["success"]) for row in right_group["trajectories"])
        task_differences.append(right_rate - left_rate)
        positive_tasks += int(right_rate > left_rate)
        negative_tasks += int(right_rate < left_rate)
        for left, right in zip(left_group["trajectories"], right_group["trajectories"]):
            _require(left["rollout_index"] == right["rollout_index"], "Phase10-C rollout pairing drift")
            for left_turn, right_turn in zip(left["turns"], right["turns"]):
                seed_mismatches += int(left_turn["sampling_seed"] != right_turn["sampling_seed"])
            rows.append({
                "task_id": left_group["task_id"],
                "rollout_index": left["rollout_index"],
                "baseline_success": bool(left["success"]),
                "candidate_success": bool(right["success"]),
                "baseline_class": _failure_class(left),
                "candidate_class": _failure_class(right),
            })
    _require(len(rows) == TASK_COUNT * K and seed_mismatches == 0,
             "Phase10-C paired rollout/seed drift")
    baseline_successes = sum(row["baseline_success"] for row in rows)
    candidate_successes = sum(row["candidate_success"] for row in rows)
    delta_pp = (candidate_successes - baseline_successes) / len(rows) * 100.0
    candidate_only = sum(row["candidate_success"] and not row["baseline_success"] for row in rows)
    baseline_only = sum(row["baseline_success"] and not row["candidate_success"] for row in rows)
    classes = {
        "baseline": dict(sorted(Counter(row["baseline_class"] for row in rows).items())),
        "candidate": dict(sorted(Counter(row["candidate_class"] for row in rows).items())),
    }
    common = {
        "strict_point_gain_positive": delta_pp > 0,
        "paired_task_net_flips_positive": positive_tasks > negative_tasks,
        "trajectory_net_flips_positive": candidate_only > baseline_only,
    }
    if stage == "qualification":
        gates = {
            **common,
            "strict_point_gain_at_least_5pp": delta_pp >= 5.0,
            "schema_action_failures_not_increased": (
                classes["candidate"].get("schema_failure", 0)
                + classes["candidate"].get("action_failure", 0)
                <= classes["baseline"].get("schema_failure", 0)
                + classes["baseline"].get("action_failure", 0)
            ),
            "nonstrict_purchase_not_increased": (
                classes["candidate"].get("partial_purchase", 0)
                + classes["candidate"].get("zero_score_purchase", 0)
                <= classes["baseline"].get("partial_purchase", 0)
                + classes["baseline"].get("zero_score_purchase", 0)
            ),
        }
    else:
        gates = common
    return {
        "schema_version": "m6_phase10c_teacher_stage_eval_stats_v1",
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "stage": stage,
        "baseline_identity": baseline_identity,
        "candidate_identity": candidate_identity,
        "paired_task_count": TASK_COUNT,
        "paired_trajectory_count": TASK_COUNT * K,
        "K": K,
        "baseline_strict_successes": baseline_successes,
        "candidate_strict_successes": candidate_successes,
        "baseline_strict_rate": baseline_successes / len(rows),
        "candidate_strict_rate": candidate_successes / len(rows),
        "candidate_minus_baseline_pp": delta_pp,
        "task_bootstrap_95ci_pp": _bootstrap_ci(task_differences, seed=20260866),
        "positive_task_count": positive_tasks,
        "negative_task_count": negative_tasks,
        "candidate_only_flips": candidate_only,
        "baseline_only_flips": baseline_only,
        "failure_classes": classes,
        "seed_mismatch_count": seed_mismatches,
        "gates": gates,
        "passed": all(gates.values()),
        "baseline_collection_report_content_sha256": baseline["report"]["content_sha256"],
        "candidate_collection_report_content_sha256": candidate["report"]["content_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-identity", required=True)
    parser.add_argument("--candidate-identity", required=True)
    parser.add_argument("--stage", choices=tuple(PAIR_BY_STAGE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase10-C evaluation stats output exists")
    report = pair_stage(
        baseline_root=args.baseline_root,
        candidate_root=args.candidate_root,
        baseline_identity=args.baseline_identity,
        candidate_identity=args.candidate_identity,
        stage=args.stage,
    )
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
