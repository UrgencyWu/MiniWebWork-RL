#!/usr/bin/env python3
"""Paired fresh-task Raw/SFT/RL statistics for the Phase4 quick loop."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _failure_class(trajectory: Mapping[str, Any]) -> str:
    if bool(trajectory["success"]):
        return "strict_success"
    termination = str(trajectory.get("termination_reason", ""))
    if termination == "purchase":
        return "partial_purchase" if float(trajectory.get("task_score", 0.0)) > 0.0 else "zero_match_purchase"
    if termination in {"max_model_turns", "max_environment_steps"}:
        return "horizon_exhaustion"
    return "schema_or_action_failure"


def _load_identity(roots: Sequence[Path]) -> dict[str, Any]:
    _require(len(roots) == 2, "Phase4 tuning identity requires two partitions")
    groups: list[dict[str, Any]] = []
    reports = []
    for root in roots:
        resolved = root.expanduser().resolve()
        report = json.loads((resolved / "collection_report.json").read_text(encoding="utf-8"))
        expected = dict(report)
        observed = expected.pop("content_sha256", None)
        _require(observed == sha256_json(expected), "Phase4 tuning collection hash drift")
        _require(
            report.get("complete") is True
            and report.get("mode") == "phase4_tuning_evaluation"
            and report.get("role") == "train"
            and report.get("K") == 4,
            "Phase4 tuning collection contract drift",
        )
        partition_groups = [
            validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=4)
            for path in sorted((resolved / "groups").glob("g*.json"))
        ]
        _require(report["group_content_sha256"] == [group["content_sha256"] for group in partition_groups], "Phase4 tuning group/report drift")
        reports.append(report)
        groups.extend(partition_groups)
    _require(len(groups) == 128 and len({group["task_id"] for group in groups}) == 128, "Phase4 tuning task count drift")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    task_rates: dict[str, float] = {}
    failures: Counter[str] = Counter()
    for group in groups:
        successes = []
        for trajectory in group["trajectories"]:
            key = (group["task_id"], int(trajectory["rollout_index"]))
            row = {
                "success": int(bool(trajectory["success"])),
                "task_score": float(trajectory["task_score"]),
                "failure_class": _failure_class(trajectory),
            }
            rows[key] = row
            successes.append(row["success"])
            failures[row["failure_class"]] += 1
        task_rates[group["task_id"]] = sum(successes) / 4
    return {
        "reports": reports,
        "rows": rows,
        "task_rates": task_rates,
        "strict_success_rate": statistics.fmean(row["success"] for row in rows.values()),
        "failure_counts": dict(sorted(failures.items())),
        "partial_purchase_rate": failures["partial_purchase"] / len(rows),
        "schema_or_action_failure_rate": failures["schema_or_action_failure"] / len(rows),
        "policy_lineage": [report["policy_lineage"] for report in reports],
    }


def _paired_bootstrap(left: Mapping[str, float], right: Mapping[str, float], *, seed: int = 20260831) -> dict[str, float]:
    tasks = sorted(left)
    _require(tasks == sorted(right), "Phase4 tuning bootstrap task mismatch")
    deltas = [right[task] - left[task] for task in tasks]
    rng = random.Random(seed)
    samples = [statistics.fmean(rng.choice(deltas) for _ in tasks) for _ in range(10000)]
    samples.sort()
    return {
        "mean": statistics.fmean(deltas),
        "ci95_low": samples[249],
        "ci95_high": samples[9749],
        "positive_task_fraction": sum(value > 0 for value in deltas) / len(deltas),
        "negative_task_fraction": sum(value < 0 for value in deltas) / len(deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for identity in ("raw", "sft", "rl"):
        parser.add_argument(f"--{identity}-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identities = {
        identity: _load_identity(getattr(args, f"{identity}_root"))
        for identity in ("raw", "sft", "rl")
    }
    keys = {identity: set(value["rows"]) for identity, value in identities.items()}
    _require(keys["raw"] == keys["sft"] == keys["rl"], "Phase4 tuning rollout keys are not paired")
    task_ids = {identity: set(value["task_rates"]) for identity, value in identities.items()}
    _require(task_ids["raw"] == task_ids["sft"] == task_ids["rl"], "Phase4 tuning tasks are not paired")
    rates = {identity: value["strict_success_rate"] for identity, value in identities.items()}
    sft_rows = identities["sft"]["rows"]
    rl_rows = identities["rl"]["rows"]
    rl_only = sum(rl_rows[key]["success"] == 1 and sft_rows[key]["success"] == 0 for key in sft_rows)
    sft_only = sum(rl_rows[key]["success"] == 0 and sft_rows[key]["success"] == 1 for key in sft_rows)
    paired = _paired_bootstrap(identities["sft"]["task_rates"], identities["rl"]["task_rates"])
    decision = {
        "raw_strictly_below_sft": rates["raw"] < rates["sft"],
        "sft_strictly_below_rl": rates["sft"] < rates["rl"],
        "rl_minus_sft_at_least_1pp": rates["rl"] - rates["sft"] >= 0.01,
        "rl_only_flips_exceed_sft_only": rl_only > sft_only,
        "partial_purchase_not_worse_by_more_than_0_5pp": identities["rl"]["partial_purchase_rate"] <= identities["sft"]["partial_purchase_rate"] + 0.005,
        "schema_or_action_not_worse_by_more_than_0_5pp": identities["rl"]["schema_or_action_failure_rate"] <= identities["sft"]["schema_or_action_failure_rate"] + 0.005,
    }
    decision["raw_sft_rl_chain_passed"] = all(decision.values())
    report = {
        "schema_version": "m6_phase4_tuning_stats_v1",
        "complete": True,
        "task_count": 128,
        "trajectory_count_per_identity": 512,
        "K": 4,
        "max_model_turns": 18,
        "max_environment_steps": 15,
        "strict_success_rates": rates,
        "raw_to_sft_delta": rates["sft"] - rates["raw"],
        "sft_to_rl_delta": rates["rl"] - rates["sft"],
        "paired_sft_to_rl_task_bootstrap": paired,
        "trajectory_flips": {"rl_only": rl_only, "sft_only": sft_only, "net": rl_only - sft_only},
        "failure_counts": {identity: value["failure_counts"] for identity, value in identities.items()},
        "partial_purchase_rates": {identity: value["partial_purchase_rate"] for identity, value in identities.items()},
        "schema_or_action_failure_rates": {identity: value["schema_or_action_failure_rate"] for identity, value in identities.items()},
        "policy_lineage": {identity: value["policy_lineage"] for identity, value in identities.items()},
        "decision": decision,
    }
    report["content_sha256"] = sha256_json(report)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

