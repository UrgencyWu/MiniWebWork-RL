#!/usr/bin/env python3
"""Summarize one immutable M6 phase-one K8 seen-task diagnostic identity."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collection = json.loads(args.collection_report.read_text(encoding="utf-8"))
    expected = dict(collection)
    observed = expected.pop("content_sha256", None)
    if observed != sha256_json(expected):
        raise ValueError("M6 seen-task collection self-hash drift")
    if collection.get("mode") != "diagnostic_evaluation" or collection.get("role") != "mini_train":
        raise ValueError("M6 seen-task collection purpose drift")
    invocation = json.loads((args.collection_report.parent / "invocation.json").read_text(encoding="utf-8"))
    invocation_expected = dict(invocation)
    invocation_observed = invocation_expected.pop("content_sha256", None)
    if invocation_observed != sha256_json(invocation_expected):
        raise ValueError("M6 seen-task invocation self-hash drift")
    if collection.get("invocation_content_sha256") != invocation.get("content_sha256"):
        raise ValueError("M6 seen-task invocation/report drift")
    if invocation.get("seed") != 20260818 or invocation.get("K") != 8:
        raise ValueError("M6 seen-task sampling contract drift")
    groups = [validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=8) for path in sorted(args.groups_dir.glob("g*.json"))]
    if collection.get("group_content_sha256") != [group["content_sha256"] for group in groups]:
        raise ValueError("M6 seen-task collection/group drift")
    rows = {}
    for group in groups:
        successes = [float(trajectory["success"]) for trajectory in group["trajectories"]]
        dense = [float(trajectory["task_score"]) for trajectory in group["trajectories"]]
        rows[group["task_id"]] = {
            "task_id": group["task_id"],
            "strict_success_rate": statistics.fmean(successes),
            "dense_score": statistics.fmean(dense),
        }
    report = {
        "schema_version": "m6_phase1_seen_task_identity_v1",
        "development_only": True,
        "formal_training": False,
        "label": args.label,
        "task_count": len(rows),
        "trajectory_count": len(rows) * 8,
        "K": 8,
        "rollout_seed": invocation["seed"],
        "strict_success_rate": statistics.fmean(row["strict_success_rate"] for row in rows.values()),
        "dense_score": statistics.fmean(row["dense_score"] for row in rows.values()),
        "task_rows": rows,
        "collection_report_content_sha256": collection["content_sha256"],
        "policy_lineage": collection["policy_lineage"],
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
