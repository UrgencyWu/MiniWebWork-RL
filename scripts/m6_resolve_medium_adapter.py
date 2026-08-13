#!/usr/bin/env python3
"""Resolve the audited final adapter of one completed M6 medium RL run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m6_mini import validate_mini_rl_audit  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_learner_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--method", choices=("multi_turn_grpo", "anchor_gigpo"), required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    audit = validate_mini_rl_audit(json.loads((root / "rl_audit.json").read_text(encoding="utf-8")))
    if audit.get("passed") is not True or audit.get("method") != args.method:
        raise ValueError("M6 medium RL audit did not authorize evaluation")
    reports = [
        validate_learner_report(path, expected_method=args.method)
        for path in sorted(root.glob("iteration_*/learner/learner_report.json"))
    ]
    matches = [item for item in reports if item["content_sha256"] == audit["learner_report_content_sha256"]]
    if len(matches) != 1:
        raise ValueError("M6 medium final learner/audit binding drift")
    report = matches[0]
    if report.get("cumulative_optimizer_updates_after") != 20:
        raise ValueError("M6 medium final adapter does not have 20 updates")
    print(report["output_adapter"])


if __name__ == "__main__":
    main()
