#!/usr/bin/env python3
"""Generate the narrow M5 authorization artifact after explicit user approval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json
from miniwebwork.webshop_rl.formal_training import (
    AUTHORIZATION_SCHEMA,
    load_formal_plan,
    load_readiness,
    self_hash,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--confirm", required=True, help="must equal I_APPROVE_SIX_ONLINE_RUNS")
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--formal-plan", type=Path, default=PROJECT_ROOT / "data" / "m5_webshop_formal_plan_v1.json")
    parser.add_argument("--readiness", type=Path, default=PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "readiness" / "readiness_manifest_v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "readiness" / "formal_authorization_v1.json")
    args = parser.parse_args()
    _require(args.confirm == "I_APPROVE_SIX_ONLINE_RUNS", "explicit six-run confirmation is missing")
    _require(args.approved_by.strip() and args.approval_record.strip(), "approval identity or record is empty")
    git_sha = _git_sha()
    _require(git_sha == args.expected_git_sha, "M5 authorization Git SHA drift")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(not status, "M5 authorization requires a clean tracked worktree")
    plan = load_formal_plan(args.formal_plan)
    readiness = load_readiness(args.readiness, git_sha=git_sha, plan_sha256=plan["sha256"])
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "M5 authorization artifact already exists")
    report = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "study_id": plan["payload"]["study_id"],
        "formal_submission_allowed": True,
        "approval_scope": "six_online_runs_only",
        "git_sha": git_sha,
        "formal_plan_sha256": plan["sha256"],
        "readiness_path": str(Path(readiness["path"])),
        "readiness_sha256": readiness["sha256"],
        "methods": plan["payload"]["matrix"]["methods"],
        "seeds": plan["payload"]["matrix"]["seeds"],
        "logical_run_count": 6,
        "approved_by": args.approved_by.strip(),
        "approval_record": args.approval_record.strip(),
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "exclusions": ["SFT rerun", "frozen test", "final analysis", "additional algorithms"],
    }
    report["content_sha256"] = self_hash(report)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
