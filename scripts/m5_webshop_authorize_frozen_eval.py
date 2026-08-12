#!/usr/bin/env python3
"""Authorize exactly the eight frozen M5 evaluation identities after approval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json  # noqa: E402
from miniwebwork.webshop_rl.formal_evaluation import (  # noqa: E402
    AUTHORIZATION_SCHEMA,
    IDENTITIES,
    load_eval_plan,
    validate_all_inference_identities,
)
from miniwebwork.webshop_rl.formal_training import self_hash  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--confirm", required=True, help="must equal I_APPROVE_EIGHT_FROZEN_EVALS")
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--eval-plan", type=Path, default=PROJECT_ROOT / "data" / "m5_webshop_frozen_eval_plan_v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "readiness" / "frozen_eval_authorization_v1.json")
    args = parser.parse_args()
    _require(args.confirm == "I_APPROVE_EIGHT_FROZEN_EVALS", "explicit eight-evaluation confirmation is missing")
    _require(args.approved_by.strip() and args.approval_record.strip(), "approval identity or record is empty")
    git_sha = _git_sha()
    _require(git_sha == args.expected_git_sha, "M5 frozen evaluation authorization Git drift")
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    _require(not status, "M5 frozen evaluation authorization requires a clean tracked worktree")
    plan = load_eval_plan(args.eval_plan)
    identity_audit = validate_all_inference_identities(plan["payload"], verify_base_model_files=True)
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "M5 frozen evaluation authorization artifact already exists")
    report = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "study_id": plan["payload"]["study_id"],
        "frozen_evaluation_submission_allowed": True,
        "approval_scope": "eight_frozen_evaluation_identities_only",
        "consumer_git_sha": git_sha,
        "producer_training_git_sha": plan["payload"]["producer_training_git_sha"],
        "eval_plan_sha256": plan["sha256"],
        "identities": list(IDENTITIES),
        "logical_job_count": 8,
        "identity_audit": identity_audit,
        "approved_by": args.approved_by.strip(),
        "approval_record": args.approval_record.strip(),
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "exclusions": ["model updates", "checkpoint selection", "additional algorithms", "SFT rerun", "automatic result analysis before all identities finish"],
    }
    report["content_sha256"] = self_hash(report)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
