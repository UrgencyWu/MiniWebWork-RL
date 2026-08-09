#!/usr/bin/env python3
"""Compute formal readiness-v2 from raw evidence; never submit a job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.readiness import JobEvidence, READINESS_ROOT
from miniwebwork.long_horizon_rl.readiness_v2 import build_readiness_v2, write_readiness_v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--clean-regression-job-id", required=True, type=int)
    parser.add_argument("--clean-regression-stdout", required=True)
    parser.add_argument("--clean-regression-stderr", required=True)
    parser.add_argument("--output-dir", type=Path, default=READINESS_ROOT)
    args = parser.parse_args()
    manifest = build_readiness_v2(
        expected_git_sha=args.expected_git_sha,
        clean_regression_job=JobEvidence(
            job_id=args.clean_regression_job_id,
            purpose="formal_v1_final_frozen_sha_clean_cpu_regression",
            expected_state="COMPLETED",
            expected_exit_code="0:0",
            stdout_path=args.clean_regression_stdout,
            stderr_path=args.clean_regression_stderr,
        ),
    )
    path = write_readiness_v2(manifest, args.output_dir)
    print(json.dumps({"path": str(path), "decision": manifest["decision"], "manifest_content_sha256": manifest["manifest_content_sha256"]}, indent=2))
    return 0 if manifest["decision"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
