#!/usr/bin/env python3
"""Generate the final fail-closed readiness manifest without submitting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from miniwebwork.long_horizon_rl.readiness import (
    JobEvidence,
    READINESS_ROOT,
    build_readiness_manifest,
    write_readiness_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--clean-regression-job-id", required=True, type=int)
    parser.add_argument("--clean-regression-stdout", required=True)
    parser.add_argument("--clean-regression-stderr", required=True)
    parser.add_argument("--output-dir", type=Path, default=READINESS_ROOT)
    args = parser.parse_args()
    manifest = build_readiness_manifest(
        expected_git_sha=args.expected_git_sha,
        clean_regression_job=JobEvidence(
            job_id=args.clean_regression_job_id,
            purpose="final_frozen_sha_clean_cpu_regression",
            expected_state="COMPLETED",
            expected_exit_code="0:0",
            stdout_path=args.clean_regression_stdout,
            stderr_path=args.clean_regression_stderr,
        ),
    )
    path = write_readiness_manifest(manifest, args.output_dir)
    print(json.dumps({"path": str(path), "decision": manifest["decision"], "manifest_content_sha256": manifest["manifest_content_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
