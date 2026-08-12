#!/usr/bin/env python3
"""Generate the final M5 WebShop statistical and failure-analysis report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.webshop_rl.final_analysis import build_final_analysis  # noqa: E402


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "m5_webshop_credit_assignment_v1"
        / "formal"
        / "frozen_test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "m5_webshop_credit_assignment_v1"
        / "formal"
        / "analysis",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=50_000)
    parser.add_argument("--expected-git-sha")
    args = parser.parse_args()
    git_sha = _git_sha()
    if args.expected_git_sha and args.expected_git_sha != git_sha:
        raise ValueError("M5 final analysis Git SHA drift")
    result = build_final_analysis(
        eval_root=args.eval_root,
        output_dir=args.output_dir,
        analysis_git_sha=git_sha,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "report"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
