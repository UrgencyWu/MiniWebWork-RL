#!/usr/bin/env python3
"""Validate all seven frozen evaluations and build the final statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.formal_analysis import FORMAL_ANALYSIS_ROOT, build_formal_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FORMAL_ANALYSIS_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument("--sacct-path", type=Path, default=Path("/opt/slurm/slurm.25.05/bin/sacct"))
    args = parser.parse_args()
    result = build_formal_analysis(
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        sacct_path=args.sacct_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
