#!/usr/bin/env python3
"""Create the audited M4 RSFT tokenized train corpus from two rollout passes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.sft.m4_rsft_dataset import DEFAULT_OUTPUT_DIR, build_m4_rsft_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = build_m4_rsft_dataset(args.artifacts, args.output_dir, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
