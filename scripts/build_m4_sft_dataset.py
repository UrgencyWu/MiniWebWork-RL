#!/usr/bin/env python3
"""Build canonical oracle-SFT examples from the M4 train split only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.sft.m4_dataset import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_DIR,
    build_m4_oracle_sft_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--task-limit", type=int, default=None)
    args = parser.parse_args()
    result = build_m4_oracle_sft_dataset(
        args.output_dir,
        task_dir=args.task_dir,
        seed_dir=args.seed_dir,
        max_steps=args.max_steps,
        task_limit=args.task_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
