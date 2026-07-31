#!/usr/bin/env python3
"""Build the M4 SFT train corpus and dev-only evaluation corpus."""

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
    build_m4_oracle_sft_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--dev-task-dir", type=Path, default=PROJECT_ROOT / "data" / "tasks" / "m4_rlvr_v1" / "dev")
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--train-task-limit", type=int, default=None)
    parser.add_argument("--dev-task-limit", type=int, default=None)
    args = parser.parse_args()
    result = build_m4_oracle_sft_corpus(
        args.output_dir,
        train_task_dir=args.train_task_dir,
        dev_task_dir=args.dev_task_dir,
        seed_dir=args.seed_dir,
        max_steps=args.max_steps,
        train_task_limit=args.train_task_limit,
        dev_task_limit=args.dev_task_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
