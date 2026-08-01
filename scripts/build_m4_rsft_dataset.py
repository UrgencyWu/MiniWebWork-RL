#!/usr/bin/env python3
"""Create the audited M4 RSFT tokenized train corpus from two rollout passes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_protocol import (
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4RunConfig,
    M4_TRAIN_TASK_COUNT,
    m4_rsft_train_task_roster,
)
from miniwebwork.sft.m4_rsft_dataset import DEFAULT_OUTPUT_DIR, build_m4_rsft_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    args = parser.parse_args()
    task_root = args.task_root.expanduser().resolve()
    seed_dir = args.seed_dir.expanduser().resolve()
    M4RunConfig("rsft", args.seed, "train").validate(task_root=task_root)
    result = build_m4_rsft_dataset(
        args.artifacts,
        args.output_dir,
        seed=args.seed,
        expected_task_ids=m4_rsft_train_task_roster(task_root, args.seed),
        expected_task_universe_count=M4_TRAIN_TASK_COUNT,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
