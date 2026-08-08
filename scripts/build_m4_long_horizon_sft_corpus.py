#!/usr/bin/env python3
"""Build unique verified train/dev SFT turn evidence for the focused study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.data_generation.m4_long_horizon import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TASK_ROOT,
    DEFAULT_SEED_DIR,
)
from miniwebwork.sft.m4_long_horizon_dataset import (
    DEFAULT_OUTPUT_DIR,
    build_verified_sft_corpus,
    validate_verified_sft_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--train-task-limit", type=int, default=None)
    parser.add_argument("--dev-task-limit", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        if args.train_task_limit is not None or args.dev_task_limit is not None:
            raise ValueError("--validate-only does not accept task limits")
        result = validate_verified_sft_corpus(
            args.output_dir,
            task_root=args.task_root,
            seed_dir=args.seed_dir,
            require_full_roster=True,
        )
    else:
        manifest = build_verified_sft_corpus(
            args.output_dir,
            task_root=args.task_root,
            seed_dir=args.seed_dir,
            max_steps=args.max_steps,
            train_task_limit=args.train_task_limit,
            dev_task_limit=args.dev_task_limit,
        )
        validation = validate_verified_sft_corpus(
            args.output_dir,
            task_root=args.task_root,
            seed_dir=args.seed_dir,
            require_full_roster=(
                args.train_task_limit is None and args.dev_task_limit is None
            ),
        )
        result = {"manifest": manifest, "validation": validation}
    if not result.get("valid", result.get("validation", {}).get("valid", False)):
        raise RuntimeError(f"verified SFT corpus validation failed: {result}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
