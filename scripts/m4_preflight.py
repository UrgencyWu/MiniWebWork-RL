#!/usr/bin/env python3
"""Validate one M4 method/seed/phase and write its immutable run manifest."""

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
    build_m4_run_manifest,
    write_m4_run_manifest,
)
from miniwebwork.m4_algorithms import ALL_ALGORITHMS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=sorted(ALL_ALGORITHMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("train", "dev", "final_test"), required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = M4RunConfig(args.algorithm, args.seed, args.phase)
    manifest = build_m4_run_manifest(
        config,
        task_root=args.task_root,
        seed_dir=args.seed_dir,
    )
    if args.output is not None:
        write_m4_run_manifest(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
