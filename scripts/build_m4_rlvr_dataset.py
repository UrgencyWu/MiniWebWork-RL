#!/usr/bin/env python3
"""Build and validate the checked M4 RLVR task worlds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.data_generation.m4_rlvr import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED_DIR,
    build_m4_rlvr_dataset,
    validate_m4_rlvr_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        build_m4_rlvr_dataset(args.output_dir, seed_dir=args.seed_dir)
    result = validate_m4_rlvr_dataset(args.output_dir, seed_dir=args.seed_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
