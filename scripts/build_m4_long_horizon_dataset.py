#!/usr/bin/env python3
"""Build and validate the focused-study long-horizon task corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.data_generation.m4_long_horizon import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED_DIR,
    build_long_horizon_dataset,
    validate_long_horizon_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    args = parser.parse_args()
    manifest = build_long_horizon_dataset(args.output_dir, seed_dir=args.seed_dir)
    validation = validate_long_horizon_dataset(args.output_dir, seed_dir=args.seed_dir)
    if not validation["valid"]:
        raise RuntimeError(f"long-horizon dataset validation failed: {validation['errors']}")
    print(json.dumps({"manifest": manifest, "validation": validation}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
