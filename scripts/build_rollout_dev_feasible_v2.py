#!/usr/bin/env python3
"""Build rollout_dev_feasible_v2 from its frozen task specification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from miniwebwork.data_generation.feasible_rollout_dev import (
    DEFAULT_PRODUCTS_PATH,
    DEFAULT_SPEC_PATH,
    DEFAULT_SUPPLIERS_PATH,
    build_feasible_rollout_dev,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic non-training feasible rollout-dev gate"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--suppliers", type=Path, default=DEFAULT_SUPPLIERS_PATH)
    args = parser.parse_args()

    manifest = build_feasible_rollout_dev(
        args.output_dir,
        spec_path=args.spec,
        products_path=args.products,
        suppliers_path=args.suppliers,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
