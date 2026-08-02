#!/usr/bin/env python3
"""Build the v3 RSFT corpus after both independent collection passes finish."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_v3_protocol import DEFAULT_SEED_DIR, DEFAULT_TASK_ROOT
from miniwebwork.sft.m4_v3_rsft_dataset import build_m4_v3_rsft_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    args = parser.parse_args()
    # Loading the v3 manifest here makes a corpus build fail closed before
    # reading any rollout data when the study protocol is absent or drifted.
    from miniwebwork.m4_v3_protocol import load_v3_study_manifest
    load_v3_study_manifest()
    result = build_m4_v3_rsft_dataset(list(args.artifacts), args.output_dir, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
