#!/usr/bin/env python3
"""Run one resumable disposable online iteration; formal mode does not exist."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from miniwebwork.long_horizon_rl.model_manifest import BASE_MODEL_MANIFEST_PATH
from miniwebwork.long_horizon_rl.online_runner import (
    prepare_online_preflight,
    run_online_preflight_once,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("multi_turn_grpo", "step_aware_gpo"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument(
        "--base-model-manifest",
        type=Path,
        default=BASE_MODEL_MANIFEST_PATH,
    )
    parser.add_argument("--browser-workers", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--maximum-tasks", type=int, required=True)
    parser.add_argument("--learner-microbatch-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--collection-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared = prepare_online_preflight(
        output_dir=args.output_dir,
        method=args.method,
        seed=args.seed,
        initial_adapter=args.initial_adapter,
        base_model_manifest=args.base_model_manifest,
        browser_workers=args.browser_workers,
        maximum_tasks=args.maximum_tasks,
        learner_microbatch_size=args.learner_microbatch_size,
        collection_only=args.collection_only,
    )
    report = asyncio.run(run_online_preflight_once(prepared))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
