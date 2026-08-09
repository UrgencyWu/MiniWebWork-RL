#!/usr/bin/env python3
"""Run or resume one frozen formal 120-task K=4 evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.formal_eval import FORMAL_EVAL_MANIFEST_NAME, expected_eval_root, prepare_formal_eval, run_formal_eval
from miniwebwork.long_horizon_rl.formal_invocation import finish_formal_invocation, start_formal_invocation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("verified_sft", "multi_turn_grpo", "step_aware_gpo"))
    parser.add_argument("--seed", required=True, type=int, choices=(20260801, 20260802, 20260803))
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--base-model-manifest", type=Path, default=PROJECT_ROOT / "data" / "m4_long_horizon_base_model_manifest_v1.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or expected_eval_root(args.method, args.seed)
    prepared = prepare_formal_eval(
        method=args.method,
        seed=args.seed,
        output_dir=output,
        expected_git_sha=args.expected_git_sha,
        readiness_path=args.readiness,
        base_model_manifest=args.base_model_manifest,
    )
    invocation = start_formal_invocation(
        root=prepared["root"], phase="frozen_test", method=args.method, seed=args.seed,
        git_sha=prepared["config"]["git_sha"], expected_cpus=8, expected_gpus=1,
        log_stem="m4_lh_formal_eval",
        telemetry_path=f"logs/m4_lh_formal_eval_{os.environ.get('SLURM_JOB_ID', '')}_gpu.csv",
    )
    try:
        result = asyncio.run(run_formal_eval(prepared))
    except BaseException:
        finish_formal_invocation(invocation, status="ERROR")
        raise
    finish_formal_invocation(invocation, status="WORKLOAD_COMPLETE")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
