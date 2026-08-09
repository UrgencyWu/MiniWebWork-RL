#!/usr/bin/env python3
"""Run or safely resume one formal 250k-token online RL method/seed."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.formal_online import (
    FORMAL_ONLINE_MANIFEST_NAME,
    expected_online_root,
    prepare_online_formal,
    run_online_formal,
)
from miniwebwork.long_horizon_rl.formal_invocation import finish_formal_invocation, start_formal_invocation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("multi_turn_grpo", "step_aware_gpo"))
    parser.add_argument("--seed", required=True, type=int, choices=(20260801, 20260802, 20260803))
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--shared-sft-root", type=Path, default=PROJECT_ROOT / "outputs" / "m4_long_horizon_credit_v1" / "formal" / "shared_sft" / "seed_20260801")
    parser.add_argument("--base-model-manifest", type=Path, default=PROJECT_ROOT / "data" / "m4_long_horizon_base_model_manifest_v1.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or expected_online_root(args.method, args.seed)
    prepared = prepare_online_formal(
        output_dir=output,
        method=args.method,
        seed=args.seed,
        expected_git_sha=args.expected_git_sha,
        readiness_path=args.readiness,
        shared_sft_root=args.shared_sft_root,
        base_model_manifest=args.base_model_manifest,
    )
    invocation = start_formal_invocation(
        root=prepared["root"], phase="online", method=args.method, seed=args.seed,
        git_sha=prepared["run_config"]["git_sha"], expected_cpus=8, expected_gpus=1,
        log_stem="m4_lh_formal_online",
        telemetry_path=f"logs/m4_lh_formal_online_{args.method}_{args.seed}_{os.environ.get('SLURM_JOB_ID', '')}_gpu.csv",
    )
    try:
        result = asyncio.run(run_online_formal(prepared))
    except BaseException:
        finish_formal_invocation(invocation, status="ERROR")
        raise
    finish_formal_invocation(invocation, status="WORKLOAD_COMPLETE")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
