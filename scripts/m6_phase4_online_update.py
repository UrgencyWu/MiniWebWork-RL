#!/usr/bin/env python3
"""Apply one minimal full-horizon K4x4 trajectory-GRPO update."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import directory_sha256  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402
from miniwebwork.webshop_rl.phase4_online import train_phase4_iteration  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--input-adapter", type=Path, required=True)
    parser.add_argument("--reference-sft-adapter", type=Path, required=True)
    parser.add_argument("--input-optimizer", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--microbatch-size", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument("--policy-credit-window", choices=("full", "tail2", "preterminal1"), default="full")
    args = parser.parse_args()

    root = args.collection_root.expanduser().resolve()
    report = json.loads((root / "collection_report.json").read_text(encoding="utf-8"))
    _require(report.get("complete") is True, "Phase4 collection is incomplete")
    _require(report.get("mode") == "phase4_online_rl_collection", "Phase4 collection mode drift")
    _require(report.get("role") == "train" and report.get("K") == 4, "Phase4 collection split/K drift")
    _require(report.get("training_updates_allowed") is True, "Phase4 collection is not update-authorized")
    _require(report.get("task_count") == 4, "Phase4 collection must contain four tasks")
    groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=4)
        for path in sorted((root / "groups").glob("g*.json"))
    ]
    _require(len(groups) == 4, "Phase4 collection group count drift")
    input_adapter = args.input_adapter.expanduser().resolve()
    _require(
        report.get("policy_lineage", {}).get("adapter_sha256") == directory_sha256(input_adapter),
        "Phase4 behavior policy does not match learner input",
    )
    input_semantic = str(report.get("policy_lineage", {}).get("adapter_semantic_sha256", ""))
    _require(len(input_semantic) == 64, "Phase4 behavior semantic identity is missing")
    _require(
        report.get("group_content_sha256") == [group["content_sha256"] for group in groups],
        "Phase4 collection/group roster drift",
    )
    result = train_phase4_iteration(
        groups=groups,
        base_model=args.base_model.expanduser().resolve(),
        input_adapter=input_adapter,
        input_adapter_semantic_sha256=input_semantic,
        reference_sft_adapter=args.reference_sft_adapter.expanduser().resolve(),
        input_optimizer=args.input_optimizer.expanduser().resolve() if args.input_optimizer else None,
        output_root=args.output_dir,
        iteration_index=args.iteration_index,
        seed=args.seed,
        microbatch_size=args.microbatch_size,
        policy_credit_window=args.policy_credit_window,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
