#!/usr/bin/env python3
"""Measure M5 HF replay parity on an already committed formal K4 iteration.

This diagnostic is read-only with respect to rollout and learner artifacts.  It
exists so a fail-closed parity rejection records the full distribution rather
than encouraging an unaudited threshold change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_file, sha256_json
from miniwebwork.long_horizon_rl.learner import load_trainable_policy_model
from miniwebwork.m5_webshop_protocol import load_protocol
from miniwebwork.webshop_rl.online_training import (
    ReplayParityError,
    audit_initial_replay_parity,
    prepare_group_training_examples,
    validate_committed_group,
)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _self_hash(payload: dict) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("multi_turn_grpo", "anchor_gigpo"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/m5_webshop_credit_assignment_v1/preflight/sft_gpu/training/final_adapter_epoch_1",
    )
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    invocation = json.loads((run_root / "invocation.json").read_text(encoding="utf-8"))
    invocation_hash = invocation.get("content_sha256")
    if invocation_hash != _self_hash(invocation):
        raise ValueError("diagnostic invocation self-hash drift")
    if invocation.get("method") != args.method or invocation.get("seed") != args.seed:
        raise ValueError("diagnostic method/seed does not match invocation")
    groups_root = run_root / "iterations/i0000/groups"
    group_paths = sorted(groups_root.glob("g*.json"))
    if len(group_paths) != 32:
        raise ValueError(f"expected 32 committed diagnostic groups, found {len(group_paths)}")
    groups = [validate_committed_group(json.loads(path.read_text(encoding="utf-8"))) for path in group_paths]
    producer_git_shas = sorted({str(group["git_sha"]) for group in groups})
    producer_protocol_shas = sorted({str(group["protocol_sha256"]) for group in groups})
    if len(producer_git_shas) != 1 or len(producer_protocol_shas) != 1:
        raise ValueError("diagnostic group lineage is not unique")

    prepared_all = [prepare_group_training_examples(group, args.method) for group in groups]
    prepared = [item for item in prepared_all if not item["zero_advantage_group"]]
    parity_source = prepared if prepared else prepared_all
    base_model = args.base_model.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    model, tokenizer = load_trainable_policy_model(base_model=base_model, adapter_path=adapter)
    protocol = load_protocol()
    try:
        try:
            parity = audit_initial_replay_parity(
                model=model,
                prepared=parity_source,
                tokenizer=tokenizer,
                device=torch.device("cuda:0"),
                microbatch_size=args.microbatch_size,
                thresholds=protocol["payload"]["online"]["parity_contract"],
            )
        except ReplayParityError as error:
            parity = error.report
    finally:
        del model
        torch.cuda.empty_cache()

    report = {
        "schema_version": "m5_webshop_replay_parity_diagnostic_v1",
        "diagnostic_git_sha": _git_sha(),
        "producer_git_sha": producer_git_shas[0],
        "producer_protocol_sha256": producer_protocol_shas[0],
        "diagnostic_protocol_sha256": protocol["sha256"],
        "invocation_sha256": sha256_file(run_root / "invocation.json"),
        "method": args.method,
        "seed": args.seed,
        "group_count": len(groups),
        "nonzero_group_count": len(prepared),
        "parity_example_group_count": len(parity_source),
        "group_files_sha256": sha256_json({path.name: sha256_file(path) for path in group_paths}),
        "base_model": str(base_model),
        "adapter": str(adapter),
        "adapter_sha256": directory_sha256(adapter),
        "microbatch_size": args.microbatch_size,
        "parity": parity,
    }
    report["content_sha256"] = _self_hash(report)
    output = (args.output or run_root / "replay_parity_diagnostic.json").expanduser().resolve()
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
