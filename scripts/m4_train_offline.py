#!/usr/bin/env python3
"""Preflight and run one budget-capped M4 SFT or RSFT LoRA control."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_algorithms import OFFLINE_ALGORITHMS
from miniwebwork.m4_offline import build_m4_offline_training_plan
from miniwebwork.m4_protocol import (
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    assert_m4_canonical_initial_adapter,
    M4_MAX_SEQUENCE_LENGTH,
    write_m4_run_manifest,
)

def _trainer_command(plan: dict, *, output_dir: Path, initial_adapter: Path, base_model: str, max_length: int, learning_rate: float, batch_size: int, grad_accum: int) -> list[str]:
    if max_length <= 0 or batch_size <= 0 or grad_accum <= 0 or learning_rate <= 0:
        raise ValueError("M4 offline trainer settings must be positive")
    if max_length != plan["max_sequence_length"]:
        raise ValueError(
            f"M4 offline max_length is frozen at {plan['max_sequence_length']}, got {max_length}"
        )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "miniwebwork" / "sft" / "train_m2_2.py"),
        "--base-model", base_model,
        "--initial-adapter", str(initial_adapter),
        "--data-dir", plan["train_data"]["data_dir"],
        "--valid-data-dir", plan["validation_data"]["data_dir"],
        "--output-dir", str(output_dir / "training"),
        "--seeds", str(plan["study_seed"]),
        "--max-length", str(max_length),
        "--lr", str(learning_rate),
        "--epochs", str(plan["supervision_passes"]),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--max-supervised-completion-tokens", str(plan["max_supervised_completion_tokens"]),
    ]
    if plan["algorithm_id"] == "sft":
        budget = plan["training_budget"]
        command.extend([
            "--target-supervised-completion-tokens", str(budget["target_supervised_completion_tokens"]),
            "--budget-selection-seed", str(plan["study_seed"]),
            "--max-zero-completion-label-fraction", str(budget["max_zero_completion_label_fraction"]),
        ])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(OFFLINE_ALGORITHMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-data-dir", type=Path, required=True)
    parser.add_argument("--validation-data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--base-model", default="/data/share/model/Qwen3.5-4B")
    parser.add_argument("--max-length", type=int, default=M4_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = build_m4_offline_training_plan(
        args.algorithm,
        args.seed,
        train_data_dir=args.train_data_dir,
        validation_data_dir=args.validation_data_dir,
        task_root=args.task_root,
        seed_dir=args.seed_dir,
    )
    output_dir = args.output_dir.expanduser().resolve()
    canonical_initial_adapter = assert_m4_canonical_initial_adapter(
        args.initial_adapter, task_root=args.task_root
    )
    initial_adapter = Path(canonical_initial_adapter["path"])
    plan["initial_adapter"] = str(initial_adapter)
    plan["initial_adapter_sha256"] = canonical_initial_adapter["sha256"]
    plan["canonical_initial_adapter"] = canonical_initial_adapter
    command = _trainer_command(
        plan,
        output_dir=output_dir,
        initial_adapter=initial_adapter,
        base_model=args.base_model,
        max_length=args.max_length,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
    print(json.dumps({"offline_plan": plan, "trainer_command": command}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    write_m4_run_manifest(output_dir / "resolved_run_manifest.json", plan)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
