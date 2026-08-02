#!/usr/bin/env python3
"""Run a v3 SFT/RSFT trainer with the same audited 250k label target."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_protocol import DEFAULT_SEED_DIR, DEFAULT_TASK_ROOT, M4RunConfig
from miniwebwork.m4_v3_protocol import (
    V3_MAX_SEQUENCE_LENGTH,
    V3_STUDY_ID,
    V3_TARGET_SUPERVISED_COMPLETION_TOKENS,
    assert_v3_initial_adapter,
    build_v3_run_manifest,
    write_v3_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sft", "rsft"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-data-dir", type=Path, required=True)
    parser.add_argument("--validation-data-dir", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--base-model", default="/data/share/model/Qwen3.5-4B")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_data = args.train_data_dir.expanduser().resolve()
    validation_data = args.validation_data_dir.expanduser().resolve()
    if not train_data.is_dir() or not validation_data.is_dir():
        raise FileNotFoundError("v3 offline training requires train and validation directories")
    canonical = assert_v3_initial_adapter(args.initial_adapter)
    config = M4RunConfig(args.algorithm, args.seed, "train")
    manifest = build_v3_run_manifest(config, task_root=args.task_root, seed_dir=args.seed_dir)
    output_dir = args.output_dir.expanduser().resolve()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "miniwebwork" / "sft" / "train_m2_2.py"),
        "--base-model", args.base_model,
        "--initial-adapter", canonical["path"],
        "--data-dir", str(train_data),
        "--valid-data-dir", str(validation_data),
        "--output-dir", str(output_dir / "training"),
        "--seeds", str(args.seed),
        "--max-length", str(V3_MAX_SEQUENCE_LENGTH),
        "--lr", str(args.learning_rate),
        "--epochs", "1",
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--max-supervised-completion-tokens", str(V3_TARGET_SUPERVISED_COMPLETION_TOKENS),
        "--target-supervised-completion-tokens", str(V3_TARGET_SUPERVISED_COMPLETION_TOKENS),
        "--budget-selection-seed", str(args.seed),
        "--max-zero-completion-label-fraction", "0.0",
    ]
    manifest["offline_training"] = {
        "algorithm": args.algorithm,
        "train_data_dir": str(train_data),
        "validation_data_dir": str(validation_data),
        "initial_adapter": canonical,
        "target_supervised_completion_tokens": V3_TARGET_SUPERVISED_COMPLETION_TOKENS,
        "max_sequence_length": V3_MAX_SEQUENCE_LENGTH,
        "max_zero_completion_label_fraction": 0.0,
        "command": command,
    }
    print(json.dumps({"run_manifest": manifest, "trainer_command": command}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    write_v3_manifest(output_dir / "resolved_run_manifest.json", manifest)
    subprocess.run(command, check=True)
    audit = output_dir / "training" / f"seed_{args.seed}" / "supervision_audit.json"
    if not audit.is_file():
        raise RuntimeError("v3 offline training did not produce supervision_audit.json")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    statistics = payload.get("statistics", {})
    if statistics.get("zero_completion_label_sample_fraction") != 0.0:
        raise ValueError("v3 offline training produced non-zero zero-label fraction")
    realized = statistics.get("completion_tokens_per_epoch")
    if not isinstance(realized, int) or realized <= 0 or V3_TARGET_SUPERVISED_COMPLETION_TOKENS - realized >= 13:
        raise ValueError("v3 offline training did not meet the auditable 250k target tolerance")
    gate = {
        "schema_version": V3_STUDY_ID + "_offline_gate_v1",
        "study_id": V3_STUDY_ID,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "target_supervised_completion_tokens": V3_TARGET_SUPERVISED_COMPLETION_TOKENS,
        "realized_supervised_completion_tokens": realized,
        "zero_completion_label_sample_fraction": 0.0,
        "supervision_audit": str(audit),
    }
    write_v3_manifest(output_dir / "v3_training_gate.json", gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
