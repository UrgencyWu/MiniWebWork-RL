#!/usr/bin/env python3
"""Collect two fixed-policy M4 passes, build RSFT data, and train one seed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_protocol import DEFAULT_SEED_DIR, DEFAULT_TASK_ROOT, M4RunConfig


def _single_artifact(directory: Path) -> Path:
    artifacts = sorted((directory / "collector").glob("single_probe_*.json"))
    if len(artifacts) != 1:
        raise ValueError(f"expected exactly one RSFT collector artifact in {directory}, got {artifacts}")
    return artifacts[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--sft-validation-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = M4RunConfig("rsft", args.seed, "train")
    config.validate(task_root=args.task_root)
    initial_adapter = args.initial_adapter.expanduser().resolve()
    validation_data = args.sft_validation_data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not initial_adapter.is_dir() or not validation_data.is_dir():
        raise FileNotFoundError("RSFT requires existing initial adapter and full SFT validation corpus")
    commands = []
    for pass_index in range(1, config.online_passes + 1):
        commands.append([
            sys.executable, str(PROJECT_ROOT / "scripts" / "m4_collect_rollouts.py"),
            "--algorithm", "rsft", "--seed", str(args.seed), "--phase", "train",
            "--train-pass-index", str(pass_index), "--adapter", str(initial_adapter),
            "--output-dir", str(output_dir / f"pass_{pass_index}" / "collection"),
            "--task-root", str(args.task_root), "--seed-dir", str(args.seed_dir),
        ])
    if args.dry_run:
        print(json.dumps({"commands": commands}, ensure_ascii=False, indent=2))
        return 0
    artifacts = []
    for pass_index, command in enumerate(commands, start=1):
        subprocess.run(command, check=True)
        artifacts.append(_single_artifact(output_dir / f"pass_{pass_index}" / "collection"))
    rsft_data = output_dir / "rsft_corpus"
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "build_m4_rsft_dataset.py"),
        "--artifacts", *(str(path) for path in artifacts), "--output-dir", str(rsft_data), "--seed", str(args.seed),
    ], check=True)
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "m4_train_offline.py"),
        "--algorithm", "rsft", "--seed", str(args.seed), "--train-data-dir", str(rsft_data),
        "--validation-data-dir", str(validation_data), "--initial-adapter", str(initial_adapter),
        "--output-dir", str(output_dir / "training"), "--task-root", str(args.task_root), "--seed-dir", str(args.seed_dir),
    ], check=True)
    print(json.dumps({"artifacts": [str(path) for path in artifacts], "rsft_corpus": str(rsft_data)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
