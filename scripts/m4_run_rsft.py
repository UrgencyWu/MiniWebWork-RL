#!/usr/bin/env python3
"""Collect two fixed-policy M4 passes, build RSFT data, and train one seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_protocol import (
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4RunConfig,
    RSFT_TRAIN_TASKS_PER_PASS,
    build_m4_run_manifest,
    write_m4_run_manifest,
)


def _single_artifact(directory: Path) -> Path:
    artifacts = sorted((directory / "collector").glob("single_probe_*.json"))
    if len(artifacts) != 1:
        raise ValueError(f"expected exactly one RSFT collector artifact in {directory}, got {artifacts}")
    return artifacts[0]


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise ValueError(f"adapter directory has no files: {path}")
    for file in files:
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(hashlib.sha256(file.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def _materialize_no_signal_adapter(
    *,
    config: M4RunConfig,
    initial_adapter: Path,
    output_dir: Path,
    task_root: Path,
    seed_dir: Path,
    rsft_manifest: dict,
) -> Path:
    """Preserve RSFT's no-positive outcome without leaking oracle examples."""
    if rsft_manifest.get("selected_task_count") != 0 or not rsft_manifest.get("no_verified_successes"):
        raise ValueError("RSFT no-signal materialization requires an audited empty verified-success corpus")
    target = output_dir / "training" / f"seed_{config.seed}" / "final_adapter"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(initial_adapter, target)
    initial_hash = _directory_sha256(initial_adapter)
    final_hash = _directory_sha256(target)
    if initial_hash != final_hash:
        raise ValueError("RSFT no-signal final adapter does not match the fixed initial adapter")
    metrics = {
        "seed": config.seed,
        "algorithm": "rsft",
        "no_signal": True,
        "reason": "no_verified_successful_rollout_in_fixed_12_task_two_pass_roster",
        "initial_adapter_sha256": initial_hash,
        "final_adapter_sha256": final_hash,
    }
    (target.parent / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run_manifest = build_m4_run_manifest(config, task_root=task_root, seed_dir=seed_dir)
    run_manifest.update(
        {
            "initial_adapter": str(initial_adapter),
            "initial_adapter_sha256": initial_hash,
            "no_signal": True,
            "rsft_corpus_manifest": rsft_manifest,
            "final_adapter": str(target),
            "final_adapter_sha256": final_hash,
        }
    )
    write_m4_run_manifest(output_dir / "resolved_run_manifest.json", run_manifest)
    return target


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
            "--max-tasks", str(RSFT_TRAIN_TASKS_PER_PASS),
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
        "--task-root", str(args.task_root), "--seed-dir", str(args.seed_dir),
    ], check=True)
    rsft_manifest = json.loads((rsft_data / "manifest.json").read_text(encoding="utf-8"))
    if rsft_manifest.get("no_verified_successes"):
        adapter = _materialize_no_signal_adapter(
            config=config,
            initial_adapter=initial_adapter,
            output_dir=output_dir,
            task_root=args.task_root,
            seed_dir=args.seed_dir,
            rsft_manifest=rsft_manifest,
        )
        print(
            json.dumps(
                {
                    "artifacts": [str(path) for path in artifacts],
                    "rsft_corpus": str(rsft_data),
                    "no_signal": True,
                    "final_adapter": str(adapter),
                },
                ensure_ascii=False,
            )
        )
        return 0
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "m4_train_offline.py"),
        "--algorithm", "rsft", "--seed", str(args.seed), "--train-data-dir", str(rsft_data),
        "--validation-data-dir", str(validation_data), "--initial-adapter", str(initial_adapter),
        "--output-dir", str(output_dir), "--task-root", str(args.task_root), "--seed-dir", str(args.seed_dir),
    ], check=True)
    print(json.dumps({"artifacts": [str(path) for path in artifacts], "rsft_corpus": str(rsft_data)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
