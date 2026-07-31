#!/usr/bin/env python3
"""Resolve a completed M4 run's final adapter and collect its frozen test set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_algorithms import ALL_ALGORITHMS, ONLINE_ALGORITHMS
from miniwebwork.m4_protocol import DEFAULT_SEED_DIR, DEFAULT_TASK_ROOT, M4RunConfig


def _resolve_final_adapter(algorithm: str, seed: int, training_root: Path) -> Path:
    """Find the only auditable final adapter for a completed M4 method/seed."""
    run_dir = training_root / algorithm / f"seed_{seed}"
    if algorithm in ONLINE_ALGORITHMS:
        summary_path = run_dir / "online_run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"completed online summary missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("algorithm") != algorithm or summary.get("seed") != seed:
            raise ValueError("online summary algorithm/seed does not match final evaluation request")
        passes = summary.get("passes")
        if not isinstance(passes, list) or len(passes) != 2:
            raise ValueError("final evaluation requires exactly two completed online passes")
        if [item.get("pass_index") for item in passes] != [1, 2]:
            raise ValueError("online pass order is incomplete or invalid")
        raw_adapter = passes[-1].get("next_adapter")
        if not isinstance(raw_adapter, str):
            raise ValueError("online summary does not record the final adapter")
        adapter = Path(raw_adapter).expanduser().resolve()
    else:
        adapter = (run_dir / "training" / f"seed_{seed}" / "final_adapter").resolve()
        metrics_path = adapter.parent / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"completed offline metrics missing: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("seed") != seed:
            raise ValueError("offline metrics seed does not match final evaluation request")
    if not adapter.is_dir():
        raise FileNotFoundError(f"completed final adapter missing: {adapter}")
    return adapter


def _command(
    config: M4RunConfig, *, adapter: Path, output_dir: Path, task_root: Path, seed_dir: Path
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "m4_collect_rollouts.py"),
        "--algorithm", config.algorithm_id,
        "--seed", str(config.seed),
        "--phase", "final_test",
        "--adapter", str(adapter),
        "--output-dir", str(output_dir),
        "--task-root", str(task_root),
        "--seed-dir", str(seed_dir),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(ALL_ALGORITHMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-root", type=Path, default=Path("outputs/m4_runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = M4RunConfig(args.algorithm, args.seed, "final_test")
    config.validate(task_root=args.task_root)
    training_root = args.training_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    task_root = args.task_root.expanduser().resolve()
    seed_dir = args.seed_dir.expanduser().resolve()
    adapter = _resolve_final_adapter(args.algorithm, args.seed, training_root)
    command = _command(
        config, adapter=adapter, output_dir=output_dir, task_root=task_root, seed_dir=seed_dir
    )
    print(json.dumps({"adapter": str(adapter), "collector_command": command}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
