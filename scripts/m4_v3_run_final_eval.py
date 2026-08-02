#!/usr/bin/env python3
"""Resolve a verified v3 adapter and collect its 120-task frozen test set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_protocol import DEFAULT_SEED_DIR, DEFAULT_TASK_ROOT
from miniwebwork.m4_v3_protocol import V3_STUDY_ID, assert_v3_initial_adapter


def _final_adapter(algorithm: str, seed: int, training_root: Path) -> Path:
    run_dir = training_root / algorithm / f"seed_{seed}"
    if algorithm in {"rloo", "grpo", "gspo"}:
        summary_path = run_dir / "online_run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"v3 online summary missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("study_id") != V3_STUDY_ID or summary.get("complete") is not True:
            raise ValueError("v3 online summary is not complete")
        passes = summary.get("passes")
        if not isinstance(passes, list) or [item.get("pass_index") for item in passes] != [1, 2]:
            raise ValueError("v3 online summary lacks ordered pass lineage")
        adapter = Path(passes[-1].get("next_adapter", "")).expanduser().resolve()
    else:
        adapter = (run_dir / "training" / f"seed_{seed}" / "final_adapter").resolve()
        gate = run_dir / "v3_training_gate.json"
        if not gate.is_file():
            raise FileNotFoundError(f"v3 offline gate missing: {gate}")
        if json.loads(gate.read_text(encoding="utf-8")).get("study_id") != V3_STUDY_ID:
            raise ValueError("v3 offline gate has the wrong study id")
    if not adapter.is_dir():
        raise FileNotFoundError(f"v3 final adapter missing: {adapter}")
    return adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sft", "rsft", "rloo", "grpo", "gspo"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-root", type=Path, default=Path("outputs/m4_v3_runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--base-model", default="/data/share/model/Qwen3.5-4B")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    adapter = _final_adapter(args.algorithm, args.seed, args.training_root.expanduser().resolve())
    if args.algorithm == "sft":
        # SFT and RSFT both use the same strict test collector; their training
        # distinction lives in the adapter lineage and offline gate.
        pass
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "m4_v3_collect_rollouts.py"),
        "--algorithm", args.algorithm,
        "--seed", str(args.seed),
        "--phase", "final_test",
        "--adapter", str(adapter),
        "--output-dir", str(args.output_dir.expanduser().resolve()),
        "--task-root", str(args.task_root),
        "--seed-dir", str(args.seed_dir),
        "--base-model", args.base_model,
    ]
    print(json.dumps({"study_id": V3_STUDY_ID, "adapter": str(adapter), "command": command}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
