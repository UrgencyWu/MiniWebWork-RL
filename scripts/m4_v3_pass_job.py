#!/usr/bin/env python3
"""Run one v3 online/RSFT pass; safe to retry within a 24h Slurm allocation."""

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


def _artifact(directory: Path) -> Path:
    paths = sorted((directory / "collection" / "collector").glob("single_probe_*.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one complete collection artifact, got {paths}")
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("rloo", "grpo", "gspo", "rsft"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pass-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--input-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--base-model", default="/data/share/model/Qwen3.5-4B")
    args = parser.parse_args()
    input_adapter = args.input_adapter.expanduser().resolve()
    if args.pass_index == 1:
        assert_v3_initial_adapter(input_adapter)
    elif not input_adapter.is_dir():
        raise FileNotFoundError(f"v3 pass input adapter not found: {input_adapter}")
    output_dir = args.output_dir.expanduser().resolve()
    update_report = output_dir / "update" / "online_update_report.json"
    if args.algorithm != "rsft" and update_report.is_file():
        payload = json.loads(update_report.read_text(encoding="utf-8"))
        if payload.get("complete") and payload.get("passed"):
            print(json.dumps({"resumed_from_completed_update": True, "report": str(update_report)}))
            return 0
    collect = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "m4_v3_collect_rollouts.py"),
        "--algorithm", args.algorithm,
        "--seed", str(args.seed),
        "--pass-index", str(args.pass_index),
        "--adapter", str(input_adapter),
        "--output-dir", str(output_dir),
        "--task-root", str(args.task_root),
        "--seed-dir", str(args.seed_dir),
        "--base-model", args.base_model,
    ]
    subprocess.run(collect, check=True)
    artifact = _artifact(output_dir)
    if args.algorithm == "rsft":
        print(json.dumps({"pass_index": args.pass_index, "artifact": str(artifact)}, ensure_ascii=False))
        return 0
    update = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "m4_apply_online_update.py"),
        "--study-id", V3_STUDY_ID,
        "--algorithm", args.algorithm,
        "--seed", str(args.seed),
        "--pass-index", str(args.pass_index),
        "--artifact", str(artifact),
        "--adapter", str(input_adapter),
        "--output-dir", str(output_dir / "update"),
        "--task-root", str(args.task_root),
        "--seed-dir", str(args.seed_dir),
    ]
    subprocess.run(update, check=True)
    print(json.dumps({"pass_index": args.pass_index, "artifact": str(artifact), "update": str(update_report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
