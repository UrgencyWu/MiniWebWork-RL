#!/usr/bin/env python3
"""Run a v3 online method as two independently recoverable 24h passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_protocol import DEFAULT_SEED_DIR, DEFAULT_TASK_ROOT, M4RunConfig
from miniwebwork.m4_v3_protocol import (
    V3_STUDY_ID,
    assert_v3_initial_adapter,
    build_v3_run_manifest,
    write_v3_manifest,
)


def _artifact(collection_dir: Path) -> Path:
    paths = sorted((collection_dir / "collector").glob("single_probe_*.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one complete v3 artifact in {collection_dir}, got {paths}")
    return paths[0]


def _report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") and payload.get("passed"):
        return payload
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("rloo", "grpo", "gspo"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--base-model", default="/data/share/model/Qwen3.5-4B")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = M4RunConfig(args.algorithm, args.seed, "train")
    manifest = build_v3_run_manifest(config, task_root=args.task_root, seed_dir=args.seed_dir)
    canonical = assert_v3_initial_adapter(args.initial_adapter)
    initial_adapter = Path(canonical["path"])
    output_dir = args.output_dir.expanduser().resolve()
    plan = []
    adapter = initial_adapter
    for pass_index in (1, 2):
        pass_dir = output_dir / f"pass_{pass_index}"
        if pass_index == 2 and adapter == initial_adapter:
            adapter = output_dir / "pass_1" / "update" / "updated_adapter"
        plan.append({"pass_index": pass_index, "input_adapter": str(adapter), "pass_dir": str(pass_dir)})
        previous = pass_dir / "update" / "online_update_report.json"
        previous_payload = _report(previous)
        if previous_payload is not None:
            adapter = Path(previous_payload["next_adapter"]).resolve()
    manifest["online_plan"] = {
        "wall_time_per_job": "24:00:00",
        "pass_jobs": plan,
        "checkpoint_atomicity": "completed_K4_rollout_group",
        "partial_group_policy": "discard_and_resample",
    }
    if args.dry_run:
        print(json.dumps({"run_manifest": manifest, "plan": plan}, ensure_ascii=False, indent=2))
        return 0

    raise RuntimeError(
        "v3 online execution is intentionally split into independent 24h jobs; "
        "invoke scripts/m4_v3_pass_job.py once per pass"
    )

    adapter = initial_adapter
    completed = []
    for pass_index in (1, 2):
        pass_dir = output_dir / f"pass_{pass_index}"
        pass_dir.mkdir(parents=True, exist_ok=True)
        update_path = pass_dir / "update" / "online_update_report.json"
        existing = _report(update_path)
        if existing is not None:
            adapter = Path(existing["next_adapter"]).resolve()
            completed.append(existing)
            continue
        collect = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "m4_v3_collect_rollouts.py"),
            "--algorithm", args.algorithm,
            "--seed", str(args.seed),
            "--pass-index", str(pass_index),
            "--adapter", str(adapter),
            "--output-dir", str(pass_dir),
            "--task-root", str(args.task_root),
            "--seed-dir", str(args.seed_dir),
            "--base-model", args.base_model,
        ]
        subprocess.run(collect, check=True)
        artifact = _artifact(pass_dir / "collection")
        update = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "m4_apply_online_update.py"),
            "--study-id", V3_STUDY_ID,
            "--algorithm", args.algorithm,
            "--seed", str(args.seed),
            "--pass-index", str(pass_index),
            "--artifact", str(artifact),
            "--adapter", str(adapter),
            "--output-dir", str(pass_dir / "update"),
            "--task-root", str(args.task_root),
            "--seed-dir", str(args.seed_dir),
        ]
        subprocess.run(update, check=True)
        payload = _report(update_path)
        if payload is None:
            raise RuntimeError(f"v3 pass {pass_index} update did not pass")
        adapter = Path(payload["next_adapter"]).resolve()
        completed.append(payload)
    summary = {
        "schema_version": V3_STUDY_ID + "_online_summary_v1",
        "study_id": V3_STUDY_ID,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "initial_adapter": str(initial_adapter),
        "initial_adapter_sha256": canonical["sha256"],
        "passes": [
            {
                "pass_index": report.get("pass_index"),
                "source_artifact": report.get("source_artifact"),
                "source_artifact_sha256": report.get("source_artifact_sha256"),
                "next_adapter": report.get("next_adapter"),
                "next_adapter_sha256": report.get("next_adapter_sha256"),
            }
            for report in completed
        ],
    }
    write_v3_manifest(output_dir / "online_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
