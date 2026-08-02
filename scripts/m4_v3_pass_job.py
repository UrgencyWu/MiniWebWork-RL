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
from miniwebwork.m4_v3_protocol import (
    V3_STUDY_ID,
    assert_v3_initial_adapter,
    load_v3_study_manifest,
    write_v3_manifest,
)


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
    canonical = None
    if args.pass_index == 1:
        canonical = assert_v3_initial_adapter(input_adapter)
    elif not input_adapter.is_dir():
        raise FileNotFoundError(f"v3 pass input adapter not found: {input_adapter}")
    if canonical is None:
        study_payload = load_v3_study_manifest()
        declared = study_payload["payload"]["canonical_initial_adapter"]
        canonical = {
            "path": str((PROJECT_ROOT / declared["relative_path"]).resolve()),
            "relative_path": declared["relative_path"],
            "sha256": declared["directory_sha256"],
            "study_manifest_sha256": study_payload["sha256"],
        }
    output_dir = args.output_dir.expanduser().resolve()
    update_report = output_dir / "update" / "online_update_report.json"
    summary_path = output_dir.parent / "online_run_summary.json"
    if args.pass_index == 2 and args.algorithm != "rsft" and summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("study_id") == V3_STUDY_ID and payload.get("complete") is True:
            print(json.dumps({"resumed_from_completed_summary": True, "summary": str(summary_path)}))
            return 0
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
    if args.pass_index == 2:
        pass1_report_path = output_dir.parent / "pass_1" / "update" / "online_update_report.json"
        pass1 = json.loads(pass1_report_path.read_text(encoding="utf-8"))
        pass2 = json.loads(update_report.read_text(encoding="utf-8"))
        summary = {
            "schema_version": V3_STUDY_ID + "_online_summary_v1",
            "study_id": V3_STUDY_ID,
            "complete": True,
            "algorithm": args.algorithm,
            "seed": args.seed,
            "initial_adapter": pass1.get("source_adapter"),
            "initial_adapter_sha256": pass1.get("source_adapter_sha256"),
            "canonical_initial_adapter": canonical,
            "passes": [
                {
                    "pass_index": 1,
                    "artifact": pass1.get("source_artifact"),
                    "artifact_sha256": pass1.get("source_artifact_sha256"),
                    "next_adapter": pass1.get("next_adapter"),
                    "next_adapter_sha256": pass1.get("next_adapter_sha256"),
                },
                {
                    "pass_index": 2,
                    "artifact": pass2.get("source_artifact"),
                    "artifact_sha256": pass2.get("source_artifact_sha256"),
                    "next_adapter": pass2.get("next_adapter"),
                    "next_adapter_sha256": pass2.get("next_adapter_sha256"),
                },
            ],
        }
        write_v3_manifest(summary_path, summary)
    print(json.dumps({"pass_index": args.pass_index, "artifact": str(artifact), "update": str(update_report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
