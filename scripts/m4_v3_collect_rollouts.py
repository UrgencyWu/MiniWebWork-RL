#!/usr/bin/env python3
"""Collect one v3 pass with 24-hour-safe, complete-group recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_protocol import (
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4RunConfig,
    ONLINE_PASS_ACTION_TOKEN_CAP,
    SPLIT_WORLD_COUNTS,
)
from miniwebwork.m4_v3_protocol import (
    V3_STUDY_ID,
    build_v3_run_manifest,
    write_v3_manifest,
)


def _collection_seed(study_seed: int, pass_index: int) -> int:
    digest = hashlib.sha256(
        f"m4_rlvr_study_v3:{study_seed}:pass:{pass_index}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _artifact(directory: Path) -> Path:
    artifacts = sorted((directory / "collector").glob("single_probe_*.json"))
    if len(artifacts) != 1:
        raise ValueError(f"expected one complete v3 collector artifact, got {artifacts}")
    return artifacts[0]


def _reuse_completed_artifact(
    directory: Path,
    *,
    manifest: dict,
    split: str,
    expected_task_count: int,
    pass_index: int | None,
) -> Path | None:
    """Make a retried 24h allocation idempotent after a completed collection."""
    paths = sorted((directory / "collector").glob("single_probe_*.json"))
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"multiple v3 complete artifacts exist; refusing to guess: {paths}")
    artifact = paths[0]
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError("collector directory contains a non-complete final artifact")
    expected = {
        "study_id": V3_STUDY_ID,
        "git_sha": manifest["git_sha"],
        "split": split,
        "study_seed": manifest["config"]["seed"],
        "collection_pass_index": pass_index,
        "K": 4,
        "max_tasks": None,
        "available_task_count": expected_task_count,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"existing v3 artifact identity mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sft", "rsft", "rloo", "grpo", "gspo"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("train", "final_test"), default="train")
    parser.add_argument("--pass-index", type=int, choices=(1, 2), default=None)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--base-model", default="/data/share/model/Qwen3.5-4B")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.adapter.expanduser().resolve().is_dir():
        raise FileNotFoundError(f"adapter directory not found: {args.adapter}")
    if args.phase == "train" and args.pass_index not in (1, 2):
        raise ValueError("train collection requires pass-index 1 or 2")
    if args.phase == "train" and args.algorithm == "sft":
        raise ValueError("SFT uses the offline corpus and cannot collect train rollouts")
    if args.phase == "final_test" and args.pass_index is not None:
        raise ValueError("final_test collection does not accept pass-index")
    config = M4RunConfig(args.algorithm, args.seed, args.phase)
    manifest = build_v3_run_manifest(config, task_root=args.task_root, seed_dir=args.seed_dir)
    output_dir = args.output_dir.expanduser().resolve()
    collection_dir = output_dir / "collection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    incremental = collection_dir / "collector" / "incremental_custom_t1_p1_k0.json"
    collection_seed = _collection_seed(args.seed, args.pass_index or 0)
    split = "train" if args.phase == "train" else "test"
    expected_task_count = 240 if args.phase == "train" else SPLIT_WORLD_COUNTS["test"] * 4
    existing = _reuse_completed_artifact(
        collection_dir,
        manifest=manifest,
        split=split,
        expected_task_count=expected_task_count,
        pass_index=args.pass_index,
    )
    if existing is not None:
        print(json.dumps({"resumed_from_completed_artifact": str(existing)}, ensure_ascii=False))
        return 0
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "m2_3_mini_single_probe.py"),
        "--policy", "custom",
        "--policy-label", f"m4_v3_{args.algorithm}_seed{args.seed}",
        "--adapter", str(args.adapter.expanduser().resolve()),
        "--base-model", args.base_model,
        "--task-dir", str(args.task_root.expanduser().resolve() / split),
        "--seed-dir", str(args.seed_dir.expanduser().resolve()),
        "--split", split,
        "--task-order-seed", str(args.seed),
        "--K", "4",
        "--seed", str(collection_seed),
        "--study-seed", str(args.seed),
        "--temperature", "1.0",
        "--top-p", "1.0",
        "--top-k", "0",
        "--max-model-turns", "20",
        "--max-env-steps", "20",
        "--max-new-tokens", "128",
        "--study-id", V3_STUDY_ID,
        "--output-dir", str(collection_dir),
    ]
    if args.phase == "train":
        command.extend([
            "--collection-pass-index", str(args.pass_index),
            "--max-collected-action-tokens", str(ONLINE_PASS_ACTION_TOKEN_CAP),
        ])
    if incremental.is_file():
        command.extend(["--resume-from", str(incremental)])
    manifest["collection"] = {
        "study_seed": args.seed,
        "collection_seed": collection_seed,
        "train_pass_index": args.pass_index,
        "task_order_seed": args.seed,
        "full_roster_count": expected_task_count,
        "max_tasks": None,
        "max_collected_action_tokens": ONLINE_PASS_ACTION_TOKEN_CAP if args.phase == "train" else None,
        "resume_capable": True,
        "checkpoint_atomicity": "completed_K4_rollout_group",
    }
    print(json.dumps({"run_manifest": manifest, "collector_command": command}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    write_v3_manifest(output_dir / "resolved_run_manifest.json", manifest)
    subprocess.run(command, check=True)
    result = _artifact(collection_dir)
    artifact = json.loads(result.read_text(encoding="utf-8"))
    if artifact.get("study_id") != V3_STUDY_ID:
        raise ValueError("collector returned an artifact with the wrong v3 study id")
    if artifact.get("max_tasks") is not None or artifact.get("available_task_count") != expected_task_count:
        raise ValueError("v3 collector did not use the complete frozen roster")
    print(json.dumps({"artifact": str(result), "artifact_sha256": hashlib.sha256(result.read_bytes()).hexdigest()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
