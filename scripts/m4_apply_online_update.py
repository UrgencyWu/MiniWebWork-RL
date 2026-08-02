#!/usr/bin/env python3
"""Audit and apply one M4 RLOO/GRPO/GSPO update from a strict pass artifact."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_algorithms import ONLINE_ALGORITHMS
from miniwebwork.m4_protocol import (
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4RunConfig,
    ONLINE_PASS_ACTION_TOKEN_CAP,
    build_m4_run_manifest,
    write_m4_run_manifest,
)
from miniwebwork.m4_training import build_online_update_plan
from miniwebwork.rl.m4_update import apply_m4_online_update


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _m3_helpers():
    """Reuse the previously audited model-load and replay primitives verbatim."""
    path = PROJECT_ROOT / "scripts" / "m3_0_single_batch_smoke.py"
    spec = importlib.util.spec_from_file_location("m3_0_single_batch_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import M3 replay helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_artifact(
    artifact: dict[str, Any],
    *,
    config: M4RunConfig,
    pass_index: int,
    adapter_hash: str,
    run_manifest: dict[str, Any],
) -> None:
    if artifact.get("schema_version") != "3.3" or not artifact.get("complete"):
        raise ValueError("M4 update requires a complete schema-3.3 rollout artifact")
    expected_study_id = run_manifest.get("study_id", "m4_rlvr_v1")
    if artifact.get("study_id") != expected_study_id or artifact.get("split") != "train":
        raise PermissionError("M4 update accepts only strict train artifacts for the active protocol")
    if artifact.get("git_sha") != run_manifest.get("git_sha"):
        raise ValueError("rollout artifact git SHA does not match the frozen M4 train manifest")
    if artifact.get("prompt_contract") != run_manifest.get("prompt_contract"):
        raise ValueError("rollout artifact prompt contract does not match the frozen M4 train manifest")
    if artifact.get("study_seed") != config.seed or artifact.get("collection_pass_index") != pass_index:
        raise ValueError("artifact study seed or pass index does not match this update")
    if artifact.get("K") != config.group_size:
        raise ValueError("artifact rollout group size does not match M4 config")
    if (artifact.get("temperature"), artifact.get("top_p"), artifact.get("top_k")) != (1.0, 1.0, 0):
        raise ValueError("M4 update requires raw temperature=1, top_p=1, top_k=0 collection")
    if artifact.get("generation_runtime") != {"use_cache": False, "strict_on_policy": True}:
        raise ValueError("M4 update requires strict no-cache on-policy collection")
    if artifact.get("adapter_sha256") != adapter_hash:
        raise ValueError("rollout artifact adapter hash does not match --adapter")
    if artifact.get("task_source_sha256") != run_manifest["hashes"]["task_source_sha256"]:
        raise ValueError("rollout artifact task source hash does not match M4 train manifest")
    if artifact.get("max_collected_action_tokens") != ONLINE_PASS_ACTION_TOKEN_CAP:
        raise ValueError("artifact does not use the fixed per-pass M4 token cap")
    collected = artifact.get("collected_action_tokens")
    if not isinstance(collected, int) or collected < 0 or collected > ONLINE_PASS_ACTION_TOKEN_CAP:
        raise ValueError("artifact collected action tokens are missing or exceed the M4 pass cap")
    if artifact.get("completed_task_count") != len(artifact.get("groups", [])):
        raise ValueError("artifact completed task count disagrees with its group evidence")


def _audit_selected_groups(model, groups, helpers) -> dict[str, int | float]:
    maximum = 0.0
    token_count = 0
    turn_count = 0
    for group in groups:
        result = helpers._audit_old_current(model, group)
        maximum = max(maximum, float(result["max_abs_difference"]))
        token_count += int(result["token_count"])
        turn_count += int(result["turn_count"])
    return {
        "max_abs_difference": maximum,
        "token_count": token_count,
        "turn_count": turn_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(ONLINE_ALGORITHMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pass-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--study-id", choices=("m4_rlvr_v1", "m4_rlvr_v3"), default="m4_rlvr_v1")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--old-current-tolerance", type=float, default=5e-2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.learning_rate <= 0 or args.gradient_clip <= 0 or args.old_current_tolerance <= 0:
        raise ValueError("learning-rate, gradient-clip and old-current-tolerance must be positive")
    if not 0 < args.clip_epsilon < 1:
        raise ValueError("clip-epsilon must be in (0, 1)")
    artifact_path = args.artifact.expanduser().resolve()
    adapter_dir = args.adapter.expanduser().resolve()
    if not artifact_path.is_file() or not adapter_dir.is_dir():
        raise FileNotFoundError("M4 update requires an existing artifact and adapter directory")
    if not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("M4 online update requires CUDA")

    config = M4RunConfig(args.algorithm, args.seed, "train")
    if args.study_id == "m4_rlvr_v3":
        from miniwebwork.m4_v3_protocol import build_v3_run_manifest, V3_STUDY_ID
        run_manifest = build_v3_run_manifest(config, task_root=args.task_root, seed_dir=args.seed_dir)
        if args.study_id != V3_STUDY_ID:
            raise ValueError("v3 study identity mismatch")
    else:
        run_manifest = build_m4_run_manifest(config, task_root=args.task_root, seed_dir=args.seed_dir)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    helpers = _m3_helpers()
    adapter_hash = helpers._directory_sha256(adapter_dir)
    _validate_artifact(
        artifact,
        config=config,
        pass_index=args.pass_index,
        adapter_hash=adapter_hash,
        run_manifest=run_manifest,
    )
    records = [helpers._record_from_dict(record) for record in artifact.get("records", [])]
    plan = build_online_update_plan(
        records,
        args.algorithm,
        action_token_budget=ONLINE_PASS_ACTION_TOKEN_CAP,
        logprob_match_tolerance=args.old_current_tolerance,
    )
    output_dir = args.output_dir.expanduser().resolve()
    report_path = output_dir / "online_update_report.json"
    report: dict[str, Any] = {
        "schema_version": "m4_online_update_v1",
        "complete": False,
        "passed": False,
        "algorithm_id": args.algorithm,
        "study_seed": args.seed,
        "pass_index": args.pass_index,
        "source_artifact": str(artifact_path),
        "source_artifact_sha256": _sha256(artifact_path),
        "source_adapter": str(adapter_dir),
        "source_adapter_sha256": adapter_hash,
        "run_manifest": run_manifest,
        "update_plan": plan.audit_dict(),
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "clip_epsilon": args.clip_epsilon,
            "gradient_clip": args.gradient_clip,
            "old_current_tolerance": args.old_current_tolerance,
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.dry_run:
        report["dry_run"] = True
        report["complete"] = True
        report["passed"] = True
        _atomic_json_write(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    model = None
    trainable = None
    started = time.time()
    try:
        selected_groups = [group.replay_group for group in plan.groups if group.included]
        if not selected_groups:
            report["no_signal"] = True
            report["next_adapter"] = str(adapter_dir)
            report["next_adapter_sha256"] = adapter_hash
            report["complete"] = True
            report["passed"] = True
            return 0
        model, trainable = helpers._load_trainable_policy(config.base_model, adapter_dir)
        replay_audit = _audit_selected_groups(model, selected_groups, helpers)
        report["pre_update_replay_audit"] = replay_audit
        if replay_audit["max_abs_difference"] > args.old_current_tolerance:
            raise ValueError("old/current logprob mismatch before M4 update")
        report["training_runtime"] = helpers._enable_memory_efficient_training(model)

        def turn_logprobs(turn):
            return helpers._turn_logprobs(
                model, turn.prompt_token_ids, turn.completion_token_ids
            )

        report["optimizer"] = apply_m4_online_update(
            selected_groups,
            args.algorithm,
            [parameter for _, parameter in trainable],
            turn_logprobs,
            learning_rate=args.learning_rate,
            clip_epsilon=args.clip_epsilon,
            gradient_clip=args.gradient_clip,
        )
        model.eval()
        checkpoint_dir = output_dir / "updated_adapter"
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        report["next_adapter"] = str(checkpoint_dir)
        report["next_adapter_sha256"] = helpers._directory_sha256(checkpoint_dir)
        report["complete"] = True
        report["passed"] = True
        return 0
    finally:
        if model is not None:
            del model
        trainable = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report["elapsed_s"] = round(time.time() - started, 3)
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic_json_write(report_path, report)
        if report.get("complete"):
            if args.study_id == "m4_rlvr_v3":
                from miniwebwork.m4_v3_protocol import write_v3_manifest
                write_v3_manifest(output_dir / "resolved_run_manifest.json", report["run_manifest"])
            else:
                write_m4_run_manifest(output_dir / "resolved_run_manifest.json", report["run_manifest"])
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
