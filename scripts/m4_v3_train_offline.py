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

from miniwebwork.m4_protocol import (
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4RunConfig,
    ONLINE_PASS_ACTION_TOKEN_CAP,
    ONLINE_PASSES,
    M4_TRAIN_TASK_COUNT,
)
from miniwebwork.m4_v3_protocol import (
    V3_MAX_SEQUENCE_LENGTH,
    V3_STUDY_ID,
    V3_TARGET_SUPERVISED_COMPLETION_TOKENS,
    assert_v3_initial_adapter,
    build_v3_run_manifest,
    write_v3_manifest,
)


def _validate_rsft_source_manifest(train_data: Path) -> dict:
    path = train_data / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("v3 RSFT training requires its tokenized manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "m4_rsft_tokenized_v3",
        "dataset_id": "m4_rsft_tokenized_v3",
        "algorithm": "rsft",
        "split": "train",
        "source_passes": ONLINE_PASSES,
        "source_pass_indices": [1, 2],
        "source_task_universe_count": M4_TRAIN_TASK_COUNT,
        "source_pass_action_token_cap": ONLINE_PASS_ACTION_TOKEN_CAP,
        "target_supervised_completion_tokens": V3_TARGET_SUPERVISED_COMPLETION_TOKENS,
        "max_sequence_length": V3_MAX_SEQUENCE_LENGTH,
        "zero_completion_label_sample_fraction": 0.0,
        "packing": "deterministic_sorted_verified_rows_repeated_without_partial_examples",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"v3 RSFT source manifest mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    artifact_paths = payload.get("source_artifacts")
    artifact_hashes = payload.get("source_artifact_sha256")
    if not isinstance(artifact_paths, list) or len(artifact_paths) != ONLINE_PASSES:
        raise ValueError("v3 RSFT source manifest must name both pass artifacts")
    if not isinstance(artifact_hashes, list) or len(artifact_hashes) != ONLINE_PASSES:
        raise ValueError("v3 RSFT source manifest lacks both artifact hashes")
    forbidden = ("m4_invalidated", "m4_v2_runs", "443d8d7")
    if any(any(fragment in str(path).lower() for fragment in forbidden) for path in artifact_paths):
        raise ValueError("v3 RSFT source manifest references an excluded artifact namespace")
    per_pass = payload.get("source_collected_action_tokens_per_pass")
    if (
        not isinstance(per_pass, list)
        or len(per_pass) != ONLINE_PASSES
        or any(not isinstance(value, int) or not 0 < value <= ONLINE_PASS_ACTION_TOKEN_CAP for value in per_pass)
    ):
        raise ValueError("v3 RSFT source manifest has invalid per-pass action-token accounting")
    minimum = payload.get("min_nonzero_completion_label_tokens")
    realized = payload.get("realized_supervised_completion_tokens")
    shortfall = payload.get("shortfall_supervised_completion_tokens")
    if not isinstance(minimum, int) or minimum <= 0:
        raise ValueError("v3 RSFT source manifest lacks a positive minimum label count")
    if not isinstance(realized, int) or realized <= 0 or not isinstance(shortfall, int):
        raise ValueError("v3 RSFT source manifest has invalid realized supervision")
    if shortfall < 0 or shortfall >= minimum:
        raise ValueError("v3 RSFT source manifest does not meet the 250k target tolerance")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "payload": payload}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_data = args.train_data_dir.expanduser().resolve()
    validation_data = args.validation_data_dir.expanduser().resolve()
    if not train_data.is_dir() or not validation_data.is_dir():
        raise FileNotFoundError("v3 offline training requires train and validation directories")
    rsft_source_manifest = None
    if args.algorithm == "rsft":
        rsft_source_manifest = _validate_rsft_source_manifest(train_data)
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
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
        command.extend(["--resume-from-checkpoint", str(checkpoint)])
    manifest["offline_training"] = {
        "algorithm": args.algorithm,
        "train_data_dir": str(train_data),
        "validation_data_dir": str(validation_data),
        "initial_adapter": canonical,
        "target_supervised_completion_tokens": V3_TARGET_SUPERVISED_COMPLETION_TOKENS,
        "max_sequence_length": V3_MAX_SEQUENCE_LENGTH,
        "max_zero_completion_label_fraction": 0.0,
        "command": command,
        "resume_capable": True,
    }
    if rsft_source_manifest is not None:
        manifest["offline_training"]["rsft_source_manifest"] = rsft_source_manifest
    manifest["initial_adapter"] = canonical["path"]
    manifest["initial_adapter_sha256"] = canonical["sha256"]
    manifest["canonical_initial_adapter"] = canonical
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
    minimum = statistics.get("min_nonzero_completion_label_tokens")
    shortfall = V3_TARGET_SUPERVISED_COMPLETION_TOKENS - realized if isinstance(realized, int) else None
    if (
        not isinstance(realized, int)
        or realized <= 0
        or not isinstance(minimum, int)
        or minimum <= 0
        or not isinstance(shortfall, int)
        or shortfall < 0
        or shortfall >= minimum
    ):
        raise ValueError("v3 offline training did not meet the auditable 250k target tolerance")
    gate = {
        "schema_version": V3_STUDY_ID + "_offline_gate_v1",
        "study_id": V3_STUDY_ID,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "target_supervised_completion_tokens": V3_TARGET_SUPERVISED_COMPLETION_TOKENS,
        "realized_supervised_completion_tokens": realized,
        "shortfall_supervised_completion_tokens": shortfall,
        "min_nonzero_completion_label_tokens": minimum,
        "zero_completion_label_sample_fraction": 0.0,
        "supervision_audit": str(audit),
    }
    if rsft_source_manifest is not None:
        gate["rsft_source_manifest"] = rsft_source_manifest
    write_v3_manifest(output_dir / "v3_training_gate.json", gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
