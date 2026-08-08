#!/usr/bin/env python3
"""Run the disposable focused-study SFT GPU preflight.

Formal training remains deliberately unavailable until the final readiness
manifest opens the frozen study gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from transformers import AutoTokenizer

from miniwebwork.long_horizon_rl.contracts import (
    atomic_write_json,
    directory_sha256,
    sha256_file,
)
from miniwebwork.long_horizon_rl.sft_trainer import (
    benchmark_microbatches,
    load_sft_examples,
    load_trainable_lora_model,
    summarize_sft_examples,
    train_sft,
    validate_sft_training_inputs,
)
from miniwebwork.m4_long_horizon_protocol import (
    SFT_PREFLIGHT_MAXIMUM_OPTIMIZER_UPDATES,
    assert_dataset_binding,
    assert_formal_submission_closed,
    assert_preflight_output,
    load_study_manifest,
)

DEFAULT_BASE_MODEL = Path("/data/share/model/Qwen3.5-4B")
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sft" / "m4_long_horizon_verified_v2"


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_clean_tracked_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("SFT preflight requires a clean tracked worktree")


def _assert_loaded_examples_match_token_audit(examples, expected: dict, split: str) -> dict:
    summary = summarize_sft_examples(examples)
    audit = expected["splits"][split]
    bindings = {
        "sample_count": "sample_count",
        "unique_sample_count": "unique_sample_count",
        "completion_label_tokens": "effective_completion_label_tokens",
        "forward_tokens": "total_forward_tokens",
        "maximum_forward_tokens": "maximum_forward_sequence_tokens",
        "zero_label_count": "zero_completion_label_sample_count",
        "truncated_count": "truncated_sample_count",
    }
    drift = {
        local: {"loaded": summary[local], "audited": audit[remote]}
        for local, remote in bindings.items()
        if summary[local] != audit[remote]
    }
    if drift:
        raise ValueError(f"loaded {split} examples drift from token audit: {drift}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight",), default="preflight")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataloader-workers", type=int, default=2, choices=range(0, 4))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    study = load_study_manifest()
    assert_formal_submission_closed(study["payload"])
    dataset_binding = assert_dataset_binding(study["payload"])
    output_dir = assert_preflight_output(args.output_dir, study["payload"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"SFT preflight output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _assert_clean_tracked_worktree()
    git_sha = _git_sha()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SFT preflight requires exactly one visible CUDA GPU")

    input_binding = validate_sft_training_inputs(args.data_dir, args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model.expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("focused SFT tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train_examples = load_sft_examples(args.data_dir, "train", tokenizer)
    dev_examples = load_sft_examples(args.data_dir, "dev", tokenizer)
    loaded_audit = {
        "train": _assert_loaded_examples_match_token_audit(train_examples, input_binding, "train"),
        "dev": _assert_loaded_examples_match_token_audit(dev_examples, input_binding, "dev"),
    }
    invocation = {
        "schema_version": "m4_long_horizon_sft_preflight_invocation_v1",
        "study_id": study["payload"]["study_id"],
        "mode": args.mode,
        "formal_training": False,
        "disposable_adapter": True,
        "git_sha": git_sha,
        "study_manifest_sha256": study["sha256"],
        "dataset_binding": dataset_binding,
        "input_binding": input_binding,
        "loaded_audit": loaded_audit,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_properties(0).name,
        "maximum_optimizer_updates": SFT_PREFLIGHT_MAXIMUM_OPTIMIZER_UPDATES,
        "dataloader_workers": args.dataloader_workers,
    }
    atomic_write_json(output_dir / "invocation.json", invocation)

    benchmark_path = output_dir / "microbatch_benchmark.json"
    try:
        model = load_trainable_lora_model(args.base_model)
        benchmark = benchmark_microbatches(
            model=model,
            examples=train_examples,
            tokenizer=tokenizer,
            output=benchmark_path,
            corpus_manifest_sha256=input_binding["manifest_sha256"],
            token_audit_sha256=input_binding["token_audit_sha256"],
        )
        training = train_sft(
            model=model,
            tokenizer=tokenizer,
            train_examples=train_examples,
            dev_examples=dev_examples,
            output_dir=output_dir / "disposable_training_smoke",
            microbatch_size=benchmark["selected_microbatch_size"],
            workers=args.dataloader_workers,
            maximum_optimizer_updates=SFT_PREFLIGHT_MAXIMUM_OPTIMIZER_UPDATES,
            maximum_epochs=1,
        )
    except Exception as exc:
        failure = {
            "schema_version": "m4_long_horizon_sft_preflight_failure_v1",
            "study_id": study["payload"]["study_id"],
            "complete": True,
            "passed": False,
            "formal_training": False,
            "git_sha": git_sha,
            "exception_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "traceback": traceback.format_exc()[-8000:],
            "elapsed_s": time.monotonic() - started,
            "benchmark_sha256": sha256_file(benchmark_path) if benchmark_path.is_file() else None,
        }
        atomic_write_json(output_dir / "preflight_failure.json", failure)
        raise
    final_adapter = Path(training["final_adapter"])
    report = {
        "schema_version": "m4_long_horizon_sft_preflight_report_v1",
        "study_id": study["payload"]["study_id"],
        "complete": True,
        "passed": True,
        "formal_training": False,
        "disposable_adapter": True,
        "git_sha": git_sha,
        "study_manifest_sha256": study["sha256"],
        "invocation_sha256": sha256_file(output_dir / "invocation.json"),
        "microbatch_benchmark_sha256": sha256_file(output_dir / "microbatch_benchmark.json"),
        "training_report_sha256": sha256_file(output_dir / "disposable_training_smoke" / "training_report.json"),
        "disposable_adapter_sha256": directory_sha256(final_adapter),
        "selected_microbatch_size": benchmark["selected_microbatch_size"],
        "optimizer_updates": training["optimizer_updates"],
        "stop_reason": training["stop_reason"],
        "elapsed_s": time.monotonic() - started,
    }
    if (
        report["optimizer_updates"] != SFT_PREFLIGHT_MAXIMUM_OPTIMIZER_UPDATES
        or report["stop_reason"] != "preflight_optimizer_update_cap"
    ):
        raise RuntimeError("SFT preflight did not execute the exact disposable update budget")
    atomic_write_json(output_dir / "preflight_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
