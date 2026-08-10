#!/usr/bin/env python3
"""Run the resumable one-epoch M5 WebShop SFT GPU preflight."""

from __future__ import annotations

import argparse
import json
import math
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

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_file, sha256_json
from miniwebwork.m5_webshop_protocol import load_protocol
from miniwebwork.webshop_rl.sft_training import (
    SFTConfig,
    benchmark_microbatches,
    load_recovery,
    load_sft_examples,
    load_trainable_lora_model,
    summarize_examples,
    train_one_epoch,
    validate_sft_training_inputs,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_clean() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(not status, "M5 SFT preflight requires a clean tracked worktree")


def _assert_loaded(summary: dict, audited: dict, split: str) -> None:
    fields = {
        "sample_count": "sample_count",
        "unique_sample_count": "unique_sample_count",
        "task_count": "task_count",
        "completion_label_tokens": "effective_completion_label_tokens_per_epoch",
        "forward_tokens": "total_forward_tokens_per_epoch",
        "maximum_forward_tokens": "maximum_forward_sequence_tokens",
        "zero_label_count": "zero_completion_label_sample_count",
        "truncated_count": "truncated_sample_count",
    }
    drift = {local: (summary[local], audited[remote]) for local, remote in fields.items() if summary[local] != audited[remote]}
    _require(not drift, f"loaded M5 {split} examples drift from token audit: {drift}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    _assert_clean()
    _require(_git_sha() == protocol["git_sha"], "M5 SFT preflight Git/protocol lineage drift")
    _require(protocol["payload"]["formal_submission_allowed"] is False, "M5 preflight cannot run after in-place formal authorization")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M5 SFT preflight requires exactly one visible GPU")
    binding = validate_sft_training_inputs(args.data_dir, args.base_model)
    config = SFTConfig.from_protocol(protocol["payload"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model.expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        _require(tokenizer.eos_token_id is not None, "M5 tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train = load_sft_examples(args.data_dir, "train", tokenizer, config)
    dev = load_sft_examples(args.data_dir, "dev", tokenizer, config)
    loaded = {"train": summarize_examples(train, config), "dev": summarize_examples(dev, config)}
    _assert_loaded(loaded["train"], binding["splits"]["train"], "train")
    _assert_loaded(loaded["dev"], binding["splits"]["dev"], "dev")
    invocation = {
        "schema_version": "m5_webshop_sft_preflight_invocation_v1",
        "formal_training": False,
        "disposable_until_authorized": True,
        "git_sha": binding["git_sha"],
        "protocol_sha256": binding["protocol_sha256"],
        "input_binding": binding,
        "loaded_audit": loaded,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_properties(0).name,
    }
    invocation["content_sha256"] = sha256_json(invocation)
    atomic_write_json(output / "invocation.json", invocation)
    training_root = output / "training"
    completed_report = training_root / "training_report.json"
    if completed_report.is_file():
        existing = json.loads(completed_report.read_text(encoding="utf-8"))
        expected = dict(existing)
        observed = expected.pop("content_sha256", None)
        _require(observed == sha256_json(expected) and existing.get("complete") is True, "existing M5 SFT report is incomplete or corrupt")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return
    recovery = load_recovery(training_root, binding)
    benchmark_path = output / "microbatch_benchmark.json"
    try:
        model = load_trainable_lora_model(
            args.base_model,
            config,
            resume_adapter=recovery["adapter"] if recovery else None,
        )
        if recovery is None:
            benchmark = benchmark_microbatches(
                model=model,
                examples=train,
                tokenizer=tokenizer,
                config=config,
                output_path=benchmark_path,
                input_binding=binding,
            )
        else:
            _require(benchmark_path.is_file(), "M5 SFT recovery lacks microbatch benchmark")
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            expected = dict(benchmark)
            observed = expected.pop("content_sha256", None)
            _require(observed == sha256_json(expected), "M5 SFT benchmark self-hash drift")
        training = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            train_examples=train,
            dev_examples=dev,
            config=config,
            input_binding=binding,
            output_dir=training_root,
            microbatch_size=int(benchmark["selected_microbatch_size"]),
            recovery=recovery,
        )
        token_floor = int(protocol["payload"]["sft"]["minimum_effective_completion_label_token_exposure"])
        checks = {
            "one_complete_epoch": training["train_completion_label_tokens"] == loaded["train"]["completion_label_tokens"],
            "token_floor": training["train_completion_label_tokens"] >= token_floor,
            "finite_train_nll": math.isfinite(float(training["train_nll"])),
            "finite_dev_nll": math.isfinite(float(training["dev"]["dev_nll"])),
            "zero_labels": loaded["train"]["zero_label_count"] == loaded["dev"]["zero_label_count"] == 0,
            "zero_truncation": loaded["train"]["truncated_count"] == loaded["dev"]["truncated_count"] == 0,
        }
        _require(all(checks.values()), f"M5 SFT preflight gates failed: {checks}")
        final_adapter = Path(training["final_adapter"])
        report = {
            "schema_version": "m5_webshop_sft_preflight_report_v1",
            "complete": True,
            "passed": True,
            "formal_training": False,
            "git_sha": binding["git_sha"],
            "protocol_sha256": binding["protocol_sha256"],
            "invocation_sha256": sha256_file(output / "invocation.json"),
            "benchmark_sha256": sha256_file(benchmark_path),
            "training_report_sha256": sha256_file(completed_report),
            "adapter_sha256": directory_sha256(final_adapter),
            "selected_microbatch_size": benchmark["selected_microbatch_size"],
            "optimizer_updates": training["optimizer_updates"],
            "resumed": training["resumed"],
            "checks": checks,
            "elapsed_s": time.monotonic() - started,
        }
        report["content_sha256"] = sha256_json(report)
        atomic_write_json(output / "preflight_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
    except Exception as exc:
        failure = {
            "schema_version": "m5_webshop_sft_preflight_failure_v1",
            "complete": True,
            "passed": False,
            "git_sha": binding["git_sha"],
            "protocol_sha256": binding["protocol_sha256"],
            "exception_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-12000:],
            "elapsed_s": time.monotonic() - started,
        }
        failure["content_sha256"] = sha256_json(failure)
        atomic_write_json(output / "preflight_failure.json", failure)
        raise


if __name__ == "__main__":
    main()
