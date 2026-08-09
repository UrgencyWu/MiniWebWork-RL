#!/usr/bin/env python3
"""Run or safely resume the one shared formal SFT training job."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from transformers import AutoTokenizer

from miniwebwork.long_horizon_rl.formal_sft import (
    FORMAL_SFT_MANIFEST_NAME,
    FORMAL_SFT_ROOT,
    build_formal_sft_run_config,
    finalize_formal_sft,
    prepare_sft_attempt,
    validate_formal_sft_manifest,
)
from miniwebwork.long_horizon_rl.formal_invocation import finish_formal_invocation, start_formal_invocation
from miniwebwork.long_horizon_rl.sft_trainer import (
    load_sft_examples,
    load_trainable_lora_model,
    summarize_sft_examples,
    train_sft,
    validate_sft_training_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "sft" / "m4_long_horizon_verified_v2")
    parser.add_argument("--output-root", type=Path, default=FORMAL_SFT_ROOT)
    parser.add_argument("--dataloader-workers", type=int, default=2, choices=range(0, 4))
    return parser.parse_args()


def _audit_loaded(examples, expected: dict, split: str) -> dict:
    summary = summarize_sft_examples(examples)
    audit = expected["splits"][split]
    fields = {
        "sample_count": "sample_count",
        "unique_sample_count": "unique_sample_count",
        "completion_label_tokens": "effective_completion_label_tokens",
        "forward_tokens": "total_forward_tokens",
        "maximum_forward_tokens": "maximum_forward_sequence_tokens",
        "zero_label_count": "zero_completion_label_sample_count",
        "truncated_count": "truncated_sample_count",
    }
    drift = {local: [summary[local], audit[remote]] for local, remote in fields.items() if summary[local] != audit[remote]}
    if drift:
        raise ValueError(f"loaded formal SFT {split} audit drift: {drift}")
    return summary


def main() -> int:
    args = parse_args()
    config = build_formal_sft_run_config(
        output_root=args.output_root,
        expected_git_sha=args.expected_git_sha,
        readiness_path=args.readiness,
        base_model=args.base_model,
        data_dir=args.data_dir,
        dataloader_workers=args.dataloader_workers,
    )
    root = Path(config["output_root"])
    already_complete = (root / FORMAL_SFT_MANIFEST_NAME).is_file()
    invocation = start_formal_invocation(
        root=root, phase="shared_sft", method="verified_sft", seed=20260801,
        git_sha=config["git_sha"], expected_cpus=4, expected_gpus=1,
        log_stem="m4_lh_formal_sft",
        telemetry_path=f"logs/m4_lh_formal_sft_{os.environ.get('SLURM_JOB_ID', '')}_gpu.csv",
    )
    try:
        if already_complete:
            result = validate_formal_sft_manifest(root)
        elif not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("formal SFT requires exactly one visible CUDA GPU")
        else:
            input_binding = validate_sft_training_inputs(args.data_dir, args.base_model)
            tokenizer = AutoTokenizer.from_pretrained(str(args.base_model.resolve()), local_files_only=True, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError("formal SFT tokenizer has neither pad nor EOS token")
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            train_examples = load_sft_examples(args.data_dir, "train", tokenizer)
            dev_examples = load_sft_examples(args.data_dir, "dev", tokenizer)
            loaded_audit = {
                "train": _audit_loaded(train_examples, input_binding, "train"),
                "dev": _audit_loaded(dev_examples, input_binding, "dev"),
            }
            attempt, completed_training = prepare_sft_attempt(root)
            if not completed_training:
                model = load_trainable_lora_model(args.base_model)
                train_sft(
                    model=model,
                    tokenizer=tokenizer,
                    train_examples=train_examples,
                    dev_examples=dev_examples,
                    output_dir=attempt / "training",
                    microbatch_size=8,
                    workers=args.dataloader_workers,
                )
            result = finalize_formal_sft(
                root=root,
                config=config,
                attempt=attempt,
                input_binding=input_binding,
                loaded_audit=loaded_audit,
                completed_attempt_reused=completed_training,
            )
    except BaseException:
        finish_formal_invocation(invocation, status="ERROR")
        raise
    finish_formal_invocation(invocation, status="WORKLOAD_COMPLETE")
    print(json.dumps({"status": "already_complete" if already_complete else "complete", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
