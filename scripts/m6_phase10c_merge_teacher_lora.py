#!/usr/bin/env python3
"""Merge the frozen Phase10-C Qwen3.5-35B LoRA into an HF checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256  # noqa: E402
from miniwebwork.long_horizon_rl.model_manifest import (  # noqa: E402
    build_base_model_manifest,
    validate_base_model_manifest,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def canonical_runtime_lora_key(key: str) -> str:
    """Map PEFT's portable adapter key to its instantiated default-adapter key."""

    for component in ("lora_A", "lora_B"):
        suffix = f".{component}.weight"
        if key.endswith(suffix):
            return f"{key[:-len(suffix)]}.{component}.default.weight"
    raise ValueError(f"Phase10-C merge found a non-LoRA tensor: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    args = parser.parse_args()

    base_model = args.base_model.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    report = args.report.expanduser().resolve()
    _require(args.expected_git_sha == _git_sha(), "Phase10-C merge Git drift")
    _require(base_model.is_dir() and adapter.is_dir(), "Phase10-C merge input is missing")
    _require(not output.exists() and not manifest.exists() and not report.exists(), "Phase10-C merge refuses overwrite")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 2, "Phase10-C merge requires two GPUs")

    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    max_memory = {
        index: int(torch.cuda.get_device_properties(index).total_memory * 0.45)
        for index in range(torch.cuda.device_count())
    }
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
    )
    # PEFT 0.19.1 attempts an incompatible Transformers WeightConverter path
    # for this Qwen3.5 checkpoint.  Instantiate the exact saved LoRA contract,
    # then load only its portable A/B tensors into the default adapter slots.
    model = get_peft_model(base, LoraConfig.from_pretrained(str(adapter)))
    portable = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    runtime = {canonical_runtime_lora_key(key): value for key, value in portable.items()}
    expected = {
        key for key in model.state_dict()
        if key.endswith(".lora_A.default.weight") or key.endswith(".lora_B.default.weight")
    }
    _require(set(runtime) == expected, "Phase10-C merge LoRA key roster drift")
    incompatible = model.load_state_dict(runtime, strict=False)
    _require(not incompatible.unexpected_keys, "Phase10-C merge produced unexpected LoRA keys")
    merged = model.merge_and_unload(safe_merge=True, progressbar=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), local_files_only=True, trust_remote_code=True)

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.mkdir(parents=True)
    try:
        merged.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(temporary)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    built = build_base_model_manifest(base_model=output, destination=manifest)
    checked = validate_base_model_manifest(path=manifest, expected_base_model=output, verify_files=True)
    payload = {
        "schema_version": "m6_phase10c_merged_teacher_lora_v1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "producer_git_sha": args.expected_git_sha,
        "base_model": str(base_model),
        "base_model_directory_sha256": directory_sha256(base_model),
        "adapter": str(adapter),
        "adapter_directory_sha256": directory_sha256(adapter),
        "merged_model": str(output),
        "merged_model_directory_sha256": directory_sha256(output),
        "merged_model_manifest": str(manifest),
        "merged_model_manifest_sha256": built["sha256"],
        "merged_model_functional_sha256": checked["payload"]["functional_file_set_sha256"],
        "merge_method": "peft_merge_and_unload_safe_v1",
        "adapter_load_method": "portable_lora_ab_direct_v1",
        "adapter_tensor_count": len(runtime),
    }
    atomic_write_json(report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
