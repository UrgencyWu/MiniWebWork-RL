#!/usr/bin/env python3
"""Progressive PEFT adapter load contract audit.

Run one adapter per process.  The script never rewrites CUDA visibility and
never compares multiple large policies in one process; Slurm is responsible for
device allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_artifact(adapter_dir: Path) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {weights_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    tensor_count = 0
    non_finite: list[str] = []
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as archive:
        for key in archive.keys():
            tensor = archive.get_tensor(key)
            tensor_count += 1
            shapes[key] = list(tensor.shape)
            dtypes[key] = str(tensor.dtype)
            if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                non_finite.append(key)

    if tensor_count == 0:
        raise ValueError("Adapter safetensors archive is empty")
    if non_finite:
        raise ValueError(f"Adapter contains non-finite tensors: {non_finite[:10]}")

    return {
        "config_sha256": _sha256(config_path),
        "weights_sha256": _sha256(weights_path),
        "tensor_count": tensor_count,
        "non_finite_tensors": non_finite,
        "target_modules": sorted(config.get("target_modules") or []),
        "rank": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "modules_to_save": config.get("modules_to_save"),
        "trainable_token_indices": config.get("trainable_token_indices"),
        "tensor_shapes": shapes,
        "tensor_dtypes": dtypes,
    }


def load_policy(base_model_path: str, adapter_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    started = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        torch_dtype=torch.bfloat16,
    )
    model.enable_adapter_layers()
    model.eval()
    if not getattr(model, "active_adapters", None):
        raise RuntimeError("Adapter loaded without an active PEFT adapter")
    return model, tokenizer, time.time() - started


def audit_tokenizer_model(tokenizer, model) -> dict:
    embedding_rows = int(model.get_input_embeddings().num_embeddings)
    tokenizer_size = len(tokenizer)
    if tokenizer_size > embedding_rows:
        raise ValueError(
            f"Tokenizer size {tokenizer_size} exceeds embedding rows {embedding_rows}"
        )
    return {
        "tokenizer_size": tokenizer_size,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "model_vocab_size": int(model.config.vocab_size),
        "embedding_rows": embedding_rows,
        "embedding_dim": int(model.get_input_embeddings().embedding_dim),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def run_forward(model, tokenizer, device: torch.device) -> dict:
    inputs = tokenizer("adapter contract audit", return_tensors="pt").to(device)
    ids = inputs.input_ids
    embedding_rows = int(model.get_input_embeddings().num_embeddings)
    minimum = int(ids.min().item())
    maximum = int(ids.max().item())
    if minimum < 0 or maximum >= embedding_rows:
        raise ValueError(
            f"Input token range [{minimum}, {maximum}] outside embedding rows {embedding_rows}"
        )

    with torch.inference_mode():
        embedded = model.get_input_embeddings()(ids)
        output = model(**inputs, use_cache=False)
    if not torch.isfinite(output.logits).all().item():
        raise ValueError("Forward logits contain NaN or Inf")
    return {
        "input_shape": list(ids.shape),
        "input_id_min": minimum,
        "input_id_max": maximum,
        "embedding_shape": list(embedded.shape),
        "logits_shape": list(output.logits.shape),
    }


def run_generate(model, tokenizer, device: torch.device) -> dict:
    inputs = tokenizer("adapter contract audit", return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = output[0, inputs.input_ids.shape[1] :]
    return {
        "generated_token_ids": [int(value) for value in new_ids.cpu().tolist()],
        "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PEFT adapter load contract audit")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    adapter_dir = args.adapter.expanduser().resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    report = {
        "schema_version": "2.0",
        "adapter_path": str(adapter_dir),
        "base_model": args.base_model,
        "requested_device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "artifact": audit_artifact(adapter_dir),
    }

    model = tokenizer = None
    try:
        model, tokenizer, load_time = load_policy(args.base_model, adapter_dir)
        report["load_time_s"] = round(load_time, 3)
        report["active_adapters"] = list(model.active_adapters)
        report["tokenizer_model"] = audit_tokenizer_model(tokenizer, model)
        report["cpu_forward"] = run_forward(model, tokenizer, torch.device("cpu"))

        if args.device == "cuda":
            model = model.to("cuda:0")
            model.eval()
            torch.cuda.synchronize()
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["cuda_forward"] = run_forward(model, tokenizer, torch.device("cuda:0"))
            report["cuda_generate"] = run_generate(model, tokenizer, torch.device("cuda:0"))
            torch.cuda.synchronize()
            report["cuda_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1024**3,
                3,
            )
        else:
            report["cpu_generate"] = run_generate(model, tokenizer, torch.device("cpu"))

        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if args.output is not None:
            output_path = args.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
