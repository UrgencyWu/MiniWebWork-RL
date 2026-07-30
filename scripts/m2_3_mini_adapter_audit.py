#!/usr/bin/env python3
"""M2.3-mini: Adapter Load Contract Audit.

Systematically tests adapter loading through progressive stages to isolate
the exact layer where the CUDA IndexKernel error occurs.

Diagnostic stages:
  1. CPU-only load (base model + adapter)
  2. CPU forward pass with test input
  3. Move to CUDA
  4. Embedding lookup only
  5. Full forward pass
  6. generate(max_new_tokens=1)

Usage:
    python scripts/m2_3_mini_adapter_audit.py \
        --adapter outputs/m2_3_mini/seed_1234/final_adapter \
        --baseline-adapter outputs/m2_2r/seed_1234/final_adapter \
        --stage 6
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")
DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
STAGE_NAMES = {
    1: "CPU-only model load",
    2: "CPU forward pass",
    3: "Move to CUDA",
    4: "Embedding lookup",
    5: "Full forward pass",
    6: "generate(max_new_tokens=1)",
}


def cuda_checkpoint(name: str):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print(f"  [CUDA OK] {name}", flush=True)


def get_tokenizer_info(tokenizer, model):
    """Collect tokenizer/model vocabulary invariants."""
    info = {
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_vocab_size_config": tokenizer.vocab_size,
        "model_config_vocab_size": model.config.vocab_size,
        "embedding_num_embeddings": model.get_input_embeddings().num_embeddings,
        "embedding_dim": model.get_input_embeddings().embedding_dim,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    return info


def test_input_ids_range(model, input_ids):
    """Check that all input IDs are within embedding bounds."""
    emb = model.get_input_embeddings()
    min_id = input_ids.min().item()
    max_id = input_ids.max().item()
    vocab_size = emb.num_embeddings

    violations = []
    if min_id < 0:
        violations.append(f"min_id={min_id} < 0")
    if max_id >= vocab_size:
        violations.append(f"max_id={max_id} >= vocab_size({vocab_size})")

    result = {
        "min_id": min_id,
        "max_id": max_id,
        "vocab_size": vocab_size,
        "all_valid": len(violations) == 0,
        "violations": violations,
    }
    return result


def load_adapter_config(path):
    """Load and return adapter config."""
    config_path = Path(path) / "adapter_config.json"
    if not config_path.exists():
        return None
    return json.load(open(config_path))


def get_model_peft_status(model):
    """Extract PEFT model status."""
    status = {
        "type": type(model).__name__,
        "active_adapters": getattr(model, "active_adapters", None),
        "peft_config_keys": list(getattr(model, "peft_config", {}).keys()),
        "is_peft_model": hasattr(model, "peft_config"),
    }
    if hasattr(model, "base_model"):
        base = model.base_model
        status["base_model_type"] = type(base).__name__
        if hasattr(base, "base_model"):
            status["nested_peft"] = type(base.base_model).__name__
    return status


def stage_1_cpu_load(base_model_path, adapter_path):
    """Stage 1: Load base model + adapter on CPU."""
    print("\n--- Stage 1: CPU-only load ---")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading base model on CPU from {base_model_path}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cpu",
    )
    load_time = time.time() - t0
    print(f"  Base model loaded in {load_time:.1f}s")

    tok_info = get_tokenizer_info(tokenizer, model)
    print(f"  Tokenizer vocab_size: {tok_info['tokenizer_vocab_size']}")
    print(f"  Model config vocab_size: {tok_info['model_config_vocab_size']}")
    print(f"  Embedding num_embeddings: {tok_info['embedding_num_embeddings']}")
    print(f"  pad_token_id: {tok_info['pad_token_id']}")
    print(f"  eos_token_id: {tok_info['eos_token_id']}")

    if tok_info['tokenizer_vocab_size'] > tok_info['embedding_num_embeddings']:
        print(f"  *** CRITICAL: tokenizer ({tok_info['tokenizer_vocab_size']}) > embedding ({tok_info['embedding_num_embeddings']}) ***")
    elif tok_info['tokenizer_vocab_size'] < tok_info['embedding_num_embeddings']:
        diff = tok_info['embedding_num_embeddings'] - tok_info['tokenizer_vocab_size']
        print(f"  INFO: embedding has {diff} extra rows vs tokenizer")

    if adapter_path and Path(adapter_path).exists():
        print(f"  Loading adapter: {adapter_path}")
        adapter_config = load_adapter_config(adapter_path)
        if adapter_config:
            print(f"    target_modules: {adapter_config.get('target_modules')}")
            print(f"    inference_mode: {adapter_config.get('inference_mode')}")
            print(f"    trainable_token_indices: {adapter_config.get('trainable_token_indices')}")
            print(f"    modules_to_save: {adapter_config.get('modules_to_save')}")
            print(f"    peft_version: {adapter_config.get('peft_version')}")
            print(f"    base_model_name_or_path: {adapter_config.get('base_model_name_or_path')}")

        model = PeftModel.from_pretrained(
            model, adapter_path, torch_dtype=torch.bfloat16
        )
        peft_status = get_model_peft_status(model)
        print(f"  PEFT type: {peft_status['type']}")
        print(f"  Active adapters: {peft_status['active_adapters']}")
        print(f"  PEFT config keys: {peft_status['peft_config_keys']}")
        if peft_status.get('nested_peft'):
            print(f"  *** CRITICAL: NESTED PEFT DETECTED: {peft_status['nested_peft']} ***")
        if peft_status['is_peft_model']:
            cfg = model.peft_config
            for name, peft_cfg in cfg.items():
                print(f"  Adapter '{name}': type={type(peft_cfg).__name__}, r={peft_cfg.r}, target_modules={peft_cfg.target_modules}")

    return model, tokenizer, tok_info


def stage_2_cpu_forward(model, tokenizer):
    """Stage 2: Forward pass on CPU."""
    print("\n--- Stage 2: CPU forward pass ---")
    model.eval()
    text = "test"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    print(f"  Input: '{text}'")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  input_ids: {input_ids[0].tolist()}")

    id_check = test_input_ids_range(model, input_ids)
    print(f"  ID range: min={id_check['min_id']}, max={id_check['max_id']}, vocab={id_check['vocab_size']}")
    if not id_check['all_valid']:
        print(f"  *** ID RANGE VIOLATION: {id_check['violations']} ***")
        return None, None

    try:
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"  Forward OK: logits shape={output.logits.shape}")
        return model, input_ids
    except Exception as e:
        print(f"  *** FORWARD FAILED: {e} ***")
        traceback.print_exc()
        return None, None


def stage_3_move_to_cuda(model, input_ids):
    """Stage 3: Move model and input to CUDA."""
    print("\n--- Stage 3: Move to CUDA ---")
    if not torch.cuda.is_available():
        print("  No CUDA available, skipping")
        return model, input_ids

    device = torch.device("cuda:0")

    print(f"  Moving model to {device}...")
    try:
        model = model.to(device)
        model.eval()
        cuda_checkpoint("model moved to CUDA")
    except Exception as e:
        print(f"  *** MOVE TO CUDA FAILED: {e} ***")
        traceback.print_exc()
        return None, None

    print(f"  Moving input_ids to {device}...")
    try:
        input_ids = input_ids.to(device)
        cuda_checkpoint("input_ids moved to CUDA")
    except Exception as e:
        print(f"  *** MOVE INPUT FAILED: {e} ***")
        traceback.print_exc()
        return None, None

    return model, input_ids


def stage_4_embedding_lookup(model, input_ids):
    """Stage 4: Embedding lookup only."""
    print("\n--- Stage 4: Embedding lookup ---")
    try:
        emb = model.get_input_embeddings()
        with torch.no_grad():
            embedded = emb(input_ids)
        print(f"  Embedding OK: shape={embedded.shape}")
        torch.cuda.synchronize()
        return model
    except Exception as e:
        print(f"  *** EMBEDDING LOOKUP FAILED: {e} ***")
        traceback.print_exc()
        return None


def stage_5_full_forward(model, input_ids):
    """Stage 5: Full forward pass."""
    print("\n--- Stage 5: Full forward pass ---")
    try:
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"  Forward OK: logits shape={output.logits.shape}")
        torch.cuda.synchronize()
        return model
    except Exception as e:
        print(f"  *** FULL FORWARD FAILED: {e} ***")
        traceback.print_exc()
        return None


def stage_6_generate(model, tokenizer):
    """Stage 6: generate(max_new_tokens=1)."""
    print("\n--- Stage 6: generate(max_new_tokens=1) ---")
    try:
        text = "test"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Generate OK: output='{new_text}'")
        torch.cuda.synchronize()
        return model
    except Exception as e:
        print(f"  *** GENERATE FAILED: {e} ***")
        traceback.print_exc()
        return None


def run_diagnostic(base_model_path, adapter_path, max_stage=6):
    """Run progressive diagnostic stages."""
    print(f"{'='*60}")
    print(f"M2.3-mini Adapter Load Contract Audit")
    print(f"{'='*60}")
    print(f"Base model: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    try:
        import peft
        print(f"PEFT version: {peft.__version__}")
    except:
        print(f"PEFT version: unknown")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = None
    tokenizer = None
    input_ids = None

    # Stage 1
    try:
        model, tokenizer, tok_info = stage_1_cpu_load(base_model_path, adapter_path)
        if model is None:
            return ["Stage 1: CPU load returned None"]
    except Exception as e:
        print(f"  *** Stage 1 EXCEPTION: {e} ***")
        traceback.print_exc()
        return [f"Stage 1: {e}"]

    # Stage 2
    if max_stage >= 2:
        try:
            model, input_ids = stage_2_cpu_forward(model, tokenizer)
            if model is None or input_ids is None:
                return ["Stage 2: CPU forward returned None"]
        except Exception as e:
            print(f"  *** Stage 2 EXCEPTION: {e} ***")
            traceback.print_exc()
            return [f"Stage 2: {e}"]

    # Stage 3
    if max_stage >= 3:
        try:
            model, input_ids = stage_3_move_to_cuda(model, input_ids)
            if model is None or input_ids is None:
                return ["Stage 3: Move to CUDA returned None"]
        except Exception as e:
            print(f"  *** Stage 3 EXCEPTION: {e} ***")
            traceback.print_exc()
            return [f"Stage 3: {e}"]

    # Stage 4
    if max_stage >= 4:
        try:
            model = stage_4_embedding_lookup(model, input_ids)
            if model is None:
                return ["Stage 4: Embedding lookup returned None"]
        except Exception as e:
            print(f"  *** Stage 4 EXCEPTION: {e} ***")
            traceback.print_exc()
            return [f"Stage 4: {e}"]

    # Stage 5
    if max_stage >= 5:
        try:
            model = stage_5_full_forward(model, input_ids)
            if model is None:
                return ["Stage 5: Full forward returned None"]
        except Exception as e:
            print(f"  *** Stage 5 EXCEPTION: {e} ***")
            traceback.print_exc()
            return [f"Stage 5: {e}"]

    # Stage 6
    if max_stage >= 6:
        try:
            model = stage_6_generate(model, tokenizer)
            if model is None:
                return ["Stage 6: generate() returned None"]
        except Exception as e:
            print(f"  *** Stage 6 EXCEPTION: {e} ***")
            traceback.print_exc()
            return [f"Stage 6: {e}"]

    print(f"\n{'='*60}")
    print(f"ALL STAGES PASSED")
    print(f"{'='*60}")
    return []


def main():
    parser = argparse.ArgumentParser(description="M2.3-mini Adapter Load Contract Audit")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to LoRA adapter to test")
    parser.add_argument("--baseline-adapter", type=str, default=None,
                        help="Path to baseline adapter for comparison")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--stage", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--baseline-first", action="store_true",
                        help="Test baseline adapter first")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        sys.exit(1)

    adapters_to_test = []
    if args.baseline_adapter and Path(args.baseline_adapter).exists():
        adapters_to_test.append(("BASELINE (M2.2R)", args.baseline_adapter))
    adapters_to_test.append(("NEW (M2.3-mini)", str(adapter_path)))

    if args.baseline_first:
        adapters_to_test.reverse()

    all_results = {}
    for label, path in adapters_to_test:
        print(f"\n{'#'*60}")
        print(f"# TESTING: {label}")
        print(f"# Path: {path}")
        print(f"{'#'*60}")
        failures = run_diagnostic(args.base_model, path, max_stage=args.stage)
        all_results[label] = {"failures": failures, "passed": len(failures) == 0}

    # Summary
    print(f"\n{'='*60}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*60}")
    for label, result in all_results.items():
        status = "PASS" if result["passed"] else f"FAIL: {result['failures']}"
        print(f"  {label}: {status}")


if __name__ == "__main__":
    main()
