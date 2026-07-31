#!/usr/bin/env python3
"""M2.3-mini: Continue LoRA SFT training from seed_1234 checkpoint.

Resumes from the seed_1234 adapter and trains on the mixed dataset
(original m2_2r + no_solution patch) for 1-2 epochs at lr=5e-5.

Usage:
    python scripts/m2_3_mini_continue_train.py
    python scripts/m2_3_mini_continue_train.py --lr 3e-5 --epochs 2
    python scripts/m2_3_mini_continue_train.py --resume-from outputs/m2_2r/seed_1234/final_adapter

Slurm:
    sbatch scripts/slurm/m2_3_mini_train.sbatch
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset as _hf_load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

# LoRA config (same as M2.2R)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
DEFAULT_MAX_LENGTH = 8192
OUTPUT_DIR = PROJECT_ROOT / "data" / "sft" / "m2_3_mini"
CANONICAL_VERSION = "browser_agent_v2"
DEFAULT_LR = 5e-5           # Lower LR for continued training
DEFAULT_EPOCHS = 2           # 1-2 epochs per M2.3-mini spec
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16


def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id
    def collator(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        for key in batch:
            batch[key] = torch.tensor(batch[key], dtype=torch.long)
        return batch
    return collator


def tokenize_with_completion_mask(example, tokenizer, max_length):
    """Tokenize with completion-only labels."""
    messages = example["messages"]
    chat_kwargs = example.get("chat_template_kwargs", {})

    prompt_messages = messages[:-1]
    completion_text = messages[-1]["content"]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    prompt_ids = tokenizer(prompt_text)["input_ids"]
    prompt_len = len(prompt_ids)

    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    if prompt_len >= max_length:
        full_ids = prompt_ids[:max_length]
        labels = [-100] * len(full_ids)
    else:
        max_completion = max_length - prompt_len
        if len(completion_ids) > max_completion:
            completion_ids = completion_ids[:max_completion]
        full_ids = prompt_ids + completion_ids
        labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def prepare_dataset(ds, tokenizer, max_length):
    def tokenize_fn(example):
        return tokenize_with_completion_mask(example, tokenizer, max_length)
    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names, num_proc=1, desc="Tokenizing")
    print(f"  Tokenized: {len(tokenized)} samples")
    return tokenized


def compute_action_metrics(model, tokenizer, raw_dataset, max_length):
    """Compute teacher-forced action accuracy."""
    model.eval()
    exact_match = 0
    action_type_match = 0
    schema_valid = 0
    total = 0

    with torch.no_grad():
        for example in raw_dataset:
            messages = example["messages"]
            completion_text = messages[-1]["content"]

            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                **example.get("chat_template_kwargs", {}),
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)

            completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

            predicted_ids = []
            curr_ids = prompt_ids[0].tolist()
            for _ in range(len(completion_ids)):
                out = model(torch.tensor([curr_ids], device=model.device))
                next_id = out.logits[0, -1, :].argmax().item()
                predicted_ids.append(next_id)
                curr_ids.append(next_id)

            predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)

            try:
                predicted_action = json.loads(predicted_text)
                expected_action = json.loads(completion_text)
                schema_valid += 1
                if predicted_action.get("action_type") == expected_action.get("action_type"):
                    action_type_match += 1
                if predicted_text.strip() == completion_text.strip():
                    exact_match += 1
            except (json.JSONDecodeError, Exception):
                pass

            total += 1
            if total >= 30:
                break

    model.train()
    return {
        "exact_match": exact_match / max(total, 1),
        "action_type_match": action_type_match / max(total, 1),
        "schema_valid": schema_valid / max(total, 1),
    }


def train(base_model_path, train_dataset, eval_dataset, output_dir, resume_from,
          max_length, learning_rate, num_epochs, batch_size, grad_accum):
    """Train a single seed by continuing from an existing adapter."""
    seed = 1234
    print(f"\n{'='*60}")
    print(f"M2.3-mini Training: continuing from {resume_from}")
    print(f"  LR={learning_rate}, epochs={num_epochs}, batch={batch_size}, grad_accum={grad_accum}")
    print(f"{'='*60}")

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenize
    print("Tokenizing datasets...")
    train_dataset = prepare_dataset(train_dataset, tokenizer, max_length)
    eval_dataset_raw = eval_dataset
    eval_dataset = prepare_dataset(eval_dataset, tokenizer, max_length)

    # Load model with existing adapter
    print(f"Loading base model from {base_model_path}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    torch.cuda.empty_cache()
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Load existing LoRA adapter
    print(f"Loading LoRA adapter from: {resume_from}")
    model = PeftModel.from_pretrained(
        model, resume_from, torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()

    # Enable training mode (adapter was saved with inference_mode=True)
    model.enable_adapter_layers()
    model.print_trainable_parameters()

    # Training
    training_args = TrainingArguments(
        output_dir=str(seed_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        eval_strategy="no",
        save_on_each_node=False,
        report_to="none",
        max_grad_norm=1.0,
        disable_tqdm=False,
        remove_unused_columns=True,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
    )

    data_collator = make_collator(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start

    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"  Train loss: {train_result.metrics.get('train_loss', 'N/A')}")

    # Save adapter
    final_adapter_dir = seed_dir / "final_adapter"
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"  Adapter saved: {final_adapter_dir}")

    # Eval
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", float("nan"))
    print(f"  Eval loss: {eval_loss:.4f}")

    # Action metrics
    action_metrics = compute_action_metrics(model, tokenizer, eval_dataset_raw, max_length)
    print(f"  Exact match: {action_metrics['exact_match']:.1%}")
    print(f"  Action type match: {action_metrics['action_type_match']:.1%}")
    print(f"  Schema valid: {action_metrics['schema_valid']:.1%}")

    all_metrics = {
        "seed": seed,
        "phase": "m2_3_mini",
        "resumed_from": str(resume_from),
        "train_loss": train_result.metrics.get("train_loss", 0),
        "eval_loss": eval_loss,
        "exact_match": action_metrics["exact_match"],
        "action_type_match": action_metrics["action_type_match"],
        "schema_valid_rate": action_metrics["schema_valid"],
        "train_time_s": train_time,
        "train_samples": len(train_dataset),
        "valid_samples": len(eval_dataset),
        "lora_r": LORA_R,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "effective_batch_size": batch_size * grad_accum,
    }
    metrics_path = seed_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"  Metrics saved: {metrics_path}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="M2.3-mini continued SFT training")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "m2_3_mini")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to LoRA adapter to resume from (default: seed_1234 from m2_2r)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    args = parser.parse_args()

    # Find resume checkpoint
    if args.resume_from:
        resume_from = Path(args.resume_from)
    else:
        # Default: seed_1234 from m2_2r
        resume_from = PROJECT_ROOT / "outputs" / "m2_2r" / "seed_1234" / "final_adapter"
        if not resume_from.exists():
            # Try m2_2r as well
            resume_from = PROJECT_ROOT / "outputs" / "m2_2r" / "seed_1234" / "final_adapter"
        if not resume_from.exists():
            print(f"ERROR: No adapter found at {resume_from}")
            print("  Specify --resume-from <path> to point to an existing LoRA adapter")
            sys.exit(1)

    print(f"=== M2.3-mini Continued SFT Training ===")
    print(f"Base model: {args.base_model}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Resuming from: {resume_from}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}")

    # Load datasets
    print("\nLoading datasets...")
    data_path = str(args.data_dir)
    train_ds = _hf_load_dataset("json", data_files={"train": f"{data_path}/train.jsonl"}, split="train")
    valid_ds = _hf_load_dataset("json", data_files={"valid": f"{data_path}/valid.jsonl"}, split="valid")
    print(f"  Train: {len(train_ds)}, Valid: {len(valid_ds)}")

    # Check data composition
    orig_count = sum(1 for s in train_ds if s.get("source") == "oracle_expert")
    patch_count = sum(1 for s in train_ds if s.get("source") == "oracle_expert_m2_3")
    nosol_count = sum(1 for s in train_ds if s.get("task_type") == "no_feasible_product")
    print(f"  Train composition: {orig_count} original, {patch_count} patch, {nosol_count} no_solution")

    # Train
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = train(
        base_model_path=args.base_model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        output_dir=args.output_dir,
        resume_from=str(resume_from),
        max_length=args.max_length,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )

    # Summary
    summary = {
        "config": {
            "phase": "m2_3_mini",
            "base_model": args.base_model,
            "data_dir": str(args.data_dir),
            "prompt_version": CANONICAL_VERSION,
            "max_length": args.max_length,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "lora_r": LORA_R,
            "resumed_from": str(resume_from),
        },
        "seeds": [metrics],
    }
    summary_path = args.output_dir / "training_summary_m2_3_mini.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary saved: {summary_path}")

    print(f"\n=== Seed Result ===")
    print(f"  Seed {metrics['seed']}: train_loss={metrics['train_loss']:.4f}, "
          f"eval_loss={metrics['eval_loss']:.4f}, "
          f"exact_match={metrics['exact_match']:.1%}, "
          f"time={metrics['train_time_s']:.0f}s")


if __name__ == "__main__":
    main()
