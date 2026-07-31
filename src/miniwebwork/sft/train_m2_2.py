"""M2.2 LoRA SFT Trainer: completion-only training on expert browser actions.

Uses standard HuggingFace Trainer with pre-tokenized data:
  Dataset is pre-tokenized with input_ids, attention_mask, labels
  Labels have -100 for prompt tokens (completion-only loss)
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

# Fix CUDA memory fragmentation for large models
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

from datasets import load_dataset as _hf_load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def make_collator(tokenizer):
    """Create a collator that properly pads pre-tokenized data with labels."""
    pad_id = tokenizer.pad_token_id

    def collator(features):
        # Find max length in this batch
        max_len = max(len(f["input_ids"]) for f in features)

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)

        # Convert to tensors
        for key in batch:
            batch[key] = torch.tensor(batch[key], dtype=torch.long)
        return batch

    return collator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# LoRA config (consistent across all seeds)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training defaults
DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
DEFAULT_MAX_LENGTH = 512
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_SEEDS = [42, 1234, 20260726]


def load_raw_dataset(split_dir: Path, split: str) -> "datasets.Dataset":
    """Load JSONL conversational dataset."""
    jsonl_path = split_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    ds = _hf_load_dataset("json", data_files={split: str(jsonl_path)}, split=split)
    print(f"  Loaded {split}: {len(ds)} samples, columns: {ds.column_names}")
    return ds


def tokenize_with_completion_mask(example: dict, tokenizer, max_length: int) -> dict:
    """Tokenize conversational example with completion-only labels."""
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
    prompt_encoding = tokenizer(prompt_text)
    prompt_ids = prompt_encoding["input_ids"]
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


def prepare_dataset(ds, tokenizer, max_length: int, num_proc: int = 1) -> "datasets.Dataset":
    """Pre-tokenize dataset with completion-only labels."""
    print(f"  Tokenizing (max_length={max_length})...")

    def tokenize_fn(example):
        return tokenize_with_completion_mask(example, tokenizer, max_length)

    tokenized = ds.map(
        tokenize_fn,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )
    print(f"  Tokenized: {len(tokenized)} samples")
    print(f"  Columns: {tokenized.column_names}")

    sample = tokenized[0]
    total_tokens = len(sample["input_ids"])
    completion_tokens = sum(1 for l in sample["labels"] if l != -100)
    print(f"  Sample: {total_tokens - completion_tokens} prompt + {completion_tokens} completion tokens")

    return tokenized


def find_latest_checkpoint(seed_dir: Path) -> str | None:
    """Find the latest checkpoint directory in seed_dir."""
    checkpoints = sorted(seed_dir.glob("checkpoint-*"))
    if checkpoints:
        return str(checkpoints[-1])
    return None


def compute_action_metrics(model, tokenizer, raw_dataset, max_length: int) -> dict:
    """Compute teacher-forced action accuracy on eval set."""
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
            if total >= 20:
                break

    model.train()
    return {
        "exact_match": exact_match / total if total > 0 else 0,
        "action_type_match": action_type_match / total if total > 0 else 0,
        "schema_valid": schema_valid / total if total > 0 else 0,
    }


def train_single_seed(
    base_model_path: str,
    train_dataset,
    eval_dataset,
    output_dir: Path,
    seed: int,
    max_length: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    resume_from_checkpoint: str = None,
) -> dict:
    """Train a single seed. Supports checkpoint resumption."""
    print(f"\n{'='*60}")
    print(f"Training seed={seed}")
    print(f"{'='*60}")

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenize datasets with completion-only labels
    print("Pre-tokenizing datasets...")
    train_dataset = prepare_dataset(train_dataset, tokenizer, max_length, num_proc=1)
    eval_dataset_raw = eval_dataset  # Keep raw for action metrics
    eval_dataset = prepare_dataset(eval_dataset, tokenizer, max_length, num_proc=1)

    # Find checkpoint to resume from
    checkpoint = resume_from_checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(seed_dir)
    if checkpoint:
        print(f"  Will resume from: {checkpoint}")
    else:
        print("  Starting fresh (no checkpoint found)")

    # Load model
    print("Loading model...")
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
    print(f"  GPU memory: {torch.cuda.memory_allocated()/(1024**3):.1f} GB")

    # LoRA config
    print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA}, targets={TARGET_MODULES}")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # Apply PEFT
    print("Applying LoRA...", flush=True)
    model = get_peft_model(model, lora_config)
    print("Enabling gradient checkpointing...", flush=True)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    print(f"  GPU after PEFT: {torch.cuda.memory_allocated()/(1024**3):.1f} GB", flush=True)

    # Training arguments
    print("Configuring trainer...")
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
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Max length: {max_length}")
    print(f"  LR: {learning_rate}, epochs: {num_epochs}")
    print(f"  Save every {training_args.save_steps} steps", flush=True)

    # Custom collator for pre-tokenized data
    data_collator = make_collator(tokenizer)

    # Standard Trainer
    print("Creating Trainer...", flush=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    print("Trainer created!", flush=True)

    # Train
    print("Starting training...", flush=True)
    train_start = time.time()
    print(f"  About to call trainer.train() at {train_start:.0f}s", flush=True)
    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    except Exception as e:
        print(f"\nTRAINING FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    train_time = time.time() - train_start
    print(f"  trainer.train() returned after {train_time:.1f}s", flush=True)

    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"  Train loss: {train_result.metrics.get('train_loss', 'N/A')}")

    # Save final adapter
    final_adapter_dir = seed_dir / "final_adapter"
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"  Adapter saved: {final_adapter_dir}")

    # Evaluation
    print("Running evaluation...")
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", float("nan"))
    print(f"  Eval loss: {eval_loss:.4f}")

    # Teacher-forced action accuracy (needs raw dataset with messages)
    action_metrics = compute_action_metrics(
        model, tokenizer, eval_dataset_raw, max_length
    )
    print(f"  Exact match: {action_metrics['exact_match']:.1%}")
    print(f"  Action type match: {action_metrics['action_type_match']:.1%}")
    print(f"  Schema valid: {action_metrics['schema_valid']:.1%}")

    # Save metrics
    all_metrics = {
        "seed": seed,
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
        "effective_batch_size": batch_size * grad_accum,
    }
    metrics_path = seed_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"  Metrics saved: {metrics_path}")

    return all_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M2.2 LoRA SFT Trainer")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=Path("data/sft/m2_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m2_2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    args = parser.parse_args()

    print("=== M2.2 LoRA SFT Trainer ===")
    print(f"Base model: {args.base_model}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Seeds: {args.seeds}")
    print(f"Config: max_length={args.max_length}, lr={args.lr}, epochs={args.epochs}, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum}")

    # Load raw conversational datasets
    print("\nLoading datasets...")
    train_ds = load_raw_dataset(args.data_dir, "train")
    valid_ds = load_raw_dataset(args.data_dir, "valid")

    # Train each seed
    all_metrics = []
    for seed in args.seeds:
        metrics = train_single_seed(
            base_model_path=args.base_model,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            output_dir=args.output_dir,
            seed=seed,
            max_length=args.max_length,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        all_metrics.append(metrics)

    # Save summary
    summary_path = args.output_dir / "training_summary.json"
    summary = {
        "config": {
            "base_model": args.base_model,
            "max_length": args.max_length,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "seeds": args.seeds,
        },
        "seeds": all_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary saved: {summary_path}")

    # Print comparison
    print("\n=== Seed Comparison ===")
    for m in all_metrics:
        print(f"  Seed {m['seed']}: train_loss={m['train_loss']:.4f}, "
              f"eval_loss={m['eval_loss']:.4f}, "
              f"exact_match={m['exact_match']:.1%}, "
              f"time={m['train_time_s']:.0f}s")


if __name__ == "__main__":
    main()
