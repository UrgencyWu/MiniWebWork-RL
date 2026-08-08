"""Dedicated unique-evidence completion-only trainer for the focused study."""

from __future__ import annotations

import gc
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from ..m4_long_horizon_protocol import (
    MAX_SEQUENCE_LENGTH,
    SFT_EFFECTIVE_BATCH_SIZE,
    SFT_GRADIENT_CHECKPOINTING,
    SFT_LEARNING_RATE,
    SFT_LORA_CONFIG,
    SFT_MICROBATCH_CANDIDATES,
    SFT_SEED,
    SFT_STOPPING_RULE,
    SFT_VRAM_HEADROOM_MINIMUM,
)
from ..model_agent import output_parser
from ..sft.m4_long_horizon_dataset import validate_verified_sft_corpus
from .contracts import atomic_write_json, sha256_file

SFT_TRAINER_SCHEMA = "m4_long_horizon_sft_trainer_v1"
SFT_BENCHMARK_SCHEMA = "m4_long_horizon_sft_microbatch_benchmark_v1"
def _chat_ids(value: Any, *, field: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token_id, int) for token_id in value):
        raise ValueError(f"{field} chat template did not return one token-id list")
    return value


@dataclass(frozen=True)
class TokenizedSFTExample:
    sample_id: str
    task_id: str
    task_family: str
    horizon_stratum: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_content: str
    prompt_tokens: int
    completion_label_tokens: int
    untruncated_forward_tokens: int

    @property
    def forward_tokens(self) -> int:
        return len(self.input_ids)


def tokenize_sft_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> TokenizedSFTExample:
    if max_length != MAX_SEQUENCE_LENGTH:
        raise ValueError(f"focused SFT max length is frozen at {MAX_SEQUENCE_LENGTH}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT record must end in one assistant message")
    chat_kwargs = dict(record.get("chat_template_kwargs") or {})
    prompt_ids = _chat_ids(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            **chat_kwargs,
        ),
        field="prompt",
    )
    full_ids = _chat_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            **chat_kwargs,
        ),
        field="full",
    )
    if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant completion is not a strict chat-template continuation")
    truncated = full_ids[:max_length]
    prompt_boundary = min(len(prompt_ids), len(truncated))
    labels = [-100] * prompt_boundary + truncated[prompt_boundary:]
    completion_count = sum(label != -100 for label in labels)
    if completion_count <= 0:
        raise ValueError(f"zero-label SFT record: {record.get('sample_id')}")
    if len(truncated) != len(labels):
        raise RuntimeError("SFT token/label length mismatch")
    return TokenizedSFTExample(
        sample_id=str(record.get("sample_id") or ""),
        task_id=str(record.get("task_id") or ""),
        task_family=str(record.get("task_family") or "unknown"),
        horizon_stratum=str(record.get("horizon_stratum") or "unknown"),
        input_ids=tuple(int(token_id) for token_id in truncated),
        labels=tuple(int(label) for label in labels),
        assistant_content=str(messages[-1].get("content") or ""),
        prompt_tokens=len(prompt_ids),
        completion_label_tokens=completion_count,
        untruncated_forward_tokens=len(full_ids),
    )


def load_sft_examples(
    data_dir: Path,
    split: str,
    tokenizer: Any,
    *,
    limit: int | None = None,
) -> tuple[TokenizedSFTExample, ...]:
    root = Path(data_dir).expanduser().resolve()
    validation = validate_verified_sft_corpus(root, require_full_roster=True)
    if not validation["valid"]:
        raise ValueError(f"invalid focused SFT corpus: {validation['errors']}")
    filename = {"train": "train.jsonl", "dev": "valid.jsonl"}.get(split)
    if filename is None:
        raise ValueError("SFT split must be train or dev")
    records = [
        json.loads(line)
        for line in (root / filename).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("SFT example limit must be positive")
        records = records[:limit]
    examples = tuple(tokenize_sft_record(record, tokenizer) for record in records)
    sample_ids = [example.sample_id for example in examples]
    if not examples or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("SFT examples must be non-empty and unique")
    return examples


def validate_sft_training_inputs(data_dir: Path, base_model: Path) -> dict[str, Any]:
    """Bind training to the audited corpus and exact local tokenizer files."""

    root = Path(data_dir).expanduser().resolve()
    model_root = Path(base_model).expanduser().resolve()
    validation = validate_verified_sft_corpus(root, require_full_roster=True)
    if not validation["valid"]:
        raise ValueError(f"invalid focused SFT corpus: {validation['errors']}")
    required = {
        "manifest": root / "manifest.json",
        "token_audit": root / "token_audit.json",
        "train": root / "train.jsonl",
        "dev": root / "valid.jsonl",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"focused SFT inputs are missing: {missing}")
    audit = json.loads(required["token_audit"].read_text(encoding="utf-8"))
    if audit.get("schema_version") != "m4_long_horizon_sft_token_audit_v1" or audit.get("passed") is not True:
        raise ValueError("focused SFT token audit is not a passing v1 artifact")
    if Path(audit.get("base_model", "")).expanduser().resolve() != model_root:
        raise ValueError("focused SFT token audit base-model binding drift")
    expected_hashes = {
        "corpus_manifest_sha256": sha256_file(required["manifest"]),
        "train_sha256": sha256_file(required["train"]),
        "valid_sha256": sha256_file(required["dev"]),
    }
    for field, actual in expected_hashes.items():
        if audit.get(field) != actual:
            raise ValueError(f"focused SFT token audit {field} drift")
    if audit.get("max_length") != MAX_SEQUENCE_LENGTH or audit.get("repetition_policy") != "none":
        raise ValueError("focused SFT tokenization or repetition contract drift")
    for split in ("train", "dev"):
        split_audit = audit.get("splits", {}).get(split, {})
        if (
            split_audit.get("sample_count") != split_audit.get("unique_sample_count")
            or split_audit.get("duplicate_sample_count") != 0
            or split_audit.get("zero_completion_label_sample_count") != 0
            or split_audit.get("zero_completion_label_sample_fraction") != 0.0
            or split_audit.get("truncated_sample_count") != 0
            or split_audit.get("maximum_forward_sequence_tokens", MAX_SEQUENCE_LENGTH + 1)
            > MAX_SEQUENCE_LENGTH
        ):
            raise ValueError(f"focused SFT {split} audit violates uniqueness/label/length gates")
    tokenizer_hashes = audit.get("tokenizer_file_sha256", {})
    if not tokenizer_hashes:
        raise ValueError("focused SFT token audit has no tokenizer binding")
    for filename, expected in tokenizer_hashes.items():
        candidate = model_root / filename
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"focused SFT tokenizer binding drift: {filename}")
    return {
        "data_dir": str(root),
        "base_model": str(model_root),
        "manifest_sha256": expected_hashes["corpus_manifest_sha256"],
        "token_audit_sha256": sha256_file(required["token_audit"]),
        "train_sha256": expected_hashes["train_sha256"],
        "dev_sha256": expected_hashes["valid_sha256"],
        "tokenizer_file_sha256": tokenizer_hashes,
        "splits": audit["splits"],
        "valid": True,
    }


class SFTExampleDataset(Dataset):
    def __init__(self, examples: Sequence[TokenizedSFTExample]):
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenizedSFTExample:
        return self.examples[index]


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Yield every unique example exactly once while limiting padding waste."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        seed: int,
        bucket_size: int = 128,
    ):
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("bucket sampler requires positive lengths")
        if batch_size <= 0 or bucket_size < batch_size:
            raise ValueError("invalid bucket sampler size")
        self.lengths = tuple(int(length) for length in lengths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_size = int(bucket_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        ordered = sorted(range(len(self.lengths)), key=lambda index: (self.lengths[index], index))
        buckets = [ordered[start : start + self.bucket_size] for start in range(0, len(ordered), self.bucket_size)]
        rng.shuffle(buckets)
        batches: list[list[int]] = []
        for bucket in buckets:
            rng.shuffle(bucket)
            batches.extend(
                bucket[start : start + self.batch_size]
                for start in range(0, len(bucket), self.batch_size)
            )
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        if not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("collator requires a non-negative pad token id")
        self.pad_token_id = pad_token_id

    def __call__(self, examples: Sequence[TokenizedSFTExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty SFT batch")
        maximum = max(example.forward_tokens for example in examples)
        maximum_completion = max(example.completion_label_tokens for example in examples)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for example in examples:
            padding = maximum - example.forward_tokens
            input_ids.append([self.pad_token_id] * padding + list(example.input_ids))
            labels.append([-100] * padding + list(example.labels))
            attention_mask.append([0] * padding + [1] * example.forward_tokens)
        attention = torch.tensor(attention_mask, dtype=torch.long)
        positions = (attention.cumsum(dim=-1) - 1).clamp_min(0)
        logits_to_keep = min(maximum, maximum_completion + 1)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": attention,
            "position_ids": positions,
            "logits_to_keep": logits_to_keep,
            "examples": tuple(examples),
        }


@dataclass
class CompletionLoss:
    mean: torch.Tensor
    total: torch.Tensor
    token_count: int
    shifted_labels: torch.Tensor


def completion_only_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> CompletionLoss:
    """Compute exact causal loss over the completion labels represented by logits."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError("invalid completion loss tensor shapes")
    if logits.shape[1] < 2 or logits.shape[1] > labels.shape[1]:
        raise ValueError("completion logits cannot align to labels")
    tail_labels = labels[:, -logits.shape[1] :]
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = tail_labels[:, 1:].contiguous()
    token_count = int((shifted_labels != -100).sum().item())
    if token_count <= 0:
        raise ValueError("batch contains no effective completion label")
    total = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return CompletionLoss(
        mean=total / token_count,
        total=total,
        token_count=token_count,
        shifted_labels=shifted_labels,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **dict(batch),
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "position_ids": batch["position_ids"].to(device, non_blocking=True),
    }


def _forward(model: Any, batch: Mapping[str, Any]):
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
        logits_to_keep=int(batch["logits_to_keep"]),
    )


def _target_action(content: str) -> dict[str, Any] | None:
    parsed = output_parser.parse(content)
    return parsed.parsed_payload if parsed.schema_valid else None


def evaluate_sft_model(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    was_training = bool(model.training)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    exact = 0
    schema_valid = 0
    sample_count = 0
    with torch.inference_mode():
        for raw_batch in batches:
            batch = _move_batch(raw_batch, device)
            output = _forward(model, batch)
            loss = completion_only_cross_entropy(output.logits, batch["labels"])
            total_loss += float(loss.total.detach().cpu())
            total_tokens += loss.token_count
            predictions = output.logits[:, :-1, :].argmax(dim=-1)
            for row_index, example in enumerate(batch["examples"]):
                mask = loss.shifted_labels[row_index] != -100
                predicted_ids = predictions[row_index][mask].detach().cpu().tolist()
                predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True).strip()
                parsed = output_parser.parse(predicted_text)
                schema_valid += int(parsed.schema_valid)
                exact += int(parsed.schema_valid and parsed.parsed_payload == _target_action(example.assistant_content))
                sample_count += 1
    if was_training:
        model.train()
    if total_tokens <= 0 or sample_count <= 0:
        raise ValueError("SFT evaluation produced no labels or samples")
    return {
        "dev_nll": total_loss / total_tokens,
        "teacher_forced_action_exact": exact / sample_count,
        "teacher_forced_schema_valid": schema_valid / sample_count,
        "sample_count": sample_count,
        "completion_label_tokens": total_tokens,
    }


@dataclass
class PlateauController:
    best_nll: float = math.inf
    best_action_exact: float = -math.inf
    best_schema_valid: float = -math.inf
    stale_evaluations: int = 0

    def update(self, metrics: Mapping[str, float], *, completed_epochs: int) -> dict[str, Any]:
        nll = float(metrics["dev_nll"])
        action = float(metrics["teacher_forced_action_exact"])
        schema = float(metrics["teacher_forced_schema_valid"])
        improvements = {
            "dev_nll": self.best_nll - nll >= SFT_STOPPING_RULE["dev_nll_minimum_improvement"],
            "teacher_forced_action_exact": action - self.best_action_exact >= SFT_STOPPING_RULE["teacher_forced_action_exact_minimum_improvement"],
            "teacher_forced_schema_valid": schema - self.best_schema_valid >= SFT_STOPPING_RULE["teacher_forced_schema_valid_minimum_improvement"],
        }
        self.best_nll = min(self.best_nll, nll)
        self.best_action_exact = max(self.best_action_exact, action)
        self.best_schema_valid = max(self.best_schema_valid, schema)
        self.stale_evaluations = 0 if any(improvements.values()) else self.stale_evaluations + 1
        stop = (
            completed_epochs >= SFT_STOPPING_RULE["minimum_epochs"]
            and self.stale_evaluations >= SFT_STOPPING_RULE["plateau_patience_evaluations"]
        )
        return {"improvements": improvements, "stale_evaluations": self.stale_evaluations, "stop": stop}


def build_dataloader(
    examples: Sequence[TokenizedSFTExample],
    *,
    microbatch_size: int,
    seed: int,
    epoch: int,
    pad_token_id: int,
    workers: int,
) -> DataLoader:
    sampler = LengthBucketBatchSampler(
        [example.forward_tokens for example in examples],
        batch_size=microbatch_size,
        seed=seed,
    )
    sampler.set_epoch(epoch)
    kwargs: dict[str, Any] = {
        "dataset": SFTExampleDataset(examples),
        "batch_sampler": sampler,
        "collate_fn": CompletionOnlyCollator(pad_token_id),
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def load_trainable_lora_model(
    base_model: Path | str,
    *,
    auto_model_class: Any | None = None,
    lora_config_class: Any | None = None,
    peft_model_factory: Any | None = None,
):
    if auto_model_class is None or lora_config_class is None or peft_model_factory is None:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        auto_model_class = auto_model_class or AutoModelForCausalLM
        lora_config_class = lora_config_class or LoraConfig
        peft_model_factory = peft_model_factory or get_peft_model

    model = auto_model_class.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": SFT_GRADIENT_CHECKPOINTING["use_reentrant"],
        }
    )
    config = lora_config_class(
        r=SFT_LORA_CONFIG["r"],
        lora_alpha=SFT_LORA_CONFIG["alpha"],
        lora_dropout=SFT_LORA_CONFIG["dropout"],
        target_modules=SFT_LORA_CONFIG["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft_model_factory(model, config)
    model.train()
    return model


def trainable_parameters(model: Any) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("SFT model has no trainable LoRA parameters")
    return parameters


def _save_adapter(model: Any, tokenizer: Any, destination: Path) -> None:
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite adapter: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{time.time_ns()}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    temporary.replace(destination)


def summarize_sft_examples(examples: Sequence[TokenizedSFTExample]) -> dict[str, Any]:
    return {
        "sample_count": len(examples),
        "unique_sample_count": len({example.sample_id for example in examples}),
        "completion_label_tokens": sum(example.completion_label_tokens for example in examples),
        "forward_tokens": sum(example.forward_tokens for example in examples),
        "maximum_forward_tokens": max(example.forward_tokens for example in examples),
        "zero_label_count": sum(example.completion_label_tokens == 0 for example in examples),
        "truncated_count": sum(example.untruncated_forward_tokens > MAX_SEQUENCE_LENGTH for example in examples),
    }


def train_sft(
    *,
    model: Any,
    tokenizer: Any,
    train_examples: Sequence[TokenizedSFTExample],
    dev_examples: Sequence[TokenizedSFTExample],
    output_dir: Path,
    microbatch_size: int,
    workers: int,
    maximum_optimizer_updates: int | None = None,
    maximum_epochs: int | None = None,
) -> dict[str, Any]:
    if microbatch_size not in SFT_MICROBATCH_CANDIDATES:
        raise ValueError("microbatch size is outside the frozen candidates")
    if SFT_EFFECTIVE_BATCH_SIZE % microbatch_size:
        raise ValueError("microbatch must divide the frozen effective batch")
    epochs = maximum_epochs or SFT_STOPPING_RULE["maximum_epochs"]
    if not 1 <= epochs <= SFT_STOPPING_RULE["maximum_epochs"]:
        raise ValueError("requested SFT epochs exceed the frozen cap")
    if maximum_optimizer_updates is not None and maximum_optimizer_updates <= 0:
        raise ValueError("maximum optimizer updates must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"SFT output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SFT_SEED)
    torch.manual_seed(SFT_SEED)
    torch.cuda.manual_seed_all(SFT_SEED)
    device = torch.device("cuda:0")
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=SFT_LEARNING_RATE, weight_decay=0.0)
    grad_accumulation = SFT_EFFECTIVE_BATCH_SIZE // microbatch_size
    plateau = PlateauController()
    history: list[dict[str, Any]] = []
    optimizer_updates = 0
    started = time.monotonic()
    stop_reason = "maximum_epochs"
    best_epoch = 0
    best_nll = math.inf
    model.train()

    for epoch in range(epochs):
        loader = build_dataloader(
            train_examples,
            microbatch_size=microbatch_size,
            seed=SFT_SEED,
            epoch=epoch,
            pad_token_id=tokenizer.pad_token_id,
            workers=workers,
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated_tokens = 0
        accumulated_microbatches = 0
        epoch_tokens = 0
        epoch_loss_sum = 0.0
        seen_sample_ids: list[str] = []
        reached_update_cap = False
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            output = _forward(model, batch)
            loss = completion_only_cross_entropy(output.logits, batch["labels"])
            loss.total.backward()
            accumulated_tokens += loss.token_count
            accumulated_microbatches += 1
            epoch_tokens += loss.token_count
            epoch_loss_sum += float(loss.total.detach().cpu())
            seen_sample_ids.extend(example.sample_id for example in batch["examples"])
            if accumulated_microbatches == grad_accumulation:
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_tokens)
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("SFT gradient norm is not finite")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                accumulated_tokens = 0
                accumulated_microbatches = 0
                if maximum_optimizer_updates is not None and optimizer_updates >= maximum_optimizer_updates:
                    reached_update_cap = True
                    break
        if accumulated_microbatches and not reached_update_cap:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_tokens)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
        if len(seen_sample_ids) != len(set(seen_sample_ids)):
            raise RuntimeError("SFT epoch repeated a source example")
        if not reached_update_cap and len(seen_sample_ids) != len(train_examples):
            raise RuntimeError("SFT epoch did not consume the full unique train roster")

        dev_loader = build_dataloader(
            dev_examples,
            microbatch_size=microbatch_size,
            seed=SFT_SEED,
            epoch=0,
            pad_token_id=tokenizer.pad_token_id,
            workers=workers,
        )
        metrics = evaluate_sft_model(model, dev_loader, tokenizer, device=device)
        plateau_state = plateau.update(metrics, completed_epochs=epoch + 1)
        checkpoint = output_dir / f"checkpoint-epoch-{epoch + 1}"
        _save_adapter(model, tokenizer, checkpoint)
        epoch_report = {
            "epoch": epoch + 1,
            "optimizer_updates": optimizer_updates,
            "train_seen_unique_samples": len(set(seen_sample_ids)),
            "train_completion_label_tokens": epoch_tokens,
            "train_nll": epoch_loss_sum / epoch_tokens,
            "dev": metrics,
            "plateau": plateau_state,
            "checkpoint": str(checkpoint),
        }
        history.append(epoch_report)
        atomic_write_json(output_dir / "training_progress.json", {"history": history, "complete": False})
        if float(metrics["dev_nll"]) < best_nll:
            best_nll = float(metrics["dev_nll"])
            best_epoch = epoch + 1
        if reached_update_cap:
            stop_reason = "preflight_optimizer_update_cap"
            break
        if plateau_state["stop"]:
            stop_reason = "dev_plateau"
            break

    if not history or best_epoch <= 0:
        raise RuntimeError("SFT training produced no evaluated checkpoint")
    source = output_dir / f"checkpoint-epoch-{best_epoch}"
    final_adapter = output_dir / "final_adapter"
    shutil.copytree(source, final_adapter)
    report = {
        "schema_version": SFT_TRAINER_SCHEMA,
        "seed": SFT_SEED,
        "lora": SFT_LORA_CONFIG,
        "gradient_checkpointing": SFT_GRADIENT_CHECKPOINTING,
        "learning_rate": SFT_LEARNING_RATE,
        "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
        "microbatch_size": microbatch_size,
        "effective_batch_size": SFT_EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation": grad_accumulation,
        "workers": workers,
        "train_audit": summarize_sft_examples(train_examples),
        "dev_audit": summarize_sft_examples(dev_examples),
        "optimizer_updates": optimizer_updates,
        "history": history,
        "best_epoch": best_epoch,
        "best_dev_nll": best_nll,
        "stop_reason": stop_reason,
        "elapsed_s": time.monotonic() - started,
        "final_adapter": str(final_adapter),
        "complete": True,
    }
    atomic_write_json(output_dir / "training_report.json", report)
    return report


def select_sft_microbatch(results: Sequence[Mapping[str, Any]]) -> int:
    """Apply the frozen fail-closed VRAM gate to benchmark results."""

    expected = list(SFT_MICROBATCH_CANDIDATES)
    observed = [result.get("microbatch_size") for result in results]
    if observed != expected:
        raise ValueError(f"SFT benchmark candidates must be exactly {expected}")
    eligible = [
        result
        for result in results
        if result.get("passed") is True
        and isinstance(result.get("vram_headroom_fraction"), (int, float))
        and float(result["vram_headroom_fraction"]) >= SFT_VRAM_HEADROOM_MINIMUM
        and result.get("headroom_gate_passed") is True
    ]
    if not eligible:
        raise RuntimeError("no SFT microbatch candidate retained 15% VRAM headroom")
    return int(max(eligible, key=lambda result: int(result["microbatch_size"]))["microbatch_size"])


def benchmark_microbatches(
    *,
    model: Any,
    examples: Sequence[TokenizedSFTExample],
    tokenizer: Any,
    output: Path,
    corpus_manifest_sha256: str,
    token_audit_sha256: str,
    warmup_microsteps: int = 2,
    measured_microsteps: int = 10,
) -> dict[str, Any]:
    if warmup_microsteps < 0 or not 10 <= measured_microsteps <= 20:
        raise ValueError("SFT benchmark requires 10-20 measured microsteps")
    if not torch.cuda.is_available():
        raise RuntimeError("SFT microbatch benchmark requires CUDA")
    device = torch.device("cuda:0")
    parameters = trainable_parameters(model)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    longest = sorted(examples, key=lambda example: example.forward_tokens, reverse=True)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    results: list[dict[str, Any]] = []
    stop_after_oom = False

    def write_report(
        *,
        complete: bool,
        passed: bool,
        selected: int | None = None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        report = {
            "schema_version": SFT_BENCHMARK_SCHEMA,
            "seed": SFT_SEED,
            "lora": SFT_LORA_CONFIG,
            "gradient_checkpointing": SFT_GRADIENT_CHECKPOINTING,
            "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
            "effective_batch_size": SFT_EFFECTIVE_BATCH_SIZE,
            "candidate_microbatches": list(SFT_MICROBATCH_CANDIDATES),
            "selection_rule": "largest passed candidate with at least 0.15 reserved-VRAM headroom",
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "token_audit_sha256": token_audit_sha256,
            "gpu_name": torch.cuda.get_device_properties(device).name,
            "results": results,
            "selected_microbatch_size": selected,
            "complete": complete,
            "passed": passed,
        }
        if failure_reason is not None:
            report["failure_reason"] = failure_reason
        atomic_write_json(output, report)
        return report

    for candidate in SFT_MICROBATCH_CANDIDATES:
        if stop_after_oom:
            results.append({"microbatch_size": candidate, "passed": False, "reason": "larger_than_oom_candidate"})
            continue
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(initial[name].to(parameter.device, dtype=parameter.dtype))
        optimizer = torch.optim.AdamW(parameters, lr=SFT_LEARNING_RATE, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        grad_accumulation = SFT_EFFECTIVE_BATCH_SIZE // candidate
        collator = CompletionOnlyCollator(tokenizer.pad_token_id)
        batch = collator(longest[:candidate])
        total_microsteps = max(warmup_microsteps + measured_microsteps, grad_accumulation)
        measured_start_index = total_microsteps - measured_microsteps
        measured_forward_tokens = 0
        measured_completion_tokens = 0
        measured_started = None
        accumulated_tokens = 0
        accumulated_microbatches = 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for microstep in range(total_microsteps):
                if microstep == measured_start_index:
                    torch.cuda.synchronize(device)
                    measured_started = time.monotonic()
                moved = _move_batch(batch, device)
                output = _forward(model, moved)
                loss = completion_only_cross_entropy(output.logits, moved["labels"])
                loss.total.backward()
                accumulated_tokens += loss.token_count
                accumulated_microbatches += 1
                if microstep >= measured_start_index:
                    measured_forward_tokens += int(moved["attention_mask"].sum().item())
                    measured_completion_tokens += loss.token_count
                if accumulated_microbatches == grad_accumulation:
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_tokens)
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_tokens = 0
                    accumulated_microbatches = 0
            if accumulated_microbatches:
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_tokens)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - float(measured_started)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            headroom = 1.0 - peak_reserved / total_memory
            results.append({
                "microbatch_size": candidate,
                "gradient_accumulation": grad_accumulation,
                "warmup_microsteps": total_microsteps - measured_microsteps,
                "measured_microsteps": measured_microsteps,
                "maximum_batch_sequence_tokens": max(example.forward_tokens for example in longest[:candidate]),
                "measured_forward_tokens": measured_forward_tokens,
                "measured_completion_label_tokens": measured_completion_tokens,
                "elapsed_s": elapsed,
                "forward_tokens_per_second": measured_forward_tokens / elapsed,
                "completion_tokens_per_second": measured_completion_tokens / elapsed,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "total_device_memory_bytes": total_memory,
                "vram_headroom_fraction": headroom,
                "headroom_gate_passed": headroom >= SFT_VRAM_HEADROOM_MINIMUM,
                "passed": True,
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise
            results.append({
                "microbatch_size": candidate,
                "gradient_accumulation": grad_accumulation,
                "passed": False,
                "reason": "cuda_out_of_memory",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            })
            stop_after_oom = True
        finally:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
        write_report(complete=False, passed=False)

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(initial[name].to(parameter.device, dtype=parameter.dtype))

    try:
        selected = select_sft_microbatch(results)
    except RuntimeError:
        write_report(
            complete=True,
            passed=False,
            failure_reason="no_candidate_retained_frozen_vram_headroom",
        )
        raise
    return write_report(complete=True, passed=True, selected=selected)
