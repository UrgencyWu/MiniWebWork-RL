"""Audited completion-only SFT runtime for the focused M5 WebShop study.

This module deliberately keeps M5 data validation, tokenization and recovery
separate from the older M4 trainer.  A recovery checkpoint is committed only
at an optimizer boundary, so a 24-hour Slurm successor can safely reopen the
same run root without accepting a partially accumulated gradient.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..long_horizon_rl.contracts import atomic_write_json, sha256_file, sha256_json
from ..long_horizon_rl.sft_trainer import (
    CompletionOnlyCollator,
    LengthBucketBatchSampler,
    TokenizedSFTExample,
    _forward,
    _move_batch,
    completion_only_cross_entropy,
)
from ..m5_webshop_protocol import load_protocol
from .actions import parse_command_output

TRAINER_SCHEMA = "m5_webshop_sft_trainer_v1"
BENCHMARK_SCHEMA = "m5_webshop_sft_microbatch_benchmark_v1"
RECOVERY_SCHEMA = "m5_webshop_sft_recovery_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any], *, label: str) -> None:
    expected = dict(payload)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), f"{label} self-hash drift")


def _chat_ids(value: Any, *, field: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    _require(
        isinstance(value, list) and all(isinstance(token_id, int) for token_id in value),
        f"{field} chat template did not return one token-id list",
    )
    return value


@dataclass(frozen=True)
class SFTConfig:
    seed: int
    maximum_sequence_tokens: int
    learning_rate: float
    effective_batch_size: int
    microbatch_candidates: tuple[int, ...]
    minimum_vram_headroom: float
    maximum_epochs: int
    minimum_epochs: int
    checkpoint_interval_updates: int
    lora: dict[str, Any]
    chat_template_kwargs: dict[str, Any]

    @classmethod
    def from_protocol(cls, payload: Mapping[str, Any]) -> "SFTConfig":
        sft = payload["sft"]
        return cls(
            seed=int(sft["selection_seed"]),
            maximum_sequence_tokens=int(sft["maximum_sequence_tokens"]),
            learning_rate=float(sft["learning_rate"]),
            effective_batch_size=int(sft["effective_batch_size"]),
            microbatch_candidates=tuple(int(value) for value in sft["microbatch_candidates"]),
            minimum_vram_headroom=float(sft["minimum_vram_headroom"]),
            maximum_epochs=int(sft["maximum_epochs"]),
            minimum_epochs=int(sft["minimum_epochs"]),
            checkpoint_interval_updates=25,
            lora=dict(sft["lora"]),
            chat_template_kwargs=dict(sft["chat_template_kwargs"]),
        )


def validate_sft_training_inputs(data_dir: Path, base_model: Path) -> dict[str, Any]:
    """Validate all corpus, tokenizer, protocol and Git bindings before CUDA."""

    protocol = load_protocol()
    root = Path(data_dir).expanduser().resolve()
    model_root = Path(base_model).expanduser().resolve()
    paths = {
        "corpus_audit": root / "corpus_audit.json",
        "token_audit": root / "token_audit.json",
        "train": root / "train.jsonl",
        "dev": root / "dev.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    _require(not missing, f"M5 SFT inputs are missing: {missing}")
    corpus = json.loads(paths["corpus_audit"].read_text(encoding="utf-8"))
    token = json.loads(paths["token_audit"].read_text(encoding="utf-8"))
    _self_hash(corpus, label="M5 SFT corpus audit")
    _self_hash(token, label="M5 SFT token audit")
    for artifact, label in ((corpus, "corpus"), (token, "token")):
        _require(artifact.get("passed") is True, f"M5 SFT {label} audit did not pass")
        _require(artifact.get("git_sha") == protocol["git_sha"], f"M5 SFT {label} Git drift")
        _require(
            artifact.get("protocol_sha256") == protocol["sha256"],
            f"M5 SFT {label} protocol drift",
        )
    _require(token.get("corpus_audit_sha256") == sha256_file(paths["corpus_audit"]), "M5 token/corpus binding drift")
    for split in ("train", "dev"):
        _require(token.get(f"{split}_sha256") == sha256_file(paths[split]), f"M5 {split} hash drift")
    config = SFTConfig.from_protocol(protocol["payload"])
    _require(token.get("maximum_sequence_tokens") == config.maximum_sequence_tokens, "M5 token length drift")
    _require(token.get("chat_template_kwargs") == config.chat_template_kwargs, "M5 chat template drift")
    for split in ("train", "dev"):
        audit = token.get("splits", {}).get(split, {})
        _require(audit.get("sample_count") == audit.get("unique_sample_count"), f"M5 {split} duplicates")
        _require(audit.get("zero_completion_label_sample_count") == 0, f"M5 {split} zero labels")
        _require(audit.get("truncated_sample_count") == 0, f"M5 {split} truncation")
        _require(audit.get("maximum_forward_sequence_tokens", config.maximum_sequence_tokens + 1) <= config.maximum_sequence_tokens, f"M5 {split} length overflow")
    tokenizer_hashes = token.get("tokenizer_file_sha256")
    _require(isinstance(tokenizer_hashes, dict) and tokenizer_hashes, "M5 tokenizer binding missing")
    for filename, expected in tokenizer_hashes.items():
        candidate = model_root / filename
        _require(candidate.is_file() and sha256_file(candidate) == expected, f"M5 tokenizer drift: {filename}")
    return {
        "valid": True,
        "git_sha": protocol["git_sha"],
        "protocol_sha256": protocol["sha256"],
        "corpus_audit_sha256": sha256_file(paths["corpus_audit"]),
        "token_audit_sha256": sha256_file(paths["token_audit"]),
        "train_sha256": sha256_file(paths["train"]),
        "dev_sha256": sha256_file(paths["dev"]),
        "base_model": str(model_root),
        "splits": token["splits"],
        "config": asdict(config),
    }


def tokenize_sft_row(row: Mapping[str, Any], tokenizer: Any, config: SFTConfig) -> TokenizedSFTExample:
    prompt_messages = row.get("messages")
    _require(isinstance(prompt_messages, list) and prompt_messages, "M5 SFT prompt messages missing")
    completion = row.get("completion")
    parsed = parse_command_output(str(completion or ""))
    _require(parsed.strict_json_success and parsed.schema_valid, "M5 SFT completion is not strict command JSON")
    full_messages = list(prompt_messages) + [{"role": "assistant", "content": completion}]
    prompt_ids = _chat_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            **config.chat_template_kwargs,
        ),
        field="prompt",
    )
    full_ids = _chat_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            **config.chat_template_kwargs,
        ),
        field="full",
    )
    _require(len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids, "M5 completion is not a chat-template continuation")
    _require(len(full_ids) <= config.maximum_sequence_tokens, "M5 audited SFT row would truncate")
    labels = tuple([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :])
    completion_tokens = sum(label != -100 for label in labels)
    _require(completion_tokens > 0, "M5 SFT row has no completion labels")
    task_id = str(row.get("task_id") or "")
    turn_index = int(row.get("turn_index", -1))
    return TokenizedSFTExample(
        sample_id=f"{task_id}:{turn_index:03d}",
        task_id=task_id,
        task_family="webshop",
        horizon_stratum="official",
        input_ids=tuple(int(value) for value in full_ids),
        labels=labels,
        assistant_content=str(completion),
        prompt_tokens=len(prompt_ids),
        completion_label_tokens=completion_tokens,
        untruncated_forward_tokens=len(full_ids),
    )


def load_sft_examples(data_dir: Path, split: str, tokenizer: Any, config: SFTConfig) -> tuple[TokenizedSFTExample, ...]:
    _require(split in {"train", "dev"}, "M5 SFT split must be train or dev")
    path = Path(data_dir).expanduser().resolve() / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = tuple(tokenize_sft_row(row, tokenizer, config) for row in rows)
    sample_ids = [example.sample_id for example in examples]
    _require(bool(examples) and len(sample_ids) == len(set(sample_ids)), "M5 SFT examples are empty or duplicated")
    return examples


def summarize_examples(examples: Sequence[TokenizedSFTExample], config: SFTConfig) -> dict[str, Any]:
    return {
        "sample_count": len(examples),
        "unique_sample_count": len({example.sample_id for example in examples}),
        "task_count": len({example.task_id for example in examples}),
        "completion_label_tokens": sum(example.completion_label_tokens for example in examples),
        "forward_tokens": sum(example.forward_tokens for example in examples),
        "maximum_forward_tokens": max(example.forward_tokens for example in examples),
        "zero_label_count": sum(example.completion_label_tokens == 0 for example in examples),
        "truncated_count": sum(example.untruncated_forward_tokens > config.maximum_sequence_tokens for example in examples),
    }


def load_trainable_lora_model(base_model: Path, config: SFTConfig, *, resume_adapter: Path | None = None) -> Any:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(Path(base_model).expanduser().resolve()),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if resume_adapter is not None:
        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
    else:
        lora = config.lora
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.train()
    _require(any(parameter.requires_grad for parameter in model.parameters()), "M5 SFT has no trainable LoRA parameters")
    return model


def _trainable_parameters(model: Any) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(bool(parameters), "M5 SFT has no trainable parameters")
    return parameters


def _replace_adapter(model: Any, tokenizer: Any, destination: Path) -> None:
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{time.time_ns()}")
    backup = destination.with_name(f".{destination.name}.old-{time.time_ns()}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    if destination.exists():
        destination.replace(backup)
    temporary.replace(destination)
    if backup.exists():
        shutil.rmtree(backup)


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{time.time_ns()}")
    torch.save(payload, temporary)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(destination)


def save_recovery(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    epoch: int,
    completed_microbatches: int,
    optimizer_updates: int,
    cumulative_completion_label_tokens: int,
    cumulative_loss_sum: float,
    input_binding: Mapping[str, Any],
) -> None:
    root = Path(output_dir) / "recovery"
    root.mkdir(parents=True, exist_ok=True)
    _replace_adapter(model, tokenizer, root / "adapter")
    _atomic_torch_save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        },
        root / "optimizer_rng.pt",
    )
    state = {
        "schema_version": RECOVERY_SCHEMA,
        "git_sha": input_binding["git_sha"],
        "protocol_sha256": input_binding["protocol_sha256"],
        "token_audit_sha256": input_binding["token_audit_sha256"],
        "epoch": epoch,
        "completed_microbatches": completed_microbatches,
        "optimizer_updates": optimizer_updates,
        "cumulative_completion_label_tokens": cumulative_completion_label_tokens,
        "cumulative_loss_sum": cumulative_loss_sum,
        "optimizer_boundary": True,
    }
    state["content_sha256"] = sha256_json(state)
    atomic_write_json(root / "state.json", state)


def load_recovery(output_dir: Path, input_binding: Mapping[str, Any]) -> dict[str, Any] | None:
    root = Path(output_dir) / "recovery"
    state_path = root / "state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _self_hash(state, label="M5 SFT recovery")
    for field in ("git_sha", "protocol_sha256", "token_audit_sha256"):
        _require(state.get(field) == input_binding[field], f"M5 SFT recovery {field} drift")
    _require(state.get("optimizer_boundary") is True, "M5 SFT recovery is not at an optimizer boundary")
    _require((root / "adapter").is_dir() and (root / "optimizer_rng.pt").is_file(), "M5 SFT recovery files missing")
    return {**state, "adapter": root / "adapter", "optimizer_rng": root / "optimizer_rng.pt"}


def select_microbatch(results: Sequence[Mapping[str, Any]], config: SFTConfig) -> int:
    _require([row.get("microbatch_size") for row in results] == list(config.microbatch_candidates), "M5 benchmark candidate drift")
    eligible = [
        row
        for row in results
        if row.get("passed") is True
        and float(row.get("vram_headroom_fraction", -1.0)) >= config.minimum_vram_headroom
    ]
    _require(bool(eligible), "no M5 SFT microbatch retained the frozen VRAM headroom")
    return max(int(row["microbatch_size"]) for row in eligible)


def benchmark_microbatches(
    *,
    model: Any,
    examples: Sequence[TokenizedSFTExample],
    tokenizer: Any,
    config: SFTConfig,
    output_path: Path,
    input_binding: Mapping[str, Any],
    measured_microsteps: int = 10,
) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "M5 SFT benchmark requires CUDA")
    device = torch.device("cuda:0")
    parameters = _trainable_parameters(model)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    longest = sorted(examples, key=lambda example: example.forward_tokens, reverse=True)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    results: list[dict[str, Any]] = []
    stop = False
    for candidate in config.microbatch_candidates:
        if stop:
            results.append({"microbatch_size": candidate, "passed": False, "reason": "larger_than_oom_candidate"})
            continue
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(initial[name].to(parameter.device, dtype=parameter.dtype))
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        collator = CompletionOnlyCollator(tokenizer.pad_token_id)
        batch = collator(longest[:candidate])
        gradient_accumulation = config.effective_batch_size // candidate
        total_microsteps = max(measured_microsteps + 2, gradient_accumulation)
        measured_start = total_microsteps - measured_microsteps
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = None
        forward_tokens = completion_tokens = 0
        accumulated_tokens = accumulated_microbatches = 0
        try:
            for microstep in range(total_microsteps):
                if microstep == measured_start:
                    torch.cuda.synchronize(device)
                    started = time.monotonic()
                moved = _move_batch(batch, device)
                output = _forward(model, moved)
                loss = completion_only_cross_entropy(output.logits, moved["labels"])
                loss.total.backward()
                accumulated_tokens += loss.token_count
                accumulated_microbatches += 1
                if microstep >= measured_start:
                    forward_tokens += int(moved["attention_mask"].sum().item())
                    completion_tokens += loss.token_count
                if accumulated_microbatches == gradient_accumulation:
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_tokens)
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_tokens = accumulated_microbatches = 0
            if accumulated_microbatches:
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_tokens)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - float(started)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            results.append(
                {
                    "microbatch_size": candidate,
                    "gradient_accumulation": gradient_accumulation,
                    "warmup_microsteps": total_microsteps - measured_microsteps,
                    "measured_microsteps": measured_microsteps,
                    "maximum_batch_sequence_tokens": max(example.forward_tokens for example in longest[:candidate]),
                    "forward_tokens_per_second": forward_tokens / elapsed,
                    "completion_tokens_per_second": completion_tokens / elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": peak_reserved,
                    "total_device_memory_bytes": total_memory,
                    "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
                    "passed": True,
                }
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise
            results.append({"microbatch_size": candidate, "passed": False, "reason": "cuda_out_of_memory", "error": str(exc)[:500]})
            stop = True
        finally:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(initial[name].to(parameter.device, dtype=parameter.dtype))
    selected = select_microbatch(results, config)
    report = {
        "schema_version": BENCHMARK_SCHEMA,
        "passed": True,
        "git_sha": input_binding["git_sha"],
        "protocol_sha256": input_binding["protocol_sha256"],
        "token_audit_sha256": input_binding["token_audit_sha256"],
        "gpu_name": torch.cuda.get_device_properties(device).name,
        "candidate_microbatches": list(config.microbatch_candidates),
        "minimum_vram_headroom": config.minimum_vram_headroom,
        "results": results,
        "selected_microbatch_size": selected,
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(output_path, report)
    return report


def _evaluate(model: Any, tokenizer: Any, examples: Sequence[TokenizedSFTExample], config: SFTConfig, microbatch_size: int) -> dict[str, Any]:
    device = torch.device("cuda:0")
    sampler = LengthBucketBatchSampler([item.forward_tokens for item in examples], batch_size=microbatch_size, seed=config.seed)
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)
    model.eval()
    total_loss = 0.0
    total_tokens = exact = schema = 0
    with torch.inference_mode():
        for indices in sampler:
            raw = collator([examples[index] for index in indices])
            batch = _move_batch(raw, device)
            output = _forward(model, batch)
            loss = completion_only_cross_entropy(output.logits, batch["labels"])
            total_loss += float(loss.total.detach().cpu())
            total_tokens += loss.token_count
            predictions = output.logits[:, :-1, :].argmax(dim=-1)
            for row, example in enumerate(batch["examples"]):
                mask = loss.shifted_labels[row] != -100
                text = tokenizer.decode(predictions[row][mask].detach().cpu().tolist(), skip_special_tokens=True).strip()
                parsed = parse_command_output(text)
                target = parse_command_output(example.assistant_content)
                schema += int(parsed.schema_valid)
                exact += int(parsed.schema_valid and target.schema_valid and parsed.action == target.action)
    model.train()
    _require(total_tokens > 0, "M5 SFT dev evaluation has no labels")
    return {
        "dev_nll": total_loss / total_tokens,
        "teacher_forced_action_exact": exact / len(examples),
        "teacher_forced_schema_valid": schema / len(examples),
        "sample_count": len(examples),
        "completion_label_tokens": total_tokens,
    }


def train_one_epoch(
    *,
    model: Any,
    tokenizer: Any,
    train_examples: Sequence[TokenizedSFTExample],
    dev_examples: Sequence[TokenizedSFTExample],
    config: SFTConfig,
    input_binding: Mapping[str, Any],
    output_dir: Path,
    microbatch_size: int,
    recovery: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require(microbatch_size in config.microbatch_candidates, "M5 SFT microbatch is not frozen")
    _require(config.effective_batch_size % microbatch_size == 0, "M5 SFT microbatch does not divide effective batch")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=config.learning_rate, weight_decay=0.0)
    start_batch = optimizer_updates = 0
    cumulative_tokens = 0
    cumulative_loss = 0.0
    if recovery is not None:
        payload = torch.load(recovery["optimizer_rng"], map_location="cpu", weights_only=False)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        random.setstate(payload["python_random_state"])
        torch.set_rng_state(payload["torch_random_state"])
        torch.cuda.set_rng_state_all(payload["cuda_random_state"])
        start_batch = int(recovery["completed_microbatches"])
        optimizer_updates = int(recovery["optimizer_updates"])
        cumulative_tokens = int(recovery["cumulative_completion_label_tokens"])
        cumulative_loss = float(recovery["cumulative_loss_sum"])
    sampler = LengthBucketBatchSampler(
        [item.forward_tokens for item in train_examples],
        batch_size=microbatch_size,
        seed=config.seed,
    )
    sampler.set_epoch(0)
    batches = list(iter(sampler))
    _require(0 <= start_batch <= len(batches), "M5 SFT recovery batch offset drift")
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)
    parameters = _trainable_parameters(model)
    accumulation = config.effective_batch_size // microbatch_size
    accumulated_tokens = accumulated_microbatches = 0
    epoch_tokens = 0
    epoch_loss = 0.0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for batch_index, indices in enumerate(batches):
        if batch_index < start_batch:
            continue
        raw = collator([train_examples[index] for index in indices])
        batch = _move_batch(raw, torch.device("cuda:0"))
        output = _forward(model, batch)
        loss = completion_only_cross_entropy(output.logits, batch["labels"])
        loss.total.backward()
        accumulated_tokens += loss.token_count
        accumulated_microbatches += 1
        epoch_tokens += loss.token_count
        epoch_loss += float(loss.total.detach().cpu())
        cumulative_tokens += loss.token_count
        cumulative_loss += float(loss.total.detach().cpu())
        boundary = accumulated_microbatches == accumulation or batch_index + 1 == len(batches)
        if not boundary:
            continue
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_tokens)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        _require(bool(torch.isfinite(gradient_norm)), "M5 SFT gradient norm is non-finite")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_updates += 1
        accumulated_tokens = accumulated_microbatches = 0
        completed = batch_index + 1
        if optimizer_updates % config.checkpoint_interval_updates == 0 or completed == len(batches):
            save_recovery(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                output_dir=root,
                epoch=0,
                completed_microbatches=completed,
                optimizer_updates=optimizer_updates,
                cumulative_completion_label_tokens=cumulative_tokens,
                cumulative_loss_sum=cumulative_loss,
                input_binding=input_binding,
            )
            atomic_write_json(
                root / "training_progress.json",
                {
                    "schema_version": "m5_webshop_sft_progress_v1",
                    "complete": False,
                    "completed_microbatches": completed,
                    "total_microbatches": len(batches),
                    "optimizer_updates": optimizer_updates,
                    "git_sha": input_binding["git_sha"],
                    "protocol_sha256": input_binding["protocol_sha256"],
                },
            )
    _require(start_batch == 0 or recovery is not None, "M5 SFT resume state missing")
    dev = _evaluate(model, tokenizer, dev_examples, config, microbatch_size)
    final_adapter = root / "final_adapter_epoch_1"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    _replace_adapter(model, tokenizer, final_adapter)
    report = {
        "schema_version": TRAINER_SCHEMA,
        "complete": True,
        "git_sha": input_binding["git_sha"],
        "protocol_sha256": input_binding["protocol_sha256"],
        "token_audit_sha256": input_binding["token_audit_sha256"],
        "seed": config.seed,
        "maximum_sequence_tokens": config.maximum_sequence_tokens,
        "microbatch_size": microbatch_size,
        "effective_batch_size": config.effective_batch_size,
        "gradient_accumulation": accumulation,
        "optimizer_updates": optimizer_updates,
        "resumed": recovery is not None,
        "resumed_from_microbatch": start_batch,
        "train_completion_label_tokens": cumulative_tokens,
        "train_nll": cumulative_loss / max(1, cumulative_tokens),
        "train_completion_label_tokens_this_allocation": epoch_tokens,
        "train_nll_this_allocation": epoch_loss / max(1, epoch_tokens),
        "train_audit": summarize_examples(train_examples, config),
        "dev": dev,
        "final_adapter": str(final_adapter),
        "elapsed_s": time.monotonic() - started,
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(root / "training_report.json", report)
    return report
