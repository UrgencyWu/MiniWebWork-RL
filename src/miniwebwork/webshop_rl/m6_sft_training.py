"""M6 completion SFT with exact 90/10 Raw-policy retention updates.

The implementation keeps the Raw reference in the same frozen base model:
LoRA enabled gives the trainable SFT policy; ``disable_adapter`` gives the Raw
policy.  Retention rows never contribute cross entropy and therefore cannot
silently become oracle labels.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_file, sha256_json
from ..long_horizon_rl.learner import extract_tail_completion_logprobs
from ..long_horizon_rl.sft_trainer import (
    CompletionOnlyCollator,
    TokenizedSFTExample,
    _forward,
    _move_batch,
    completion_only_cross_entropy,
)
from ..m6_posttraining_protocol import validate_protocol
from .actions import parse_command_output
from .m6_corpus import validate_retention_states

TRAINER_SCHEMA = "m6_mini_sft_trainer_v1"
RECOVERY_SCHEMA = "m6_mini_sft_recovery_v1"
TOKEN_AUDIT_SCHEMA = "m6_mini_sft_token_audit_v1"
BENCHMARK_SCHEMA = "m6_mini_sft_microbatch_benchmark_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _chat_ids(value: Any, *, field: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    _require(isinstance(value, list) and all(isinstance(item, int) for item in value), f"{field} is not one token-id list")
    return [int(item) for item in value]


@dataclass(frozen=True)
class M6SFTConfig:
    seed: int
    maximum_sequence_tokens: int
    learning_rate: float
    raw_reference_kl: float
    effective_batch_size: int
    imitation_per_update: int
    retention_per_update: int
    microbatch_candidates: tuple[int, ...]
    minimum_vram_headroom: float
    maximum_epochs: int
    lora: dict[str, Any]
    chat_template_kwargs: dict[str, Any]

    @classmethod
    def from_protocol(cls, payload: Mapping[str, Any], *, seed: int = 20260812) -> "M6SFTConfig":
        validate_protocol(payload)
        sft = payload["sft"]
        effective = int(sft["effective_batch_size"])
        imitation = round(effective * float(sft["success_imitation_fraction"]))
        retention = effective - imitation
        _require(imitation == 9 and retention == 1, "M6 SFT batch is not exact 90/10")
        return cls(
            seed=seed,
            maximum_sequence_tokens=int(sft["maximum_sequence_tokens"]),
            learning_rate=float(sft["mini_learning_rate"]),
            raw_reference_kl=float(sft["mini_raw_reference_kl"]),
            effective_batch_size=effective,
            imitation_per_update=imitation,
            retention_per_update=retention,
            microbatch_candidates=tuple(int(item) for item in sft["microbatch_candidates"]),
            minimum_vram_headroom=float(sft["minimum_vram_headroom"]),
            maximum_epochs=int(sft["maximum_epochs"]),
            lora=dict(sft["lora"]),
            chat_template_kwargs=dict(sft["chat_template_kwargs"]),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical JSON representation used by durable artifacts."""

        return json.loads(json.dumps(asdict(self), sort_keys=True))


@dataclass(frozen=True)
class RawRetentionExample:
    sample_id: str
    task_id: str
    page_type: str
    prompt_token_ids: tuple[int, ...]
    raw_generated_token_ids: tuple[int, ...]

    @property
    def completion_tokens(self) -> int:
        return len(self.raw_generated_token_ids)

    @property
    def forward_tokens(self) -> int:
        return len(self.prompt_token_ids) + self.completion_tokens


def tokenize_sft_row(row: Mapping[str, Any], tokenizer: Any, config: M6SFTConfig) -> TokenizedSFTExample:
    messages = row.get("messages")
    completion = str(row.get("completion") or "")
    parsed = parse_command_output(completion)
    _require(isinstance(messages, list) and messages, "M6 SFT row lacks prompt messages")
    _require(parsed.strict_json_success and parsed.schema_valid, "M6 SFT completion is not strict command JSON")
    prompt_ids = _chat_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **config.chat_template_kwargs),
        field="M6 SFT prompt",
    )
    full_ids = _chat_ids(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": completion}],
            tokenize=True,
            add_generation_prompt=False,
            **config.chat_template_kwargs,
        ),
        field="M6 SFT full row",
    )
    _require(full_ids[: len(prompt_ids)] == prompt_ids, "M6 SFT completion is not a strict continuation")
    _require(len(full_ids) <= config.maximum_sequence_tokens, "M6 SFT row would truncate")
    labels = tuple([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :])
    completion_tokens = sum(item != -100 for item in labels)
    _require(completion_tokens > 0, "M6 SFT row has zero labels")
    return TokenizedSFTExample(
        sample_id=f"{row.get('trajectory_id')}:{int(row.get('turn_index', 0)):03d}",
        task_id=str(row.get("task_id") or ""),
        task_family="webshop",
        horizon_stratum="recovery" if row.get("trajectory_recovery") else "direct",
        input_ids=tuple(full_ids),
        labels=labels,
        assistant_content=completion,
        prompt_tokens=len(prompt_ids),
        completion_label_tokens=completion_tokens,
        untruncated_forward_tokens=len(full_ids),
    )


def load_jsonl_examples(path: Path, tokenizer: Any, config: M6SFTConfig) -> tuple[TokenizedSFTExample, ...]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = tuple(tokenize_sft_row(row, tokenizer, config) for row in rows)
    ids = [item.sample_id for item in examples]
    _require(examples and len(ids) == len(set(ids)), "M6 SFT examples are empty or duplicated")
    return examples


def load_retention_examples(path: Path, config: M6SFTConfig) -> tuple[RawRetentionExample, ...]:
    payload = validate_retention_states(json.loads(Path(path).read_text(encoding="utf-8")))
    examples = []
    for index, state in enumerate(payload["states"]):
        prompt = tuple(int(item) for item in state["prompt_token_ids"])
        generated = tuple(int(item) for item in state["raw_generated_token_ids"])
        _require(prompt and generated, "M6 retention row is empty")
        _require(len(prompt) + len(generated) <= config.maximum_sequence_tokens, "M6 retention row would truncate")
        examples.append(
            RawRetentionExample(
                sample_id=f"retention:{index:06d}",
                task_id=str(state["task_id"]),
                page_type=str(state["page_type"]),
                prompt_token_ids=prompt,
                raw_generated_token_ids=generated,
            )
        )
    return tuple(examples)


def build_update_schedule(
    imitation_count: int,
    retention_count: int,
    *,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Use every imitation row once in exact 9-imitation/1-retention updates."""

    _require(imitation_count >= 9 and retention_count > 0, "M6 SFT schedule lacks examples")
    imitation = list(range(imitation_count))
    retention = list(range(retention_count))
    random.Random(seed).shuffle(imitation)
    random.Random(seed + 1).shuffle(retention)
    # Incomplete final rows cannot honestly claim the per-optimizer-batch
    # 90/10 contract.  Leave at most eight deterministic tail rows unused and
    # record that fact in the training report.
    updates = imitation_count // 9
    schedule = []
    for update in range(updates):
        selected = imitation[update * 9 : (update + 1) * 9]
        _require(len(selected) == 9, "M6 SFT update is not exact 90/10")
        schedule.append({"update_index": update, "imitation_indices": selected, "retention_index": retention[update % len(retention)]})
    used = [index for row in schedule for index in row["imitation_indices"]]
    _require(len(used) == updates * 9 and len(set(used)) == len(used), "M6 SFT schedule duplicated imitation rows")
    return tuple(schedule)


def _collate_retention(examples: Sequence[RawRetentionExample], pad_token_id: int) -> dict[str, Any]:
    _require(examples, "M6 retention microbatch is empty")
    maximum_forward = max(item.forward_tokens for item in examples)
    maximum_completion = max(item.completion_tokens for item in examples)
    inputs, attention, generated, mask, lengths = [], [], [], [], []
    for example in examples:
        full = list(example.prompt_token_ids + example.raw_generated_token_ids)
        left = maximum_forward - len(full)
        tail = maximum_completion - example.completion_tokens
        inputs.append([pad_token_id] * left + full)
        attention.append([0] * left + [1] * len(full))
        generated.append(list(example.raw_generated_token_ids) + [pad_token_id] * tail)
        mask.append([True] * example.completion_tokens + [False] * tail)
        lengths.append(example.completion_tokens)
    attention_tensor = torch.tensor(attention, dtype=torch.long)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "attention_mask": attention_tensor,
        "position_ids": (attention_tensor.cumsum(dim=-1) - 1).clamp_min(0),
        "generated_token_ids": torch.tensor(generated, dtype=torch.long),
        "completion_mask": torch.tensor(mask, dtype=torch.bool),
        "completion_lengths": torch.tensor(lengths, dtype=torch.long),
        "logits_to_keep": maximum_completion + 1,
        "maximum_forward_tokens": maximum_forward,
        "examples": tuple(examples),
    }


def _move_retention(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    value = dict(batch)
    for field in ("input_ids", "attention_mask", "position_ids", "generated_token_ids", "completion_mask", "completion_lengths"):
        value[field] = batch[field].to(device, non_blocking=True)
    return value


def _retention_logprobs(model: Any, batch: Mapping[str, Any]) -> torch.Tensor:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
        logits_to_keep=int(batch["logits_to_keep"]),
    )
    logits = output.logits if hasattr(output, "logits") else output[0]
    logprobs, _ = extract_tail_completion_logprobs(logits, batch)
    return logprobs


def sampled_action_kl(current: torch.Tensor, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Non-negative k3 estimate of KL(Raw || current) on Raw action tokens.

    Retention completions are sampled by the frozen Raw policy.  For samples
    ``x ~ q=Raw``, ``exp(log p-log q) - (log p-log q) - 1`` is the k3
    estimator whose expectation is ``KL(q || p)``.  Reversing this ratio would
    silently estimate neither advertised direction from these samples.
    """

    _require(current.shape == raw.shape == mask.shape, "M6 retention KL tensor shape drift")
    log_ratio = current - raw
    k3 = torch.exp(log_ratio) - log_ratio - 1.0
    selected = k3[mask]
    _require(selected.numel() > 0, "M6 retention KL has no action tokens")
    value = selected.mean()
    _require(bool(torch.isfinite(value)), "M6 retention KL is non-finite")
    return value


def load_trainable_lora_model(base_model: Path, config: M6SFTConfig, *, resume_adapter: Path | None = None) -> Any:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        str(Path(base_model).expanduser().resolve()),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if resume_adapter is not None:
        model = PeftModel.from_pretrained(base, str(resume_adapter), is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=int(config.lora["r"]),
                lora_alpha=int(config.lora["alpha"]),
                lora_dropout=float(config.lora["dropout"]),
                target_modules=list(config.lora["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.train()
    return model


def benchmark_microbatches(
    *,
    model: Any,
    examples: Sequence[TokenizedSFTExample],
    retention_examples: Sequence[RawRetentionExample],
    tokenizer: Any,
    config: M6SFTConfig,
    output_path: Path,
    input_sha256: Mapping[str, str],
    measured_microsteps: int = 10,
) -> dict[str, Any]:
    """Select the largest frozen candidate with at least 15% VRAM headroom."""

    _require(torch.cuda.is_available(), "M6 SFT benchmark requires CUDA")
    _require(len(examples) >= max(config.microbatch_candidates), "M6 SFT benchmark lacks examples")
    _require(bool(retention_examples), "M6 SFT benchmark lacks Raw retention states")
    _require(10 <= measured_microsteps <= 20, "M6 SFT benchmark must measure 10-20 microsteps")
    device = torch.device("cuda:0")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(parameters, "M6 SFT benchmark has no trainable parameters")
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    longest = sorted(examples, key=lambda item: item.forward_tokens, reverse=True)
    longest_retention = max(retention_examples, key=lambda item: item.forward_tokens)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)
    results: list[dict[str, Any]] = []
    stop_after_oom = False
    for candidate in config.microbatch_candidates:
        if stop_after_oom:
            results.append({"microbatch_size": candidate, "passed": False, "reason": "larger_than_oom_candidate"})
            continue
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(initial[name].to(parameter.device, dtype=parameter.dtype))
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        batch = collator(longest[:candidate])
        retention_batch = _collate_retention([longest_retention], tokenizer.pad_token_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = None
        forward_tokens = completion_tokens = 0
        try:
            for microstep in range(measured_microsteps + 2):
                if microstep == 2:
                    torch.cuda.synchronize(device)
                    started = time.monotonic()
                moved = _move_batch(batch, device)
                result = completion_only_cross_entropy(_forward(model, moved).logits, moved["labels"])
                (result.total / result.token_count).backward()
                moved_retention = _move_retention(retention_batch, device)
                model.eval()
                with torch.inference_mode(), model.disable_adapter():
                    raw_logprobs = _retention_logprobs(model, moved_retention)
                model.train()
                current_logprobs = _retention_logprobs(model, moved_retention)
                retention_kl = sampled_action_kl(
                    current_logprobs,
                    raw_logprobs,
                    moved_retention["completion_mask"],
                )
                (config.raw_reference_kl * retention_kl).backward()
                if microstep >= 2:
                    retention_forward = int(moved_retention["attention_mask"].sum().item())
                    retention_completion = int(moved_retention["completion_mask"].sum().item())
                    # Retention evaluates Raw and current policy on the same
                    # action tokens, hence two forward passes.
                    forward_tokens += int(moved["attention_mask"].sum().item()) + 2 * retention_forward
                    completion_tokens += result.token_count + 2 * retention_completion
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - float(started)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            headroom = 1.0 - peak_reserved / total_memory
            results.append(
                {
                    "microbatch_size": candidate,
                    "measured_microsteps": measured_microsteps,
                    "maximum_batch_sequence_tokens": max(item.forward_tokens for item in longest[:candidate]),
                    "maximum_retention_sequence_tokens": longest_retention.forward_tokens,
                    "benchmark_objective": "completion_CE_plus_sampled_Raw_reference_KL",
                    "forward_tokens_per_second": forward_tokens / elapsed,
                    "completion_tokens_per_second": completion_tokens / elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": peak_reserved,
                    "total_device_memory_bytes": total_memory,
                    "vram_headroom_fraction": headroom,
                    "headroom_gate_passed": headroom >= config.minimum_vram_headroom,
                    "passed": True,
                }
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise
            results.append(
                {
                    "microbatch_size": candidate,
                    "passed": False,
                    "reason": "cuda_out_of_memory",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )
            stop_after_oom = True
        finally:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(initial[name].to(parameter.device, dtype=parameter.dtype))
    eligible = [
        item
        for item in results
        if item.get("passed") is True
        and item.get("headroom_gate_passed") is True
        and float(item["vram_headroom_fraction"]) >= config.minimum_vram_headroom
    ]
    _require(eligible, "no M6 SFT microbatch retained the frozen VRAM headroom")
    selected = max(int(item["microbatch_size"]) for item in eligible)
    report = {
        "schema_version": BENCHMARK_SCHEMA,
        "passed": True,
        "development_only": True,
        "input_sha256": dict(input_sha256),
        "candidate_microbatches": list(config.microbatch_candidates),
        "minimum_vram_headroom": config.minimum_vram_headroom,
        "gpu_name": torch.cuda.get_device_properties(device).name,
        "results": results,
        "selected_microbatch_size": selected,
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(output_path, report)
    return report


def _atomic_adapter(model: Any, tokenizer: Any, destination: Path) -> None:
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


def _atomic_torch_save(value: Any, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{time.time_ns()}")
    torch.save(value, temporary)
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
    completed_updates: int,
    schedule_sha256: str,
    input_sha256: Mapping[str, str],
    cumulative_metrics: Mapping[str, float | int],
) -> None:
    root = output_dir / "recovery"
    root.mkdir(parents=True, exist_ok=True)
    generation_name = f"update_{completed_updates:06d}"
    generation = root / generation_name
    staging = root / f".{generation_name}.staging-{time.time_ns()}"
    staging.mkdir()
    # Adapter, optimizer/RNG and state become visible as one immutable
    # generation.  The tiny latest.json pointer is replaced only after the
    # generation directory has been fsynced, so a 24h signal between files can
    # never combine artifacts from different optimizer boundaries.
    _atomic_adapter(model, tokenizer, staging / "adapter")
    _atomic_torch_save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        },
        staging / "optimizer_rng.pt",
    )
    state = {
        "schema_version": RECOVERY_SCHEMA,
        "completed_updates": completed_updates,
        "optimizer_boundary": True,
        "schedule_sha256": schedule_sha256,
        "input_sha256": dict(input_sha256),
        "cumulative_metrics": dict(cumulative_metrics),
        "adapter_sha256": directory_sha256(staging / "adapter"),
        "optimizer_rng_sha256": sha256_file(staging / "optimizer_rng.pt"),
    }
    state["content_sha256"] = sha256_json(state)
    atomic_write_json(staging / "state.json", state)
    for file_path in staging.rglob("*"):
        if file_path.is_file():
            descriptor = os.open(file_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    pointer = {
        "schema_version": "m6_mini_sft_recovery_pointer_v1",
        "generation": generation_name,
        "completed_updates": completed_updates,
        "state_content_sha256": state["content_sha256"],
    }
    pointer["content_sha256"] = sha256_json(pointer)
    if generation.exists():
        # The only legitimate way this can happen is a process death after a
        # complete generation rename but before latest.json advanced. Preserve
        # that uncommitted generation for diagnosis, then deterministically
        # replay the update from the last committed boundary.
        interrupted = root / "interrupted_generations"
        interrupted.mkdir(exist_ok=True)
        os.rename(
            generation,
            interrupted / f"{generation_name}_{time.time_ns()}",
        )
    os.rename(staging, generation)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    atomic_write_json(root / "latest.json", pointer)


def load_recovery(output_dir: Path, schedule_sha256: str, input_sha256: Mapping[str, str]) -> dict[str, Any] | None:
    root = output_dir / "recovery"
    pointer_path = root / "latest.json"
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    expected_pointer = dict(pointer)
    observed_pointer_hash = expected_pointer.pop("content_sha256", None)
    _require(
        observed_pointer_hash == sha256_json(expected_pointer),
        "M6 SFT recovery pointer self-hash drift",
    )
    _require(
        pointer.get("schema_version") == "m6_mini_sft_recovery_pointer_v1",
        "M6 SFT recovery pointer schema drift",
    )
    generation_name = pointer.get("generation")
    _require(
        isinstance(generation_name, str)
        and generation_name.startswith("update_")
        and "/" not in generation_name,
        "M6 SFT recovery generation identity drift",
    )
    generation = root / generation_name
    state_path = generation / "state.json"
    _require(state_path.is_file(), "M6 SFT recovery generation state is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected = dict(state)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 SFT recovery self-hash drift")
    _require(state.get("optimizer_boundary") is True, "M6 SFT recovery is not an optimizer boundary")
    _require(state.get("schedule_sha256") == schedule_sha256, "M6 SFT recovery schedule drift")
    _require(state.get("input_sha256") == dict(input_sha256), "M6 SFT recovery inputs drift")
    _require(pointer.get("completed_updates") == state.get("completed_updates"), "M6 SFT recovery pointer update drift")
    _require(pointer.get("state_content_sha256") == state.get("content_sha256"), "M6 SFT recovery pointer state drift")
    _require((generation / "adapter").is_dir() and (generation / "optimizer_rng.pt").is_file(), "M6 SFT recovery files missing")
    _require(
        state.get("adapter_sha256") == directory_sha256(generation / "adapter"),
        "M6 SFT recovery adapter hash drift",
    )
    _require(
        state.get("optimizer_rng_sha256") == sha256_file(generation / "optimizer_rng.pt"),
        "M6 SFT recovery optimizer/RNG hash drift",
    )
    return {
        **state,
        "generation": generation,
        "adapter": generation / "adapter",
        "optimizer_rng": generation / "optimizer_rng.pt",
    }


def train_mini_sft(
    *,
    base_model: Path,
    tokenizer: Any,
    train_examples: Sequence[TokenizedSFTExample],
    dev_examples: Sequence[TokenizedSFTExample],
    retention_examples: Sequence[RawRetentionExample],
    output_dir: Path,
    input_sha256: Mapping[str, str],
    config: M6SFTConfig,
    microbatch_size: int,
) -> dict[str, Any]:
    """Train one development-only epoch with optimizer-boundary recovery."""

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M6 mini SFT requires exactly one GPU")
    _require(microbatch_size in config.microbatch_candidates, "M6 SFT microbatch is not frozen")
    _require(config.maximum_epochs == 1, "M6 mini SFT exceeds one epoch")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    schedule = build_update_schedule(len(train_examples), len(retention_examples), seed=config.seed)
    schedule_sha = sha256_json(schedule)
    report_path = root / "training_report.json"
    if report_path.is_file():
        report = validate_mini_sft_training_report(report_path)
        _require(report.get("input_sha256") == dict(input_sha256), "M6 completed SFT input drift")
        _require(report.get("config") == config.to_payload(), "M6 completed SFT config drift")
        _require(report.get("schedule_sha256") == schedule_sha, "M6 completed SFT schedule drift")
        return report
    recovery = load_recovery(root, schedule_sha, input_sha256)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = load_trainable_lora_model(base_model, config, resume_adapter=recovery["adapter"] if recovery else None)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(parameters, "M6 mini SFT has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
    start_update = 0
    if recovery:
        checkpoint = torch.load(recovery["optimizer_rng"], map_location="cpu", weights_only=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        start_update = int(recovery["completed_updates"])
    completion_collator = CompletionOnlyCollator(tokenizer.pad_token_id)
    device = torch.device("cuda:0")
    cumulative = {
        "ce_sum": 0.0,
        "ce_tokens": 0,
        "kl_sum": 0.0,
        "retention_tokens": 0,
        "gradient_sum": 0.0,
    }
    if recovery:
        cumulative.update(recovery.get("cumulative_metrics", {}))
    started = time.monotonic()
    for scheduled in schedule[start_update:]:
        optimizer.zero_grad(set_to_none=True)
        imitation_indices = scheduled["imitation_indices"]
        _require(imitation_indices, "M6 SFT final update contains no imitation rows")
        update_ce_tokens = 0
        for start in range(0, len(imitation_indices), microbatch_size):
            selected = [train_examples[index] for index in imitation_indices[start : start + microbatch_size]]
            batch = _move_batch(completion_collator(selected), device)
            result = completion_only_cross_entropy(_forward(model, batch).logits, batch["labels"])
            result.total.backward()
            update_ce_tokens += result.token_count
            cumulative["ce_sum"] += float(result.total.detach().cpu())
            cumulative["ce_tokens"] += result.token_count
        retention = _move_retention(
            _collate_retention([retention_examples[scheduled["retention_index"]]], tokenizer.pad_token_id),
            device,
        )
        model.eval()
        with torch.inference_mode(), model.disable_adapter():
            raw_logprobs = _retention_logprobs(model, retention)
        model.train()
        current_logprobs = _retention_logprobs(model, retention)
        kl = sampled_action_kl(current_logprobs, raw_logprobs, retention["completion_mask"])
        # CE was accumulated as a token sum. Normalize both objectives at the
        # optimizer boundary so varying action lengths cannot change beta.
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(update_ce_tokens)
        (config.raw_reference_kl * kl).backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        _require(bool(torch.isfinite(gradient)), "M6 SFT gradient is non-finite")
        optimizer.step()
        cumulative["kl_sum"] += float(kl.detach().cpu())
        cumulative["retention_tokens"] += int(retention["completion_mask"].sum().item())
        cumulative["gradient_sum"] += float(gradient.detach().cpu())
        completed = int(scheduled["update_index"]) + 1
        save_recovery(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            output_dir=root,
            completed_updates=completed,
            schedule_sha256=schedule_sha,
            input_sha256=input_sha256,
            cumulative_metrics=cumulative,
        )
    # Teacher-forced dev is diagnostic only; closed-loop K4 selects the stage.
    model.eval()
    dev_loss = dev_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(dev_examples), microbatch_size):
            batch = _move_batch(completion_collator(dev_examples[start : start + microbatch_size]), device)
            result = completion_only_cross_entropy(_forward(model, batch).logits, batch["labels"])
            dev_loss += float(result.total.cpu())
            dev_tokens += result.token_count
    model.train()
    final_adapter = root / "final_adapter"
    _atomic_adapter(model, tokenizer, final_adapter)
    report = {
        "schema_version": TRAINER_SCHEMA,
        "complete": True,
        "development_only": True,
        "formal_checkpoint_reusable": False,
        "config": config.to_payload(),
        "input_sha256": dict(input_sha256),
        "schedule_sha256": schedule_sha,
        "optimizer_updates": len(schedule),
        "resumed": recovery is not None,
        "resumed_from_update": start_update,
        "imitation_sample_count": len(train_examples),
        "imitation_sample_count_used": len(schedule) * 9,
        "imitation_tail_sample_count_unused": len(train_examples) - len(schedule) * 9,
        "retention_state_count": len(retention_examples),
        "optimizer_batch_contract": {"imitation_slots": 9, "retention_slots": 1, "retention_supervised_labels": 0},
        "train_completion_label_tokens": int(cumulative["ce_tokens"]),
        "retention_action_tokens": int(cumulative["retention_tokens"]),
        "train_nll": cumulative["ce_sum"] / max(1, cumulative["ce_tokens"]),
        "mean_retention_kl": cumulative["kl_sum"] / max(1, len(schedule)),
        "mean_gradient_norm": cumulative["gradient_sum"] / max(1, len(schedule)),
        "dev_nll": dev_loss / max(1, dev_tokens),
        "final_adapter": str(final_adapter),
        "final_adapter_sha256": directory_sha256(final_adapter),
        "elapsed_seconds": time.monotonic() - started,
    }
    _require(report["optimizer_updates"] > 0 and math.isfinite(report["train_nll"]), "M6 SFT did not produce a finite update")
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(root / "training_report.json", report)
    return report


def validate_mini_sft_training_report(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    _require(report.get("schema_version") == TRAINER_SCHEMA and report.get("complete") is True, "M6 SFT report schema drift")
    _require(report.get("development_only") is True and report.get("formal_checkpoint_reusable") is False, "M6 SFT report scope drift")
    expected = dict(report)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 SFT report self-hash drift")
    adapter = Path(str(report.get("final_adapter", "")))
    _require(adapter.is_dir(), "M6 completed SFT adapter is missing")
    _require(directory_sha256(adapter) == report.get("final_adapter_sha256"), "M6 completed SFT adapter hash drift")
    _require(int(report.get("optimizer_updates", 0)) > 0, "M6 completed SFT has no optimizer updates")
    for field in ("train_nll", "mean_retention_kl", "mean_gradient_norm", "dev_nll"):
        _require(math.isfinite(float(report.get(field))), f"M6 completed SFT {field} is non-finite")
    return report


def build_token_audit(
    *,
    train_examples: Sequence[TokenizedSFTExample],
    dev_examples: Sequence[TokenizedSFTExample],
    retention_examples: Sequence[RawRetentionExample],
    base_model: Path,
    input_files: Mapping[str, Path],
    config: M6SFTConfig,
) -> dict[str, Any]:
    def summary(values: Sequence[TokenizedSFTExample]) -> dict[str, Any]:
        return {
            "sample_count": len(values),
            "unique_sample_count": len({item.sample_id for item in values}),
            "task_count": len({item.task_id for item in values}),
            "effective_completion_label_tokens": sum(item.completion_label_tokens for item in values),
            "zero_completion_label_sample_count": sum(item.completion_label_tokens == 0 for item in values),
            "truncated_sample_count": sum(item.forward_tokens > config.maximum_sequence_tokens for item in values),
            "maximum_forward_sequence_tokens": max(item.forward_tokens for item in values),
        }

    model = Path(base_model).expanduser().resolve()
    tokenizer_files = {
        name: sha256_file(model / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")
        if (model / name).is_file()
    }
    train_summary = summary(train_examples)
    dev_summary = summary(dev_examples)
    checks = {
        "train_unique": train_summary["sample_count"] == train_summary["unique_sample_count"],
        "dev_unique": dev_summary["sample_count"] == dev_summary["unique_sample_count"],
        "zero_labels": train_summary["zero_completion_label_sample_count"] == dev_summary["zero_completion_label_sample_count"] == 0,
        "zero_truncation": train_summary["truncated_sample_count"] == dev_summary["truncated_sample_count"] == 0,
        "retention_no_labels": bool(retention_examples),
        "mini_label_token_range": 20_000 <= train_summary["effective_completion_label_tokens"] <= 80_000,
    }
    report = {
        "schema_version": TOKEN_AUDIT_SCHEMA,
        "passed": all(checks.values()),
        "development_only": True,
        "base_model": str(model),
        "config": config.to_payload(),
        "input_file_sha256": {name: sha256_file(path) for name, path in input_files.items()},
        "tokenizer_file_sha256": tokenizer_files,
        "splits": {"train": train_summary, "dev": dev_summary},
        "retention": {
            "state_count": len(retention_examples),
            "supervised_label_count": 0,
            "maximum_forward_sequence_tokens": max(item.forward_tokens for item in retention_examples),
        },
        "checks": checks,
    }
    report["content_sha256"] = sha256_json(report)
    return report
