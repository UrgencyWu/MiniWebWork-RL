#!/usr/bin/env python3
"""Probe or train a capability-weighted completion-only Raw35 LoRA.

The weighted corpus already encodes the frozen hierarchy
capability -> task -> path -> action row -> mean action-token CE. This
runtime preserves those row masses exactly and updates only text token-mixer
projections. MoE experts, routers, the vision tower, and MTP remain frozen.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from miniwebwork.long_horizon_rl.contracts import (  # noqa: E402
    atomic_write_json,
    directory_sha256,
    sha256_file,
    sha256_json,
)
from miniwebwork.webshop_rl.phase10_corrective import (  # noqa: E402
    parameter_displacement,
    parameter_snapshot,
    parameter_tensor_sha256,
)

BASE_MODEL = Path("/data/share/model/Qwen3.5-35B-A3B")
CAPABILITY_WEIGHTS = {"nav": 0.20, "match": 0.40, "finish": 0.40}
SEED = 20260865
MAX_SEQUENCE_TOKENS = 16_384
LEARNING_RATE = 5e-5
MAXIMUM_EPOCHS = 2
LORA_CONFIG = {"r": 8, "alpha": 16, "dropout": 0.05}
TEXT_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)
PROBE_TASK_QUOTAS = {"nav": 1, "match": 2, "finish": 2}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def _chat_ids(value: Any, *, label: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    _require(isinstance(value, list) and all(isinstance(item, int) for item in value), f"{label} is not one token list")
    return [int(item) for item in value]


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


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        _require(isinstance(pad_token_id, int) and pad_token_id >= 0, "Phase10-C pad token is invalid")
        self.pad_token_id = pad_token_id

    def __call__(self, examples: Sequence[TokenizedSFTExample]) -> dict[str, Any]:
        _require(bool(examples), "Phase10-C cannot collate an empty batch")
        maximum = max(example.forward_tokens for example in examples)
        maximum_completion = max(example.completion_label_tokens for example in examples)
        input_ids, labels, attention = [], [], []
        for example in examples:
            padding = maximum - example.forward_tokens
            input_ids.append([self.pad_token_id] * padding + list(example.input_ids))
            labels.append([-100] * padding + list(example.labels))
            attention.append([0] * padding + [1] * example.forward_tokens)
        attention_tensor = torch.tensor(attention, dtype=torch.long)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": attention_tensor,
            "position_ids": (attention_tensor.cumsum(dim=-1) - 1).clamp_min(0),
            "logits_to_keep": min(maximum, maximum_completion + 1),
        }


def completion_mean_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _require(logits.ndim == 3 and labels.ndim == 2 and logits.shape[0] == labels.shape[0],
             "Phase10-C loss tensor shape drift")
    tail_labels = labels[:, -logits.shape[1] :]
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = tail_labels[:, 1:].contiguous()
    token_count = int((shifted_labels != -100).sum().item())
    _require(token_count > 0, "Phase10-C loss has zero action labels")
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / token_count


@dataclass(frozen=True)
class WeightedExample:
    tokenized: TokenizedSFTExample
    capability: str
    trajectory_id: str
    row_weight: float


def load_and_validate_rows(corpus_dir: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(split in {"train", "dev"}, "Phase10-C weighted split must be train or dev")
    root = corpus_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    data_path = root / f"{split}.jsonl"
    _require(manifest_path.is_file() and data_path.is_file(), "Phase10-C weighted corpus is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("content_sha256") == _self_hash(manifest), "Phase10-C weighted manifest hash drift")
    _require(manifest.get("schema_version") == "m6_phase10c_weighted_raw35_sft_corpus_v1", "Phase10-C manifest schema drift")
    _require(manifest.get("base_model") == str(BASE_MODEL), "Phase10-C Raw35 model identity drift")
    _require(manifest.get("capability_weights") == CAPABILITY_WEIGHTS, "Phase10-C capability weights drift")
    _require(manifest.get("loss_hierarchy") == "capability_task_path_action_row_token_mean_v1", "Phase10-C loss hierarchy drift")
    _require(manifest.get("task_overlap") == 0, "Phase10-C weighted corpus task overlap")
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_count = int(manifest[f"{split}_action_row_count"])
    expected_tasks = int(manifest[f"{split}_task_count"])
    _require(len(rows) == expected_count and len({str(row.get('task_id')) for row in rows}) == expected_tasks,
             f"Phase10-C {split} row/task count drift")
    _require(all(row.get("source") == "raw35_replay_strict_self_exploration" for row in rows),
             "Phase10-C weighted source drift")
    _require(all(row.get("capability") in CAPABILITY_WEIGHTS for row in rows), "Phase10-C capability identity drift")
    _require(all(row.get("loss_hierarchy") == manifest["loss_hierarchy"] for row in rows), "Phase10-C row hierarchy drift")
    _require(all(float(row.get("row_loss_weight", 0.0)) > 0 for row in rows), "Phase10-C non-positive row weight")
    _require(math.isclose(sum(float(row["row_loss_weight"]) for row in rows), 1.0, abs_tol=1e-12),
             "Phase10-C row loss mass drift")
    observed = {
        capability: sum(float(row["row_loss_weight"]) for row in rows if row["capability"] == capability)
        for capability in CAPABILITY_WEIGHTS
    }
    _require(all(math.isclose(observed[key], value, abs_tol=1e-12) for key, value in CAPABILITY_WEIGHTS.items()),
             "Phase10-C capability loss mass drift")
    return rows, {
        "manifest": manifest,
        "manifest_file_sha256": sha256_file(manifest_path),
        "jsonl_sha256": sha256_file(data_path),
    }


def tokenize_weighted_row(row: Mapping[str, Any], tokenizer: Any) -> WeightedExample:
    messages = row.get("messages")
    completion = str(row.get("completion") or "")
    _require(isinstance(messages, list) and messages and completion, "Phase10-C weighted row lacks prompt/completion")
    prompt_ids = _chat_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False),
        label="Phase10-C prompt",
    )
    full_ids = _chat_ids(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": completion}],
            tokenize=True, add_generation_prompt=False, enable_thinking=False,
        ),
        label="Phase10-C full row",
    )
    _require(len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids,
             "Phase10-C completion is not a strict continuation")
    _require(len(full_ids) <= MAX_SEQUENCE_TOKENS, "Phase10-C row exceeds the no-truncation limit")
    labels = tuple([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :])
    label_tokens = sum(item != -100 for item in labels)
    _require(label_tokens > 0, "Phase10-C row has zero action labels")
    task_id = str(row.get("task_id") or "")
    trajectory_id = str(row.get("trajectory_id") or "")
    turn_index = int(row.get("turn_index", -1))
    _require(task_id and trajectory_id and turn_index >= 0, "Phase10-C weighted provenance is incomplete")
    tokenized = TokenizedSFTExample(
        sample_id=f"{trajectory_id}:{turn_index:03d}",
        task_id=task_id,
        task_family="webshop",
        horizon_stratum=str(row["capability"]),
        input_ids=tuple(full_ids),
        labels=labels,
        assistant_content=completion,
        prompt_tokens=len(prompt_ids),
        completion_label_tokens=label_tokens,
        untruncated_forward_tokens=len(full_ids),
    )
    return WeightedExample(tokenized, str(row["capability"]), trajectory_id, float(row["row_loss_weight"]))


def select_probe_examples(examples: Sequence[WeightedExample], *, seed: int = SEED) -> list[WeightedExample]:
    """Choose five unique tasks whose equal row masses realize 20/40/40."""

    selected: list[WeightedExample] = []
    for capability, quota in PROBE_TASK_QUOTAS.items():
        by_task: dict[str, list[WeightedExample]] = defaultdict(list)
        for example in examples:
            if example.capability == capability:
                by_task[example.tokenized.task_id].append(example)
        tasks = sorted(by_task, key=lambda task: sha256_json({"seed": seed, "capability": capability, "task": task}))
        _require(len(tasks) >= quota, f"Phase10-C probe lacks {capability} tasks")
        for task in tasks[:quota]:
            rows = sorted(by_task[task], key=lambda item: sha256_json({"seed": seed, "sample": item.tokenized.sample_id}))
            selected.append(replace(rows[0], row_weight=CAPABILITY_WEIGHTS[capability] / quota))
    _require(len(selected) == 5 and len({item.tokenized.task_id for item in selected}) == 5,
             "Phase10-C probe task selection drift")
    _require(math.isclose(sum(item.row_weight for item in selected), 1.0, abs_tol=1e-12),
             "Phase10-C probe loss mass drift")
    return selected


def resolve_text_target_modules(model: Any) -> tuple[str, ...]:
    """Return exact text-layer linear module names, never suffix-wide matches."""

    targets = []
    counts = Counter()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        in_text_layer = ".layers." in name and "visual" not in name and not name.startswith("mtp.")
        if in_text_layer and suffix in TEXT_TARGET_SUFFIXES and isinstance(module, torch.nn.Linear):
            targets.append(name)
            counts[suffix] += 1
    for suffix in TEXT_TARGET_SUFFIXES:
        _require(counts[suffix] > 0, f"Phase10-C Raw35 lacks text target module {suffix}")
    _require(len(targets) == len(set(targets)), "Phase10-C target module duplication")
    return tuple(sorted(targets))


def _input_device(model: Any) -> torch.device:
    device = model.get_input_embeddings().weight.device
    _require(device.type == "cuda", "Phase10-C input embeddings are not resident on CUDA")
    return device


def _forward_loss(model: Any, collator: CompletionOnlyCollator, example: WeightedExample) -> torch.Tensor:
    batch = collator([example.tokenized])
    device = _input_device(model)
    output = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        position_ids=batch["position_ids"].to(device),
        use_cache=False,
        logits_to_keep=int(batch["logits_to_keep"]),
    )
    logits = output.logits if hasattr(output, "logits") else output[0]
    return completion_mean_cross_entropy(logits, batch["labels"].to(logits.device))


def evaluate_weighted(model: Any, examples: Sequence[WeightedExample], collator: CompletionOnlyCollator) -> dict[str, Any]:
    was_training = bool(model.training)
    model.eval()
    total = 0.0
    by_capability = defaultdict(float)
    with torch.inference_mode():
        for example in examples:
            loss = float(_forward_loss(model, collator, example).detach().cpu())
            total += example.row_weight * loss
            by_capability[example.capability] += example.row_weight * loss
    if was_training:
        model.train()
    return {
        "weighted_nll": total,
        "capability_mean_nll": {
            capability: by_capability[capability] / CAPABILITY_WEIGHTS[capability]
            for capability in CAPABILITY_WEIGHTS
        },
        "sample_count": len(examples),
        "task_count": len({item.tokenized.task_id for item in examples}),
        "completion_label_tokens": sum(item.tokenized.completion_label_tokens for item in examples),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_raw35_lora(base_model: Path) -> tuple[Any, tuple[str, ...], dict[str, Any]]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 2, "Phase10-C Raw35 LoRA requires exactly two GPUs")
    # Keep each device below half of its capacity so Accelerate cannot place
    # the ~70GB BF16 checkpoint on GPU0 alone. This both forces two-device
    # model parallelism and leaves activation/gradient workspace on each GPU.
    max_memory = {
        index: int(torch.cuda.get_device_properties(index).total_memory * 0.45)
        for index in range(torch.cuda.device_count())
    }
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model.expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
    )
    base.config.use_cache = False
    targets = resolve_text_target_modules(base)
    model = get_peft_model(
        base,
        LoraConfig(
            r=LORA_CONFIG["r"], lora_alpha=LORA_CONFIG["alpha"], lora_dropout=LORA_CONFIG["dropout"],
            target_modules=list(targets), bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    placement = Counter(str(parameter.device) for parameter in model.parameters())
    return model, targets, {"max_memory_bytes": max_memory, "parameter_placement": dict(placement)}


def _gpu_memory() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_properties(index).name,
            "total_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            "reserved_headroom_fraction": 1.0 - torch.cuda.max_memory_reserved(index) / torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ]


def _save_adapter(model: Any, tokenizer: Any, destination: Path) -> str:
    _require(not destination.exists(), f"Phase10-C refuses to overwrite adapter: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    temporary.replace(destination)
    return directory_sha256(destination)


def _train_update(
    model: Any,
    examples: Sequence[WeightedExample],
    collator: CompletionOnlyCollator,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
) -> dict[str, Any]:
    ordered = sorted(examples, key=lambda item: sha256_json({"seed": seed, "sample": item.tokenized.sample_id}))
    optimizer.zero_grad(set_to_none=True)
    objective = 0.0
    by_capability = defaultdict(float)
    for example in ordered:
        mean_loss = _forward_loss(model, collator, example)
        weighted = mean_loss * example.row_weight
        weighted.backward()
        value = float(weighted.detach().cpu())
        objective += value
        by_capability[example.capability] += value
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    _require(bool(torch.isfinite(gradient_norm)) and float(gradient_norm) > 0, "Phase10-C gradient is zero/non-finite")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "weighted_train_nll": objective,
        "capability_weighted_contribution": dict(sorted(by_capability.items())),
        "gradient_norm": float(gradient_norm),
        "row_count": len(examples),
        "task_count": len({item.tokenized.task_id for item in examples}),
        "completion_label_tokens": sum(item.tokenized.completion_label_tokens for item in examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(args.base_model.expanduser().resolve() == BASE_MODEL, "Phase10-C base model path drift")
    output = args.output_dir.expanduser().resolve()
    _require(not (output / "training_report.json").exists(), "Phase10-C training report already exists")
    output.mkdir(parents=True, exist_ok=True)
    _set_seed(SEED)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    train_rows, train_binding = load_and_validate_rows(args.corpus_dir, "train")
    dev_rows, dev_binding = load_and_validate_rows(args.corpus_dir, "dev")
    _require(train_binding["manifest"]["content_sha256"] == dev_binding["manifest"]["content_sha256"],
             "Phase10-C train/dev manifest drift")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples = [tokenize_weighted_row(row, tokenizer) for row in train_rows]
    dev_examples = [tokenize_weighted_row(row, tokenizer) for row in dev_rows]
    _require(len({item.tokenized.sample_id for item in train_examples}) == len(train_examples), "Phase10-C train duplicate")
    _require(not ({item.tokenized.task_id for item in train_examples} & {item.tokenized.task_id for item in dev_examples}),
             "Phase10-C tokenized train/dev task overlap")
    if args.mode == "probe":
        active_train = select_probe_examples(train_examples)
        active_dev = select_probe_examples(dev_examples, seed=SEED + 1)
        maximum_epochs = 1
    else:
        active_train = train_examples
        active_dev = dev_examples
        maximum_epochs = MAXIMUM_EPOCHS

    started = time.monotonic()
    model, target_modules, placement = load_raw35_lora(BASE_MODEL)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(trainable, "Phase10-C LoRA has no trainable parameters")
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    initial_sha = parameter_tensor_sha256(model)
    initial_parameters = parameter_snapshot(model)
    pre_train = evaluate_weighted(model, active_train, collator)
    pre_dev = evaluate_weighted(model, active_dev, collator)
    history = []
    best_nll = math.inf
    best_epoch = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        update = _train_update(model, active_train, collator, optimizer, seed=SEED + epoch)
        post_train = evaluate_weighted(model, active_train, collator)
        post_dev = evaluate_weighted(model, active_dev, collator)
        adapter = output / f"checkpoint_epoch_{epoch:02d}"
        adapter_sha = _save_adapter(model, tokenizer, adapter)
        history.append({
            "epoch": epoch,
            "optimizer_updates": epoch,
            "update": update,
            "train": post_train,
            "dev": post_dev,
            "adapter": str(adapter),
            "adapter_sha256": adapter_sha,
        })
        if post_dev["weighted_nll"] < best_nll:
            best_nll = float(post_dev["weighted_nll"])
            best_epoch = epoch
    output_sha = parameter_tensor_sha256(model)
    displacement = parameter_displacement(initial_parameters, model)
    best_source = output / f"checkpoint_epoch_{best_epoch:02d}"
    final_adapter = output / "final_adapter"
    shutil.copytree(best_source, final_adapter)
    final_adapter_sha = directory_sha256(final_adapter)
    memory = _gpu_memory()
    final = history[-1]
    gates = {
        "completion_only_prefix_mask": all(
            all(label == -100 for label in item.tokenized.labels[: item.tokenized.prompt_tokens])
            and item.tokenized.completion_label_tokens > 0
            for item in [*active_train, *active_dev]
        ),
        "exact_capability_loss_mass": all(
            math.isclose(sum(item.row_weight for item in active_train if item.capability == capability), weight, abs_tol=1e-12)
            for capability, weight in CAPABILITY_WEIGHTS.items()
        ),
        "real_parameter_update": initial_sha != output_sha and displacement["changed_tensor_count"] > 0,
        "finite_gradient": all(math.isfinite(item["update"]["gradient_norm"]) for item in history),
        "train_objective_decreased": final["train"]["weighted_nll"] < pre_train["weighted_nll"],
        "dev_nll_safe": best_nll <= pre_dev["weighted_nll"] + 0.10,
        "gpu_headroom_safe": min(item["reserved_headroom_fraction"] for item in memory) >= 0.05,
        "two_gpu_model_parallel": len({str(parameter.device) for parameter in model.parameters()}) >= 2,
    }
    report = {
        "schema_version": "m6_phase10c_weighted_raw35_lora_v1",
        "complete": True,
        "development_only": True,
        "mode": args.mode,
        "formal_checkpoint_reusable": args.mode == "train",
        "base_model": str(BASE_MODEL),
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "lora": LORA_CONFIG,
        "maximum_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "optimizer_updates": len(history),
        "loss_hierarchy": "capability_task_path_action_row_token_mean_v1",
        "capability_weights": CAPABILITY_WEIGHTS,
        "target_policy": "text_token_mixer_projections_only; experts_routers_vision_mtp_frozen",
        "target_module_count": len(target_modules),
        "target_module_suffix_counts": dict(sorted(Counter(name.rsplit('.', 1)[-1] for name in target_modules).items())),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "train_binding": {key: value for key, value in train_binding.items() if key != "manifest"},
        "dev_binding": {key: value for key, value in dev_binding.items() if key != "manifest"},
        "corpus_manifest_content_sha256": train_binding["manifest"]["content_sha256"],
        "active_train_rows": len(active_train),
        "active_dev_rows": len(active_dev),
        "pre_train": pre_train,
        "pre_dev": pre_dev,
        "history": history,
        "best_epoch": best_epoch,
        "best_dev_weighted_nll": best_nll,
        "initial_parameter_sha256": initial_sha,
        "output_parameter_sha256": output_sha,
        "parameter_displacement": displacement,
        "placement": placement,
        "gpu_memory": memory,
        "final_adapter": str(final_adapter),
        "final_adapter_sha256": final_adapter_sha,
        "gates": gates,
        "passed": all(gates.values()),
        "elapsed_seconds": time.monotonic() - started,
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(output / "training_report.json", report)
    print(json.dumps({
        "mode": args.mode,
        "passed": report["passed"],
        "optimizer_updates": report["optimizer_updates"],
        "pre_train_weighted_nll": pre_train["weighted_nll"],
        "best_dev_weighted_nll": best_nll,
        "final_adapter": str(final_adapter),
        "report_content_sha256": report["content_sha256"],
    }, indent=2, sort_keys=True))
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
