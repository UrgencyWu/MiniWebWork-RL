#!/usr/bin/env python3
"""Train Raw35 with the successful M6 4B SFT update paradigm.

This is intentionally a training-paradigm ablation, not a one-variable
optimizer probe.  It preserves the Phase10-C capability-weighted strict Raw35
corpus while adopting the 4B student's effective optimizer batch, Raw-policy
retention, LoRA rank/alpha/dropout, learning rate, and one-epoch schedule.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import m6_phase10c_train_weighted_teacher_lora as weighted  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.phase10_corrective import (  # noqa: E402
    parameter_displacement,
    parameter_snapshot,
    parameter_tensor_sha256,
)

BASE_MODEL = weighted.BASE_MODEL
CAPABILITY_WEIGHTS = weighted.CAPABILITY_WEIGHTS
SEED = 20260866
LEARNING_RATE = 2e-5
RAW_REFERENCE_KL = 0.03
IMITATION_PER_UPDATE = 9
RETENTION_PER_UPDATE = 1
MAXIMUM_EPOCHS = 1
LORA_CONFIG = {"r": 16, "alpha": 32, "dropout": 0.05}
SHARED_EXPERT_SUFFIXES = ("gate_proj", "up_proj", "down_proj")
PROBE_CAPABILITY_ROWS = {"nav": 4, "match": 7, "finish": 7}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def resolve_4b_paradigm_targets(model: Any) -> tuple[str, ...]:
    """Cover text token mixers plus the main shared-expert MLP.

    Qwen3.5-35B-A3B is MoE.  Applying the 4B suffix list globally would add
    LoRA to every routed expert and materially change memory/optimization.
    The closest bounded analogue is the always-active shared expert; routed
    experts, router, vision tower, and MTP remain frozen.
    """

    token_mixers = {
        name
        for name in weighted.resolve_text_target_modules(model)
        if ".mtp." not in name and not name.startswith("mtp.")
    }
    shared_mlp = set()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if (
            ".layers." in name
            and ".mlp.shared_expert." in name
            and "visual" not in name
            and ".mtp." not in name
            and not name.startswith("mtp.")
            and suffix in SHARED_EXPERT_SUFFIXES
            and isinstance(module, torch.nn.Linear)
        ):
            shared_mlp.add(name)
    for suffix in SHARED_EXPERT_SUFFIXES:
        _require(any(name.endswith(f".{suffix}") for name in shared_mlp), f"Raw35 lacks shared-expert {suffix}")
    targets = tuple(sorted(token_mixers | shared_mlp))
    _require(len(targets) == len(token_mixers) + len(shared_mlp), "4B-paradigm target overlap")
    return targets


def build_update_schedule(imitation_count: int, retention_count: int, *, seed: int) -> tuple[dict[str, Any], ...]:
    """Use one epoch with exact 9-imitation/1-Raw-retention updates."""

    _require(imitation_count >= IMITATION_PER_UPDATE and retention_count > 0, "4B-paradigm schedule lacks rows")
    imitation = list(range(imitation_count))
    retention = list(range(retention_count))
    random.Random(seed).shuffle(imitation)
    random.Random(seed + 1).shuffle(retention)
    updates = imitation_count // IMITATION_PER_UPDATE
    schedule = tuple(
        {
            "update_index": update,
            "imitation_indices": imitation[update * IMITATION_PER_UPDATE : (update + 1) * IMITATION_PER_UPDATE],
            "retention_index": retention[update % retention_count],
        }
        for update in range(updates)
    )
    used = [index for item in schedule for index in item["imitation_indices"]]
    _require(len(used) == updates * IMITATION_PER_UPDATE and len(set(used)) == len(used), "schedule duplicated rows")
    _require(imitation_count - len(used) < IMITATION_PER_UPDATE, "schedule discarded a full optimizer batch")
    return schedule


def select_probe_examples(
    examples: Sequence[weighted.WeightedExample], *, seed: int = SEED
) -> list[weighted.WeightedExample]:
    """Select 18 deterministic rows with exact 20/40/40 objective mass."""

    selected = []
    for capability, quota in PROBE_CAPABILITY_ROWS.items():
        candidates = sorted(
            (item for item in examples if item.capability == capability),
            key=lambda item: sha256_json({"seed": seed, "sample": item.tokenized.sample_id}),
        )
        _require(len(candidates) >= quota, f"probe lacks {capability} rows")
        selected.extend(replace(item, row_weight=CAPABILITY_WEIGHTS[capability] / quota) for item in candidates[:quota])
    _require(len(selected) == 18, "probe row count drift")
    return selected


def renormalize_used_weights(
    examples: Sequence[weighted.WeightedExample], schedule: Sequence[Mapping[str, Any]]
) -> list[weighted.WeightedExample]:
    """Preserve hierarchy within each capability after dropping <=8 tail rows."""

    used = {int(index) for update in schedule for index in update["imitation_indices"]}
    mass = {
        capability: sum(item.row_weight for index, item in enumerate(examples) if index in used and item.capability == capability)
        for capability in CAPABILITY_WEIGHTS
    }
    _require(all(value > 0 for value in mass.values()), "used epoch lost one capability")
    result = [
        replace(item, row_weight=item.row_weight * CAPABILITY_WEIGHTS[item.capability] / mass[item.capability])
        for item in examples
    ]
    for capability, expected in CAPABILITY_WEIGHTS.items():
        observed = sum(result[index].row_weight for index in used if result[index].capability == capability)
        _require(math.isclose(observed, expected, abs_tol=1e-12), f"{capability} epoch mass drift")
    return result


def sampled_raw_action_kl(current: torch.Tensor, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Non-negative k3 estimate of KL(Raw || current) on Raw action tokens."""

    _require(current.shape == raw.shape == mask.shape, "retention KL shape drift")
    log_ratio = current - raw
    selected = (torch.exp(log_ratio) - log_ratio - 1.0)[mask]
    _require(selected.numel() > 0, "retention KL has no tokens")
    value = selected.mean()
    _require(bool(torch.isfinite(value)), "retention KL is non-finite")
    return value


def _action_logprobs(
    model: Any, collator: weighted.CompletionOnlyCollator, example: weighted.WeightedExample
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = collator([example.tokenized])
    device = weighted._input_device(model)
    labels = batch["labels"].to(device)
    output = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        position_ids=batch["position_ids"].to(device),
        use_cache=False,
        logits_to_keep=int(batch["logits_to_keep"]),
    )
    logits = output.logits if hasattr(output, "logits") else output[0]
    tail_labels = labels[:, -logits.shape[1] :]
    shifted_labels = tail_labels[:, 1:]
    shifted_logits = logits[:, :-1, :].float()
    mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~mask, 0).to(shifted_logits.device)
    logprobs = F.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return logprobs, mask.to(logprobs.device)


def load_raw35_4b_paradigm_lora(base_model: Path) -> tuple[Any, tuple[str, ...], dict[str, Any]]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 2, "4B-paradigm Raw35 requires exactly two GPUs")
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
    targets = resolve_4b_paradigm_targets(base)
    model = get_peft_model(
        base,
        LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["alpha"],
            lora_dropout=LORA_CONFIG["dropout"],
            target_modules=list(targets),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, targets, {
        "max_memory_bytes": max_memory,
        "parameter_placement": dict(Counter(str(parameter.device) for parameter in model.parameters())),
    }


def _run_update(
    model: Any,
    collator: weighted.CompletionOnlyCollator,
    optimizer: torch.optim.Optimizer,
    imitation: Sequence[weighted.WeightedExample],
    retention: weighted.WeightedExample,
    *,
    epoch_update_count: int,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    ce_objective = 0.0
    by_capability = defaultdict(float)
    label_tokens = 0
    for example in imitation:
        mean_ce = weighted._forward_loss(model, collator, example)
        contribution = mean_ce * example.row_weight * epoch_update_count
        contribution.backward()
        value = float(contribution.detach().cpu())
        ce_objective += value
        by_capability[example.capability] += value
        label_tokens += example.tokenized.completion_label_tokens

    model.eval()
    with torch.inference_mode(), model.disable_adapter():
        raw_logprobs, raw_mask = _action_logprobs(model, collator, retention)
    model.train()
    current_logprobs, current_mask = _action_logprobs(model, collator, retention)
    _require(torch.equal(raw_mask, current_mask), "Raw/current retention token mask drift")
    retention_kl = sampled_raw_action_kl(current_logprobs, raw_logprobs, current_mask)
    (RAW_REFERENCE_KL * retention_kl).backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    _require(bool(torch.isfinite(gradient_norm)) and float(gradient_norm) > 0, "gradient is zero/non-finite")
    optimizer.step()
    return {
        "imitation_weighted_nll": ce_objective,
        "capability_weighted_contribution": dict(sorted(by_capability.items())),
        "retention_kl": float(retention_kl.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "imitation_rows": len(imitation),
        "imitation_label_tokens": label_tokens,
        "retention_action_tokens": int(current_mask.sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(args.base_model.expanduser().resolve() == BASE_MODEL, "base model path drift")
    output = args.output_dir.expanduser().resolve()
    _require(not (output / "training_report.json").exists(), "training report already exists")
    output.mkdir(parents=True, exist_ok=True)
    weighted._set_seed(SEED)

    train_rows, train_binding = weighted.load_and_validate_rows(args.corpus_dir, "train")
    dev_rows, dev_binding = weighted.load_and_validate_rows(args.corpus_dir, "dev")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    all_train = [weighted.tokenize_weighted_row(row, tokenizer) for row in train_rows]
    all_dev = [weighted.tokenize_weighted_row(row, tokenizer) for row in dev_rows]
    if args.mode == "probe":
        active_train = select_probe_examples(all_train)
        active_dev = weighted.select_probe_examples(all_dev, seed=SEED + 1)
    else:
        active_train = all_train
        active_dev = all_dev
    schedule = build_update_schedule(len(active_train), len(active_train), seed=SEED)
    active_train = renormalize_used_weights(active_train, schedule)
    used_indices = [index for item in schedule for index in item["imitation_indices"]]

    started = time.monotonic()
    model, targets, placement = load_raw35_4b_paradigm_lora(BASE_MODEL)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(trainable, "LoRA has no trainable parameters")
    collator = weighted.CompletionOnlyCollator(tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    initial_sha = parameter_tensor_sha256(model)
    initial_parameters = parameter_snapshot(model)
    pre_dev = weighted.evaluate_weighted(model, active_dev, collator)
    history = []
    for item in schedule:
        imitation = [active_train[index] for index in item["imitation_indices"]]
        retention = active_train[item["retention_index"]]
        update = _run_update(
            model,
            collator,
            optimizer,
            imitation,
            retention,
            epoch_update_count=len(schedule),
        )
        history.append({"update_index": item["update_index"], **update})
    post_dev = weighted.evaluate_weighted(model, active_dev, collator)
    final_adapter = output / "final_adapter"
    final_adapter_sha = weighted._save_adapter(model, tokenizer, final_adapter)
    output_sha = parameter_tensor_sha256(model)
    displacement = parameter_displacement(initial_parameters, model)
    memory = weighted._gpu_memory()
    capability_mass = {
        capability: sum(active_train[index].row_weight for index in used_indices if active_train[index].capability == capability)
        for capability in CAPABILITY_WEIGHTS
    }
    gates = {
        "exact_9_to_1_batch_contract": all(item["imitation_rows"] == 9 for item in history),
        "one_epoch_without_duplicate_imitation": len(used_indices) == len(set(used_indices)),
        "exact_capability_loss_mass": all(
            math.isclose(capability_mass[key], value, abs_tol=1e-12) for key, value in CAPABILITY_WEIGHTS.items()
        ),
        "finite_nonzero_gradients": all(math.isfinite(item["gradient_norm"]) and item["gradient_norm"] > 0 for item in history),
        "finite_nonnegative_retention_kl": all(math.isfinite(item["retention_kl"]) and item["retention_kl"] >= 0 for item in history),
        "real_parameter_update": initial_sha != output_sha and displacement["changed_tensor_count"] > 0,
        "dev_nll_safe": post_dev["weighted_nll"] <= pre_dev["weighted_nll"] + 0.10,
        "gpu_headroom_safe": min(item["reserved_headroom_fraction"] for item in memory) >= 0.05,
        "two_gpu_model_parallel": len({str(parameter.device) for parameter in model.parameters()}) >= 2,
    }
    report = {
        "schema_version": "m6_phase10c_raw35_4b_sft_paradigm_v1",
        "complete": True,
        "development_only": True,
        "mode": args.mode,
        "formal_checkpoint_reusable": args.mode == "train",
        "base_model": str(BASE_MODEL),
        "seed": SEED,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "raw_reference_kl": RAW_REFERENCE_KL,
        "raw_reference_kl_direction": "KL(raw_policy||sft_policy)_sampled_on_raw_actions",
        "retention_source": "deterministic Raw35 strict action rows from the frozen train split; KL-only, no CE labels",
        "lora": LORA_CONFIG,
        "optimizer_batch_contract": {"imitation_slots": 9, "retention_slots": 1, "retention_supervised_labels": 0},
        "optimizer_updates": len(schedule),
        "imitation_rows_available": len(active_train),
        "imitation_rows_used": len(used_indices),
        "imitation_rows_unused": len(active_train) - len(used_indices),
        "capability_weights": CAPABILITY_WEIGHTS,
        "observed_epoch_capability_mass": capability_mass,
        "target_policy": "text_token_mixers_plus_always_active_shared_expert_mlp; routed_experts_router_vision_mtp_frozen",
        "target_module_count": len(targets),
        "target_module_suffix_counts": dict(sorted(Counter(name.rsplit(".", 1)[-1] for name in targets).items())),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "corpus_manifest_content_sha256": train_binding["manifest"]["content_sha256"],
        "train_binding": {key: value for key, value in train_binding.items() if key != "manifest"},
        "dev_binding": {key: value for key, value in dev_binding.items() if key != "manifest"},
        "pre_dev": pre_dev,
        "post_dev": post_dev,
        "history": history,
        "mean_retention_kl": sum(item["retention_kl"] for item in history) / len(history),
        "mean_gradient_norm": sum(item["gradient_norm"] for item in history) / len(history),
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
        "pre_dev_weighted_nll": pre_dev["weighted_nll"],
        "post_dev_weighted_nll": post_dev["weighted_nll"],
        "mean_retention_kl": report["mean_retention_kl"],
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
