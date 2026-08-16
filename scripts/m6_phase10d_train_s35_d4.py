#!/usr/bin/env python3
"""Train Qwen3.5-35B-A3B on the frozen D4 corpus used by SFT4.

The causal variable is model scale.  This entrypoint keeps the D4 train/dev
rows, one-epoch 9-imitation/1-Raw-reference schedule, completion-only labels,
and token-normalized imitation objective.  It retokenizes the same public
prompt/action rows with the 35B tokenizer and uses adapter-disabled Raw35 for
the retention reference; the historical Raw4 retention token IDs are never
reused across model identities.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402

import m6_phase10c_train_4b_paradigm_teacher_lora as paradigm  # noqa: E402
import m6_phase10c_train_weighted_teacher_lora as weighted  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.phase10_corrective import (  # noqa: E402
    parameter_displacement,
    parameter_snapshot,
    parameter_tensor_sha256,
)

BASE_MODEL = Path("/data/share/model/Qwen3.5-35B-A3B")
SEED = 20260866
MAX_SEQUENCE_TOKENS = 8192
LEARNING_RATE = 2e-5
RAW_REFERENCE_KL = 0.03
IMITATION_PER_UPDATE = 9
RETENTION_PER_UPDATE = 1
MAXIMUM_EPOCHS = 1
LORA_CONFIG = {"r": 16, "alpha": 32, "dropout": 0.05}
PROBE_UPDATES = 2

FROZEN_D4 = {
    "train": {
        "filename": "train.jsonl",
        "sha256": "9f8dcd1ffb036a67eb2589bbfc78c3e661b4b5e95bbbcea601ad8e2bc876df1e",
        "row_count": 2209,
        "task_count": 141,
    },
    "dev": {
        "filename": "dev.jsonl",
        "sha256": "1de4a6f23c4f02dc22c235e210cfb3bec6e86e3ce36b9ebc9d20eaef17fa067a",
        "row_count": 232,
        "task_count": 15,
    },
    "retention_sha256": "aa171c543543682299dd513398d94a5cd053ce678de387d0a543b188a9a8ef84",
    "corpus_audit_sha256": "c83c7846c0db70d9a8e69f053d3b74b3988959dd01bf4382326c69ebdd0ac12b",
    "token_audit_sha256": "11b17eab17f554a76b750693c627dfa2cebf491acff7d04bfc9eb2efebaf3a00",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


@dataclass(frozen=True)
class S35D4Config:
    seed: int
    maximum_sequence_tokens: int
    learning_rate: float
    raw_reference_kl: float
    effective_batch_size: int
    imitation_per_update: int
    retention_per_update: int
    maximum_epochs: int
    lora: dict[str, Any]
    chat_template_kwargs: dict[str, Any]


def s35_d4_config() -> S35D4Config:
    return S35D4Config(
        seed=SEED,
        maximum_sequence_tokens=MAX_SEQUENCE_TOKENS,
        learning_rate=LEARNING_RATE,
        raw_reference_kl=RAW_REFERENCE_KL,
        effective_batch_size=10,
        imitation_per_update=IMITATION_PER_UPDATE,
        retention_per_update=RETENTION_PER_UPDATE,
        maximum_epochs=MAXIMUM_EPOCHS,
        lora=dict(LORA_CONFIG),
        chat_template_kwargs={"enable_thinking": False},
    )


def tokenize_d4_row(row: Mapping[str, Any], tokenizer: Any, config: S35D4Config) -> weighted.TokenizedSFTExample:
    messages = row.get("messages")
    completion = str(row.get("completion") or "")
    _require(isinstance(messages, list) and messages and completion, "D4 row lacks prompt/completion")
    command_payload = json.loads(completion)
    _require(isinstance(command_payload, dict) and set(command_payload) == {"command"}, "D4 completion schema drift")
    _require(isinstance(command_payload["command"], str) and command_payload["command"], "D4 command is empty")
    prompt_ids = weighted._chat_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **config.chat_template_kwargs,
        ),
        label="S35(D4) prompt",
    )
    full_ids = weighted._chat_ids(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": completion}],
            tokenize=True,
            add_generation_prompt=False,
            **config.chat_template_kwargs,
        ),
        label="S35(D4) full row",
    )
    _require(full_ids[: len(prompt_ids)] == prompt_ids, "D4 completion is not a strict 35B continuation")
    _require(len(full_ids) <= config.maximum_sequence_tokens, "D4 row exceeds the frozen no-truncation limit")
    labels = tuple([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :])
    label_tokens = sum(value != -100 for value in labels)
    _require(label_tokens > 0, "D4 row has zero 35B labels")
    task_id = str(row.get("task_id") or "")
    trajectory_id = str(row.get("trajectory_id") or "")
    turn_index = int(row.get("turn_index", -1))
    _require(task_id and trajectory_id and turn_index >= 0, "D4 row provenance is incomplete")
    return weighted.TokenizedSFTExample(
        sample_id=f"{task_id}:{trajectory_id}:{turn_index:03d}",
        task_id=task_id,
        task_family="webshop",
        horizon_stratum="recovery" if row.get("trajectory_recovery") else "direct",
        input_ids=tuple(full_ids),
        labels=labels,
        assistant_content=completion,
        prompt_tokens=len(prompt_ids),
        completion_label_tokens=label_tokens,
        untruncated_forward_tokens=len(full_ids),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_d4_corpus(corpus_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    root = corpus_dir.expanduser().resolve()
    required = {
        "train": root / FROZEN_D4["train"]["filename"],
        "dev": root / FROZEN_D4["dev"]["filename"],
        "retention": root / "retention.json",
        "corpus_audit": root / "corpus_audit.json",
        "token_audit": root / "token_audit.json",
    }
    _require(all(path.is_file() for path in required.values()), "frozen D4 corpus is incomplete")
    expected_hashes = {
        "train": FROZEN_D4["train"]["sha256"],
        "dev": FROZEN_D4["dev"]["sha256"],
        "retention": FROZEN_D4["retention_sha256"],
        "corpus_audit": FROZEN_D4["corpus_audit_sha256"],
        "token_audit": FROZEN_D4["token_audit_sha256"],
    }
    observed_hashes = {name: sha256_file(path) for name, path in required.items()}
    _require(observed_hashes == expected_hashes, "frozen D4 file hash drift")
    rows = {split: _read_jsonl(required[split]) for split in ("train", "dev")}
    required_fields = {
        "command", "completion", "messages", "prompt_sha256", "task_id",
        "trajectory_id", "turn_index", "trajectory_command_sequence_sha256",
    }
    task_sets: dict[str, set[str]] = {}
    for split in ("train", "dev"):
        expected = FROZEN_D4[split]
        task_ids = {str(row.get("task_id") or "") for row in rows[split]}
        sample_ids = {
            (str(row.get("trajectory_id") or ""), int(row.get("turn_index", -1)))
            for row in rows[split]
        }
        _require(len(rows[split]) == expected["row_count"], f"D4 {split} row-count drift")
        _require(len(task_ids) == expected["task_count"] and "" not in task_ids, f"D4 {split} task-count drift")
        _require(len(sample_ids) == len(rows[split]), f"D4 {split} duplicate trajectory turn")
        _require(all(required_fields <= set(row) for row in rows[split]), f"D4 {split} row schema drift")
        task_sets[split] = task_ids
    _require(task_sets["train"].isdisjoint(task_sets["dev"]), "D4 train/dev task overlap")
    return rows, {
        "root": str(root),
        "file_sha256": observed_hashes,
        "train_row_count": len(rows["train"]),
        "train_task_count": len(task_sets["train"]),
        "dev_row_count": len(rows["dev"]),
        "dev_task_count": len(task_sets["dev"]),
        "train_dev_task_overlap": 0,
        "historical_raw4_retention_used_for_s35": False,
    }


def build_update_schedule(imitation_count: int, *, seed: int = SEED) -> tuple[dict[str, Any], ...]:
    return paradigm.build_update_schedule(imitation_count, imitation_count, seed=seed)


def active_schedule(formal: Sequence[Mapping[str, Any]], mode: str) -> tuple[dict[str, Any], ...]:
    _require(mode in {"probe", "train"}, "S35(D4) mode drift")
    selected = tuple(formal[:PROBE_UPDATES]) if mode == "probe" else tuple(formal)
    _require(len(selected) == (PROBE_UPDATES if mode == "probe" else len(formal)), "S35(D4) schedule drift")
    return selected


def select_probe_dev(examples: Sequence[Any], *, count: int = 18, seed: int = SEED + 1) -> tuple[Any, ...]:
    ordered = sorted(examples, key=lambda item: sha256_json({"seed": seed, "sample": item.sample_id}))
    _require(len(ordered) >= count, "D4 dev lacks probe rows")
    selected = tuple(ordered[:count])
    _require(len({item.task_id for item in selected}) >= 2, "D4 probe dev collapsed to one task")
    return selected


def token_weighted_coefficients(examples: Sequence[Any]) -> tuple[float, ...]:
    total = sum(int(item.completion_label_tokens) for item in examples)
    _require(total > 0, "imitation batch has zero label tokens")
    values = tuple(int(item.completion_label_tokens) / total for item in examples)
    _require(math.isclose(sum(values), 1.0, abs_tol=1e-12), "token weights do not sum to one")
    return values


def evaluate_token_weighted(
    model: Any, examples: Sequence[Any], collator: weighted.CompletionOnlyCollator
) -> dict[str, Any]:
    was_training = bool(model.training)
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for example in examples:
            wrapper = weighted.WeightedExample(example, "d4", "", 1.0)
            mean_loss = float(weighted._forward_loss(model, collator, wrapper).detach().cpu())
            loss_sum += mean_loss * example.completion_label_tokens
            token_count += example.completion_label_tokens
    if was_training:
        model.train()
    _require(token_count > 0 and math.isfinite(loss_sum), "D4 dev loss is invalid")
    return {
        "token_mean_nll": loss_sum / token_count,
        "sample_count": len(examples),
        "task_count": len({item.task_id for item in examples}),
        "completion_label_tokens": token_count,
    }


def _action_logprobs(
    model: Any, collator: weighted.CompletionOnlyCollator, example: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    wrapper = weighted.WeightedExample(example, "d4", "", 1.0)
    return paradigm._action_logprobs(model, collator, wrapper)


def run_update(
    model: Any,
    collator: weighted.CompletionOnlyCollator,
    optimizer: torch.optim.Optimizer,
    imitation: Sequence[Any],
    retention: Any,
) -> dict[str, Any]:
    _require(len(imitation) == IMITATION_PER_UPDATE, "S35(D4) imitation slot drift")
    optimizer.zero_grad(set_to_none=True)
    coefficients = token_weighted_coefficients(imitation)
    imitation_nll = 0.0
    for example, coefficient in zip(imitation, coefficients, strict=True):
        wrapper = weighted.WeightedExample(example, "d4", "", 1.0)
        mean_ce = weighted._forward_loss(model, collator, wrapper)
        (mean_ce * coefficient).backward()
        imitation_nll += float(mean_ce.detach().cpu()) * coefficient

    model.eval()
    with torch.inference_mode(), model.disable_adapter():
        raw_logprobs, raw_mask = _action_logprobs(model, collator, retention)
    model.train()
    current_logprobs, current_mask = _action_logprobs(model, collator, retention)
    _require(torch.equal(raw_mask, current_mask), "Raw35/current retention token mask drift")
    retention_kl = paradigm.sampled_raw_action_kl(current_logprobs, raw_logprobs, current_mask)
    (RAW_REFERENCE_KL * retention_kl).backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    _require(bool(torch.isfinite(gradient_norm)) and float(gradient_norm) > 0, "gradient is zero/non-finite")
    optimizer.step()
    return {
        "imitation_token_mean_nll": imitation_nll,
        "imitation_rows": len(imitation),
        "imitation_label_tokens": sum(item.completion_label_tokens for item in imitation),
        "retention_kl": float(retention_kl.detach().cpu()),
        "retention_action_tokens": int(current_mask.sum().item()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(args.base_model.expanduser().resolve() == BASE_MODEL, "S35(D4) base-model path drift")
    output = args.output_dir.expanduser().resolve()
    _require(not (output / "training_report.json").exists(), "S35(D4) training report already exists")
    output.mkdir(parents=True, exist_ok=True)
    weighted._set_seed(SEED)

    rows, corpus_binding = validate_d4_corpus(args.corpus_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = s35_d4_config()
    train = tuple(tokenize_d4_row(row, tokenizer, config) for row in rows["train"])
    dev = tuple(tokenize_d4_row(row, tokenizer, config) for row in rows["dev"])
    formal_schedule = build_update_schedule(len(train), seed=SEED)
    schedule = active_schedule(formal_schedule, args.mode)
    active_dev = select_probe_dev(dev) if args.mode == "probe" else dev
    used_indices = [index for item in schedule for index in item["imitation_indices"]]

    started = time.monotonic()
    model, targets, placement = paradigm.load_raw35_4b_paradigm_lora(BASE_MODEL)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(trainable, "S35(D4) LoRA has no trainable parameters")
    collator = weighted.CompletionOnlyCollator(tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    initial_sha = parameter_tensor_sha256(model)
    initial_parameters = parameter_snapshot(model)
    pre_dev = evaluate_token_weighted(model, active_dev, collator)
    history = []
    for item in schedule:
        imitation = [train[index] for index in item["imitation_indices"]]
        retention = train[int(item["retention_index"])]
        history.append({
            "update_index": int(item["update_index"]),
            **run_update(model, collator, optimizer, imitation, retention),
        })
    post_dev = evaluate_token_weighted(model, active_dev, collator)
    final_adapter = output / "final_adapter"
    final_adapter_sha = weighted._save_adapter(model, tokenizer, final_adapter)
    final_sha = parameter_tensor_sha256(model)
    displacement = parameter_displacement(initial_parameters, model)
    memory = weighted._gpu_memory()
    gates = {
        "frozen_d4_identity": corpus_binding["file_sha256"] == {
            "train": FROZEN_D4["train"]["sha256"],
            "dev": FROZEN_D4["dev"]["sha256"],
            "retention": FROZEN_D4["retention_sha256"],
            "corpus_audit": FROZEN_D4["corpus_audit_sha256"],
            "token_audit": FROZEN_D4["token_audit_sha256"],
        },
        "exact_9_to_1_batch_contract": all(item["imitation_rows"] == 9 for item in history),
        "token_normalized_d4_objective": all(item["imitation_label_tokens"] > 0 for item in history),
        "no_duplicate_imitation_rows": len(used_indices) == len(set(used_indices)),
        "finite_nonzero_gradients": all(math.isfinite(item["gradient_norm"]) and item["gradient_norm"] > 0 for item in history),
        "finite_nonnegative_retention_kl": all(math.isfinite(item["retention_kl"]) and item["retention_kl"] >= 0 for item in history),
        "real_parameter_update": initial_sha != final_sha and displacement["changed_tensor_count"] > 0,
        "dev_nll_safe": post_dev["token_mean_nll"] <= pre_dev["token_mean_nll"] + 0.10,
        "gpu_headroom_safe": min(item["reserved_headroom_fraction"] for item in memory) >= 0.05,
        "two_gpu_model_parallel": len({str(parameter.device) for parameter in model.parameters()}) >= 2,
    }
    report = {
        "schema_version": "m6_phase10d_s35_d4_training_v1",
        "complete": True,
        "development_only": True,
        "mode": args.mode,
        "formal_checkpoint_reusable": args.mode == "train",
        "causal_variable": "model_scale_with_frozen_D4_supervision_and_objective",
        "base_model": str(BASE_MODEL),
        "seed": SEED,
        "maximum_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "raw_reference_kl": RAW_REFERENCE_KL,
        "raw_reference_kl_direction": "KL(Raw35||S35_D4)_sampled_on_D4_public_actions",
        "retention_source": "D4 train public prompt/action rows retokenized for 35B; adapter-disabled Raw35 reference",
        "historical_raw4_retention_token_ids_used": False,
        "lora": LORA_CONFIG,
        "optimizer_batch_contract": {"imitation_slots": 9, "retention_slots": 1, "retention_supervised_labels": 0},
        "imitation_loss": "sum_cross_entropy_over_9_rows/divide_by_all_effective_completion_tokens",
        "capability_reweighting": None,
        "optimizer_updates": len(schedule),
        "formal_optimizer_updates": len(formal_schedule),
        "imitation_rows_available": len(train),
        "imitation_rows_used": len(used_indices),
        "formal_imitation_rows_unused": len(train) - len(formal_schedule) * IMITATION_PER_UPDATE,
        "train_35b_completion_label_tokens": sum(item.completion_label_tokens for item in train),
        "dev_35b_completion_label_tokens": sum(item.completion_label_tokens for item in dev),
        "target_policy": "text_token_mixers_plus_always_active_shared_expert_mlp; routed_experts_router_vision_mtp_frozen",
        "target_module_count": len(targets),
        "target_module_suffix_counts": dict(sorted(Counter(name.rsplit(".", 1)[-1] for name in targets).items())),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "corpus_binding": corpus_binding,
        "pre_dev": pre_dev,
        "post_dev": post_dev,
        "history": history,
        "mean_retention_kl": sum(item["retention_kl"] for item in history) / len(history),
        "mean_gradient_norm": sum(item["gradient_norm"] for item in history) / len(history),
        "initial_parameter_sha256": initial_sha,
        "output_parameter_sha256": final_sha,
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
        "pre_dev_token_mean_nll": pre_dev["token_mean_nll"],
        "post_dev_token_mean_nll": post_dev["token_mean_nll"],
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
