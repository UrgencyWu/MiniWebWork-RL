#!/usr/bin/env python3
"""Train Qwen3.5-4B on the frozen D35 corpus used by SFT35(D35).

This completes the 2x2 cross: SFT4(D4), SFT35(D4) and SFT35(D35) exist;
this entrypoint adds SFT4(D35).  The causal variable is the training-data
source policy: the identical 4B-paradigm batch schedule, capability
reweighting, one-epoch budget, learning rate, LoRA coverage and adapter
disabled retention reference as the D35 formal training are preserved;
only the base model changes from Qwen3.5-35B-A3B to Qwen3.5-4B.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
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

RAW35_PRODUCER_MODEL = "/data/share/model/Qwen3.5-35B-A3B"
BASE_MODEL = Path("/data/share/model/Qwen3.5-4B")
SEED = 20260866
LEARNING_RATE = 2e-5
RAW_REFERENCE_KL = 0.03
IMITATION_PER_UPDATE = 9
RETENTION_PER_UPDATE = 1
MAXIMUM_EPOCHS = 1
LORA_CONFIG = {"r": 16, "alpha": 32, "dropout": 0.05}
CAPABILITY_WEIGHTS = weighted.CAPABILITY_WEIGHTS
TEXT_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

FROZEN_D35 = {
    "manifest_sha256": "b60b06a2264160e2202a343b7488eace81c4b709b25cf474ee4b2946cff0f138",
    "train_sha256": "76491b419dbfc27c3d5c48805437f994d8f35045e1155f055e99792ff3caf69a",
    "dev_sha256": "42e49891820def03caaf2c15ceea729fc00230a74abb00ece8dfeb2f61868c49",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def resolve_4b_targets(module_names: Sequence[str]) -> tuple[str, ...]:
    """Apply the frozen SFT4 LoRA suffix list, excluding the MTP tower."""

    targets = sorted(
        name for name in module_names
        if name.rsplit(".", 1)[-1] in TEXT_TARGET_SUFFIXES
        and ".mtp." not in name
        and not name.startswith("mtp.")
    )
    _require(bool(targets), "S4(D35) resolved no LoRA targets")
    return tuple(targets)


def load_d35_rows(corpus_dir: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the frozen D35 corpus binding without importing the Raw35 trainer."""

    _require(split in {"train", "dev"}, "S4(D35) weighted split must be train or dev")
    root = corpus_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    data_path = root / f"{split}.jsonl"
    _require(manifest_path.is_file() and data_path.is_file(), "S4(D35) weighted corpus is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("content_sha256") == _self_hash(manifest), "S4(D35) manifest hash drift")
    _require(manifest.get("schema_version") == "m6_phase10c_weighted_raw35_sft_corpus_v1",
             "S4(D35) manifest schema drift")
    _require(manifest.get("base_model") == RAW35_PRODUCER_MODEL,
             "S4(D35) corpus producer identity drift")
    _require(manifest.get("capability_weights") == CAPABILITY_WEIGHTS, "S4(D35) capability weights drift")
    _require(manifest.get("loss_hierarchy") == "capability_task_path_action_row_token_mean_v1",
             "S4(D35) loss hierarchy drift")
    _require(manifest.get("task_overlap") == 0, "S4(D35) corpus task overlap")
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_count = int(manifest[f"{split}_action_row_count"])
    expected_tasks = int(manifest[f"{split}_task_count"])
    _require(len(rows) == expected_count and len({str(row.get("task_id")) for row in rows}) == expected_tasks,
             f"S4(D35) {split} row/task count drift")
    _require(all(row.get("source") == "raw35_replay_strict_self_exploration" for row in rows),
             "S4(D35) weighted source drift")
    _require(all(row.get("capability") in CAPABILITY_WEIGHTS for row in rows), "S4(D35) capability identity drift")
    _require(all(row.get("loss_hierarchy") == manifest["loss_hierarchy"] for row in rows),
             "S4(D35) row hierarchy drift")
    _require(all(float(row.get("row_loss_weight", 0.0)) > 0 for row in rows), "S4(D35) non-positive row weight")
    _require(math.isclose(sum(float(row["row_loss_weight"]) for row in rows), 1.0, abs_tol=1e-12),
             "S4(D35) row loss mass drift")
    observed = {
        capability: sum(float(row["row_loss_weight"]) for row in rows if row["capability"] == capability)
        for capability in CAPABILITY_WEIGHTS
    }
    _require(all(math.isclose(observed[key], value, abs_tol=1e-12) for key, value in CAPABILITY_WEIGHTS.items()),
             "S4(D35) capability loss mass drift")
    return rows, {
        "manifest": manifest,
        "manifest_file_sha256": sha256_file(manifest_path),
        "jsonl_sha256": sha256_file(data_path),
    }


def load_qwen35_4b_lora(base_model: Path) -> tuple[Any, tuple[str, ...], dict[str, Any]]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1,
             "S4(D35) requires at least one GPU")
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model.expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    base.config.use_cache = False
    targets = resolve_4b_targets(name for name, _module in base.named_modules())
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
    placement = {
        "single_gpu": "cuda:0",
        "parameter_device_sha256": sha256_json(
            sorted({str(parameter.device) for parameter in model.parameters()})
        ),
    }
    return model, targets, placement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(args.base_model.expanduser().resolve() == BASE_MODEL, "S4(D35) base-model path drift")
    output = args.output_dir.expanduser().resolve()
    _require(not (output / "training_report.json").exists(), "S4(D35) training report already exists")
    output.mkdir(parents=True, exist_ok=True)
    weighted._set_seed(SEED)

    train_rows, train_binding = load_d35_rows(args.corpus_dir, "train")
    dev_rows, dev_binding = load_d35_rows(args.corpus_dir, "dev")
    _require(train_binding["manifest_file_sha256"] == FROZEN_D35["manifest_sha256"],
             "S4(D35) frozen manifest SHA drift")
    _require(train_binding["jsonl_sha256"] == FROZEN_D35["train_sha256"]
             and dev_binding["jsonl_sha256"] == FROZEN_D35["dev_sha256"],
             "S4(D35) frozen corpus SHA drift")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    all_train = [weighted.tokenize_weighted_row(row, tokenizer) for row in train_rows]
    all_dev = [weighted.tokenize_weighted_row(row, tokenizer) for row in dev_rows]
    if args.mode == "probe":
        active_train = paradigm.select_probe_examples(all_train)
        active_dev = weighted.select_probe_examples(all_dev, seed=SEED + 1)
    else:
        active_train = all_train
        active_dev = all_dev
    schedule = paradigm.build_update_schedule(len(active_train), len(active_train), seed=SEED)
    active_train = paradigm.renormalize_used_weights(active_train, schedule)
    used_indices = [index for item in schedule for index in item["imitation_indices"]]

    started = time.monotonic()
    model, targets, placement = load_qwen35_4b_lora(BASE_MODEL)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(trainable, "S4(D35) LoRA has no trainable parameters")
    collator = weighted.CompletionOnlyCollator(tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    initial_sha = parameter_tensor_sha256(model)
    initial_parameters = parameter_snapshot(model)
    pre_dev = weighted.evaluate_weighted(model, active_dev, collator)
    history = []
    for item in schedule:
        imitation = [active_train[index] for index in item["imitation_indices"]]
        retention = active_train[item["retention_index"]]
        update = paradigm._run_update(
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
        capability: sum(
            active_train[index].row_weight
            for index in used_indices
            if active_train[index].capability == capability
        )
        for capability in CAPABILITY_WEIGHTS
    }
    gates = {
        "exact_9_to_1_batch_contract": all(item["imitation_rows"] == 9 for item in history),
        "one_epoch_without_duplicate_imitation": len(used_indices) == len(set(used_indices)),
        "exact_capability_loss_mass": all(
            math.isclose(capability_mass[key], value, abs_tol=1e-12)
            for key, value in CAPABILITY_WEIGHTS.items()
        ),
        "finite_nonzero_gradients": all(math.isfinite(item["gradient_norm"]) and item["gradient_norm"] > 0 for item in history),
        "finite_nonnegative_retention_kl": all(math.isfinite(item["retention_kl"]) and item["retention_kl"] >= 0 for item in history),
        "real_parameter_update": initial_sha != output_sha and displacement["changed_tensor_count"] > 0,
        "dev_nll_safe": post_dev["weighted_nll"] <= pre_dev["weighted_nll"] + 0.10,
        "gpu_headroom_safe": min(item["reserved_headroom_fraction"] for item in memory) >= 0.05,
        "single_gpu_placement": len({str(parameter.device) for parameter in model.parameters()}) == 1,
    }
    report = {
        "schema_version": "m6_phase10d_s4_d35_training_v1",
        "complete": True,
        "development_only": True,
        "mode": args.mode,
        "formal_checkpoint_reusable": args.mode == "train",
        "causal_variable": "data_source_with_frozen_4b_paradigm_protocol",
        "base_model": str(BASE_MODEL),
        "corpus_producer_model": RAW35_PRODUCER_MODEL,
        "seed": SEED,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "raw_reference_kl": RAW_REFERENCE_KL,
        "raw_reference_kl_direction": "KL(Raw4||S4_D35)_sampled_on_D35_public_actions",
        "retention_source": "deterministic Raw4 (adapter-disabled base) reference on D35 train rows; KL-only, no CE labels",
        "lora": LORA_CONFIG,
        "optimizer_batch_contract": {"imitation_slots": 9, "retention_slots": 1, "retention_supervised_labels": 0},
        "optimizer_updates": len(schedule),
        "imitation_rows_available": len(active_train),
        "imitation_rows_used": len(used_indices),
        "imitation_rows_unused": len(active_train) - len(used_indices),
        "capability_weights": CAPABILITY_WEIGHTS,
        "observed_epoch_capability_mass": capability_mass,
        "target_policy": "q_proj_k_proj_v_proj_o_proj_gate_proj_up_proj_down_proj",
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
