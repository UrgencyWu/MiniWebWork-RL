#!/usr/bin/env python3
"""Same-batch gradient comparison and one-step LR probes for M6 phase one."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import (  # noqa: E402
    REFERENCE_ADAPTER_NAME,
    _chunks,
    _forward_logprobs,
    _move_batch,
    _optimizer_examples,
    prepare_group_training_examples,
    strict_grpo_kl_loss,
    validate_committed_group,
)
from miniwebwork.webshop_rl.verifier_td import ANCHOR_METHOD, BASELINE_METHOD  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def _flatten_gradients(parameters: list[Any], torch: Any) -> Any:
    return torch.cat([
        (parameter.grad.detach().float().reshape(-1) if parameter.grad is not None else torch.zeros_like(parameter.detach(), dtype=torch.float32).reshape(-1))
        for parameter in parameters
    ])


def _active_adapter_name(model: Any) -> str:
    active = getattr(model, "active_adapters", None)
    active = active() if callable(active) else active
    if isinstance(active, (list, tuple)):
        _require(len(active) == 1, "M6 phase-one probe requires one active adapter")
        return str(active[0])
    value = getattr(model, "active_adapter", None)
    value = value() if callable(value) else value
    _require(isinstance(value, str) and value, "M6 phase-one active adapter identity drift")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(1e-6, 3e-6, 1e-5))
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    import torch
    from miniwebwork.long_horizon_rl.learner import collate_turn_training_examples, load_trainable_policy_model

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M6 phase-one GPU probe requires one GPU")
    _require(set(args.learning_rates) <= {1e-6, 3e-6, 1e-5}, "M6 phase-one LR grid drift")
    group = validate_committed_group(json.loads(args.group.read_text(encoding="utf-8")), require_k=8)
    base_model = args.base_model.expanduser().resolve()
    adapter = args.sft_adapter.expanduser().resolve()
    _require(directory_sha256(adapter) == group["adapter_sha256"], "M6 phase-one group was not sampled from the frozen SFT adapter")
    device = torch.device("cuda:0")

    def load() -> tuple[Any, Any, list[Any], str]:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, tokenizer = load_trainable_policy_model(base_model=base_model, adapter_path=adapter)
        current = _active_adapter_name(model)
        model.load_adapter(str(adapter), adapter_name=REFERENCE_ADAPTER_NAME, is_trainable=False)
        for name, parameter in model.named_parameters():
            if REFERENCE_ADAPTER_NAME in name:
                parameter.requires_grad_(False)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return model, tokenizer, parameters, current

    def gradient(method: str) -> tuple[Any, float, float]:
        model, tokenizer, parameters, current = load()
        prepared = prepare_group_training_examples(group, method=method)
        model.zero_grad(set_to_none=True)
        loss_value = 0.0
        for chunk in _chunks(_optimizer_examples(prepared), args.microbatch_size):
            batch = _move_batch(collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
            model.set_adapter(REFERENCE_ADAPTER_NAME)
            model.eval()
            with torch.inference_mode():
                reference = _forward_logprobs(model, batch)
            model.set_adapter(current)
            model.train()
            replay = _forward_logprobs(model, batch)
            result = strict_grpo_kl_loss(replay, reference, batch, clip_epsilon=0.2, kl_coefficient=0.03)
            result["loss"].backward()
            loss_value += float(result["loss"].detach().cpu())
        vector = _flatten_gradients(parameters, torch).cpu()
        norm = float(torch.linalg.vector_norm(vector))
        del model, tokenizer, parameters, vector
        torch.cuda.empty_cache()
        return vector, norm, loss_value

    grpo, grpo_norm, grpo_loss = gradient(BASELINE_METHOD)
    anchor, anchor_norm, anchor_loss = gradient(ANCHOR_METHOD)
    cosine = float(torch.nn.functional.cosine_similarity(grpo, anchor, dim=0))
    relative_norm_difference = abs(anchor_norm - grpo_norm) / grpo_norm

    lr_rows = []
    learning_rate_labels = {1e-6: "lr_1e_6", 3e-6: "lr_3e_6", 1e-5: "lr_1e_5"}
    for learning_rate in args.learning_rates:
        model, tokenizer, parameters, current = load()
        before = torch.cat([parameter.detach().float().cpu().reshape(-1) for parameter in parameters])
        prepared = prepare_group_training_examples(group, method=BASELINE_METHOD)
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for chunk in _chunks(_optimizer_examples(prepared), args.microbatch_size):
            batch = _move_batch(collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
            model.set_adapter(REFERENCE_ADAPTER_NAME)
            model.eval()
            with torch.inference_mode():
                reference = _forward_logprobs(model, batch)
            model.set_adapter(current)
            model.train()
            replay = _forward_logprobs(model, batch)
            result = strict_grpo_kl_loss(replay, reference, batch, clip_epsilon=0.2, kl_coefficient=0.03)
            result["loss"].backward()
            loss_value += float(result["loss"].detach().cpu())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu())
        optimizer.step()
        after = torch.cat([parameter.detach().float().cpu().reshape(-1) for parameter in parameters])
        relative_l2 = float(torch.linalg.vector_norm(after - before) / torch.linalg.vector_norm(before))
        fixed_state_kl_sum = 0.0
        fixed_state_tokens = 0
        model.eval()
        with torch.inference_mode():
            for chunk in _chunks(_optimizer_examples(prepared), args.microbatch_size):
                batch = _move_batch(collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
                model.set_adapter(REFERENCE_ADAPTER_NAME)
                reference = _forward_logprobs(model, batch)
                model.set_adapter(current)
                updated = _forward_logprobs(model, batch)
                selected = batch["completion_mask"].bool()
                log_ratio = reference - updated
                k3 = torch.exp(log_ratio) - log_ratio - 1.0
                fixed_state_kl_sum += float(k3[selected].sum().cpu())
                fixed_state_tokens += int(selected.sum().cpu())
        fixed_state_kl = fixed_state_kl_sum / fixed_state_tokens
        label = learning_rate_labels[learning_rate]
        model.set_adapter(current)
        if hasattr(model, "delete_adapter"):
            model.delete_adapter(REFERENCE_ADAPTER_NAME)
        output_adapter = args.output.expanduser().resolve().parent / label / "adapter"
        _require(not output_adapter.exists(), "M6 phase-one LR adapter output already exists")
        output_adapter.parent.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(output_adapter, safe_serialization=True)
        tokenizer.save_pretrained(output_adapter)
        lr_rows.append({
            "label": label,
            "learning_rate": learning_rate,
            "loss": loss_value,
            "gradient_norm_before_clip": grad_norm,
            "adapter_relative_l2_delta": relative_l2,
            "fixed_state_mean_token_reference_kl": fixed_state_kl,
            "fixed_state_token_count": fixed_state_tokens,
            "kl_safety_passed": fixed_state_kl <= 0.01,
            "output_adapter": str(output_adapter),
            "output_adapter_sha256": directory_sha256(output_adapter),
            "formal_checkpoint_reusable": False,
            "finite": all(math.isfinite(value) for value in (loss_value, grad_norm, relative_l2, fixed_state_kl)),
        })
        del model, tokenizer, parameters, optimizer, before, after
        torch.cuda.empty_cache()

    report = {
        "schema_version": "m6_phase1_gpu_probe_v1",
        "development_only": True,
        "formal_training": False,
        "source_group": str(args.group.resolve()),
        "source_group_content_sha256": group["content_sha256"],
        "sft_adapter": str(adapter),
        "sft_adapter_sha256": directory_sha256(adapter),
        "gradient_counterfactual": {
            "grpo_loss": grpo_loss,
            "anchor_loss": anchor_loss,
            "grpo_gradient_norm": grpo_norm,
            "anchor_gradient_norm": anchor_norm,
            "gradient_cosine": cosine,
            "relative_gradient_norm_difference": relative_norm_difference,
            "anchor_redundant": cosine >= 0.98 and relative_norm_difference < 0.10,
        },
        "learning_rate_probe": lr_rows,
        "safety_thresholds": {"maximum_fixed_state_kl": 0.01, "maximum_retention_drop_pp": 1.0},
        "note": "Behavior and retention checks require a separate frozen diagnostic rollout; this probe performs no online sampling.",
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
