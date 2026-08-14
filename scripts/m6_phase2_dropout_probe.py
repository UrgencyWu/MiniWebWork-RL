#!/usr/bin/env python3
"""Measure training-forward gradient noise with RL dropout enabled versus disabled."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from miniwebwork.webshop_rl.verifier_td import BASELINE_METHOD  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def _active_adapter_name(model: Any) -> str:
    active = getattr(model, "active_adapters", None)
    active = active() if callable(active) else active
    if isinstance(active, (list, tuple)):
        _require(len(active) == 1, "M6 Phase2 probe requires one active adapter")
        return str(active[0])
    value = getattr(model, "active_adapter", None)
    value = value() if callable(value) else value
    _require(isinstance(value, str) and value, "M6 Phase2 active adapter identity drift")
    return value


def _coefficient_of_variation(values: Sequence[float]) -> float:
    _require(bool(values), "coefficient of variation requires values")
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _gradient_cosine(left: Any, right: Any, torch: Any, *, chunk_size: int = 1_000_000) -> float:
    """Accumulate a large-vector cosine in float64 and enforce its invariant."""

    _require(left.shape == right.shape and left.ndim == 1, "M6 Phase2 gradient vector shape drift")
    numerator = left_square = right_square = 0.0
    for start in range(0, int(left.numel()), chunk_size):
        left_chunk = left[start : start + chunk_size].double()
        right_chunk = right[start : start + chunk_size].double()
        numerator += float(torch.dot(left_chunk, right_chunk))
        left_square += float(torch.dot(left_chunk, left_chunk))
        right_square += float(torch.dot(right_chunk, right_chunk))
    _require(left_square > 0.0 and right_square > 0.0, "M6 Phase2 zero gradient cannot define cosine")
    value = numerator / math.sqrt(left_square * right_square)
    _require(math.isfinite(value), "M6 Phase2 gradient cosine is non-finite")
    _require(-1.0 - 1e-12 <= value <= 1.0 + 1e-12, "M6 Phase2 gradient cosine left [-1, 1]")
    return min(1.0, max(-1.0, value))


def _arm_summary(rows: Sequence[Mapping[str, Any]], vectors: Sequence[Any], torch: Any) -> dict[str, Any]:
    _require(len(rows) == len(vectors) >= 2, "M6 Phase2 dropout arm is incomplete")
    pairwise = [
        _gradient_cosine(vectors[left], vectors[right], torch)
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    norms = [float(row["gradient_norm"]) for row in rows]
    return {
        "repeat_count": len(rows),
        "rows": list(rows),
        "mean_pairwise_gradient_cosine": statistics.fmean(pairwise),
        "minimum_pairwise_gradient_cosine": min(pairwise),
        "gradient_angular_dispersion": 1.0 - statistics.fmean(pairwise),
        "gradient_norm_coefficient_of_variation": _coefficient_of_variation(norms),
        "maximum_training_clip_fraction": max(float(row["clip_fraction"]) for row in rows),
        "maximum_training_absolute_log_ratio": max(float(row["maximum_absolute_log_ratio"]) for row in rows),
        "maximum_reference_kl": max(float(row["mean_token_reference_kl"]) for row in rows),
        "all_finite": all(bool(row["finite"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--repeat-seeds",
        type=int,
        nargs="+",
        default=(20260821, 20260822, 20260823, 20260824, 20260825, 20260826, 20260827, 20260828),
    )
    args = parser.parse_args()

    import torch
    from miniwebwork.long_horizon_rl.learner import collate_turn_training_examples, load_trainable_policy_model

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M6 Phase2 dropout probe requires one GPU")
    _require(len(args.repeat_seeds) >= 8 and len(set(args.repeat_seeds)) == len(args.repeat_seeds), "M6 Phase2 requires at least eight unique repeat seeds")
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "M6 Phase2 dropout report already exists")
    group = validate_committed_group(json.loads(args.group.read_text(encoding="utf-8")), require_k=8)
    adapter = args.sft_adapter.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    adapter_sha256 = directory_sha256(adapter)
    _require(adapter_sha256 == group["adapter_sha256"], "M6 Phase2 group/SFT adapter drift")
    prepared = prepare_group_training_examples(group, method=BASELINE_METHOD)
    device = torch.device("cuda:0")

    def run_arm(*, dropout_enabled: bool) -> dict[str, Any]:
        torch.manual_seed(args.repeat_seeds[0])
        torch.cuda.manual_seed_all(args.repeat_seeds[0])
        model, tokenizer = load_trainable_policy_model(base_model=base_model, adapter_path=adapter)
        current = _active_adapter_name(model)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        _require(parameters, "M6 Phase2 dropout probe found no trainable policy parameters")
        model.load_adapter(str(adapter), adapter_name=REFERENCE_ADAPTER_NAME, is_trainable=False)
        for name, parameter in model.named_parameters():
            if REFERENCE_ADAPTER_NAME in name:
                parameter.requires_grad_(False)
        dropout_modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
        _require(dropout_modules, "M6 Phase2 probe found no dropout modules")

        rows: list[dict[str, Any]] = []
        vectors: list[Any] = []
        for seed in args.repeat_seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True)
            loss_sum = 0.0
            policy_loss_sum = 0.0
            kl_sum = 0.0
            token_count = 0
            clip_count_weighted = 0.0
            maximum_log_ratio = 0.0
            for chunk in _chunks(_optimizer_examples(prepared), args.microbatch_size):
                batch = _move_batch(
                    collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id),
                    device,
                )
                model.set_adapter(REFERENCE_ADAPTER_NAME)
                model.eval()
                with torch.inference_mode():
                    reference = _forward_logprobs(model, batch)
                model.set_adapter(current)
                model.train()
                if not dropout_enabled:
                    for module in dropout_modules:
                        module.eval()
                replay = _forward_logprobs(model, batch)
                result = strict_grpo_kl_loss(
                    replay,
                    reference,
                    batch,
                    clip_epsilon=0.2,
                    kl_coefficient=0.03,
                )
                result["loss"].backward()
                count = int(result["token_count"])
                loss_sum += float(result["loss"].detach().cpu())
                policy_loss_sum += float(result["policy_loss"].detach().cpu())
                kl_sum += float(result["mean_token_reference_kl"].detach().cpu()) * count
                clip_count_weighted += float(result["clip_fraction"].detach().cpu()) * count
                maximum_log_ratio = max(maximum_log_ratio, float(result["maximum_absolute_log_ratio"].detach().cpu()))
                token_count += count
            vector = torch.cat([
                (
                    parameter.grad.detach().float().cpu().reshape(-1)
                    if parameter.grad is not None
                    else torch.zeros_like(parameter.detach(), dtype=torch.float32, device="cpu").reshape(-1)
                )
                for parameter in parameters
            ])
            norm = float(torch.linalg.vector_norm(vector))
            values = (loss_sum, policy_loss_sum, kl_sum / token_count, clip_count_weighted / token_count, maximum_log_ratio, norm)
            rows.append({
                "seed": seed,
                "loss": loss_sum,
                "policy_loss": policy_loss_sum,
                "mean_token_reference_kl": kl_sum / token_count,
                "clip_fraction": clip_count_weighted / token_count,
                "maximum_absolute_log_ratio": maximum_log_ratio,
                "gradient_norm": norm,
                "token_count": token_count,
                "finite": all(math.isfinite(value) for value in values),
            })
            vectors.append(vector)
        summary = _arm_summary(rows, vectors, torch)
        summary["dropout_enabled"] = dropout_enabled
        summary["dropout_module_count"] = len(dropout_modules)
        del vectors, model, tokenizer, parameters
        torch.cuda.empty_cache()
        return summary

    dropout_on = run_arm(dropout_enabled=True)
    dropout_off = run_arm(dropout_enabled=False)
    variance_reduction = (
        float(dropout_off["gradient_angular_dispersion"]) < float(dropout_on["gradient_angular_dispersion"])
        or float(dropout_off["gradient_norm_coefficient_of_variation"])
        < float(dropout_on["gradient_norm_coefficient_of_variation"])
    )
    safety = {
        "maximum_training_clip_fraction": 0.10,
        "maximum_reference_kl": 0.01,
        "dropout_off_all_finite": bool(dropout_off["all_finite"]),
        "dropout_off_clip_passed": float(dropout_off["maximum_training_clip_fraction"]) <= 0.10,
        "dropout_off_kl_passed": float(dropout_off["maximum_reference_kl"]) <= 0.01,
        "variance_reduction_observed": variance_reduction,
    }
    safety["dropout_off_recommended"] = all(
        safety[key]
        for key in (
            "dropout_off_all_finite",
            "dropout_off_clip_passed",
            "dropout_off_kl_passed",
            "variance_reduction_observed",
        )
    )
    report = {
        "schema_version": "m6_phase2_dropout_probe_v2",
        "development_only": True,
        "formal_checkpoint_reusable": False,
        "optimizer_steps": 0,
        "source_group": str(args.group.expanduser().resolve()),
        "source_group_content_sha256": group["content_sha256"],
        "sft_adapter": str(adapter),
        "sft_adapter_sha256": adapter_sha256,
        "repeat_seeds": list(args.repeat_seeds),
        "microbatch_size": args.microbatch_size,
        "dropout_on": dropout_on,
        "dropout_off": dropout_off,
        "decision": safety,
        "interpretation_guardrail": "Dropout-off may repair training/sampling parity; this probe alone cannot attribute historical performance loss to dropout.",
    }
    report["content_sha256"] = _self_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
