"""Zero-update full-trajectory versus tail-two-turn gradient counterfactual."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..long_horizon_rl.contracts import directory_sha256
from .m6_online_training import strict_grpo_kl_loss
from .phase4_online import (
    REFERENCE_ADAPTER_NAME,
    _active_adapter_name,
    _chunks,
    _forward_logprobs,
    _move_batch,
    collate_phase4_examples,
    prepare_k4x4_examples,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def probe_tail2_panel(
    *,
    groups: Sequence[Mapping[str, Any]],
    base_model: Path,
    input_adapter: Path,
    input_adapter_semantic_sha256: str,
    reference_sft_adapter: Path,
    seed: int,
    microbatch_size: int = 4,
    kl_coefficient: float = 0.03,
    baseline_window: str = "full",
    candidate_window: str = "tail2",
) -> dict[str, Any]:
    """Compare two gradients on identical parameters and trajectories; never step."""

    import torch
    from ..long_horizon_rl.learner import load_trainable_policy_model

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Phase5 probe requires one GPU")
    _require(microbatch_size in {1, 2, 4, 8}, "Phase5 microbatch size drift")
    _require(baseline_window != candidate_window, "Phase5 probe windows must differ")
    baseline = prepare_k4x4_examples(groups, policy_credit_window=baseline_window)
    candidate = prepare_k4x4_examples(groups, policy_credit_window=candidate_window)
    _require(
        [(row.group_id, row.trajectory_id, row.turn_index) for row in baseline["examples"]]
        == [(row.group_id, row.trajectory_id, row.turn_index) for row in candidate["examples"]],
        "Phase5 arm example identity drift",
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    model, tokenizer = load_trainable_policy_model(base_model=base_model, adapter_path=input_adapter)
    current = _active_adapter_name(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(bool(parameters), "Phase5 probe has no trainable parameters")
    model.load_adapter(str(reference_sft_adapter.expanduser().resolve()), adapter_name=REFERENCE_ADAPTER_NAME, is_trainable=False)
    for name, parameter in model.named_parameters():
        if REFERENCE_ADAPTER_NAME in name:
            parameter.requires_grad_(False)
    dropout_modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    _require(bool(dropout_modules), "Phase5 probe found no dropout modules")

    def calculate(prepared: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any]]:
        model.zero_grad(set_to_none=True)
        metric_tokens = 0
        policy_loss = 0.0
        reference_kl = 0.0
        active_policy_tokens = 0
        for chunk in _chunks(prepared["examples"], microbatch_size):
            batch = _move_batch(collate_phase4_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
            active_policy_tokens += sum(
                example.completion_tokens for example in chunk if example.policy_token_loss_weight > 0.0
            )
            model.set_adapter(REFERENCE_ADAPTER_NAME)
            model.eval()
            with torch.inference_mode():
                reference = _forward_logprobs(model, batch)
            model.set_adapter(current)
            model.train()
            for module in dropout_modules:
                module.eval()
            replay = _forward_logprobs(model, batch)
            result = strict_grpo_kl_loss(
                replay,
                reference,
                batch,
                clip_epsilon=0.2,
                kl_coefficient=kl_coefficient,
            )
            result["loss"].backward()
            count = int(result["token_count"])
            metric_tokens += count
            policy_loss += float(result["policy_loss"].detach().cpu())
            reference_kl += float(result["mean_token_reference_kl"].detach().cpu()) * count
        gradients = [
            torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            if parameter.grad is None
            else parameter.grad.detach().float().cpu().clone()
            for parameter in parameters
        ]
        norm_sq = sum(float((gradient.double() * gradient.double()).sum()) for gradient in gradients)
        norm = math.sqrt(norm_sq)
        metrics = {
            "policy_credit_window": prepared["policy_credit_window"],
            "policy_loss": policy_loss,
            "mean_reference_kl": reference_kl / metric_tokens,
            "gradient_norm": norm,
            "gradient_finite": math.isfinite(norm) and all(bool(torch.isfinite(value).all()) for value in gradients),
            "completion_tokens": metric_tokens,
            "active_policy_tokens": active_policy_tokens,
            "active_policy_token_fraction": active_policy_tokens / metric_tokens,
            "active_mixed_task_groups": prepared["active_mixed_group_count"],
        }
        return metrics, gradients

    baseline_metrics, baseline_gradients = calculate(baseline)
    candidate_metrics, candidate_gradients = calculate(candidate)
    dot = baseline_sq = candidate_sq = 0.0
    for baseline_gradient, candidate_gradient in zip(baseline_gradients, candidate_gradients):
        baseline64 = baseline_gradient.double()
        candidate64 = candidate_gradient.double()
        dot += float((baseline64 * candidate64).sum())
        baseline_sq += float((baseline64 * baseline64).sum())
        candidate_sq += float((candidate64 * candidate64).sum())
    _require(baseline_sq > 0.0 and candidate_sq > 0.0, "Phase5 probe produced a zero gradient")
    cosine = dot / math.sqrt(baseline_sq * candidate_sq)
    cosine = max(-1.0, min(1.0, cosine))
    ratio = math.sqrt(candidate_sq / baseline_sq)
    _require(math.isfinite(cosine) and math.isfinite(ratio), "Phase5 gradient comparison is non-finite")
    comparison = f"{baseline_window}_vs_{candidate_window}"
    return {
        baseline_window: baseline_metrics,
        candidate_window: candidate_metrics,
        f"{comparison}_gradient_cosine": cosine,
        f"{candidate_window}_to_{baseline_window}_gradient_norm_ratio": ratio,
        "directionally_distinct_below_0_98": cosine < 0.98,
        "input_adapter": str(input_adapter.expanduser().resolve()),
        "input_adapter_sha256": directory_sha256(input_adapter),
        "input_adapter_semantic_sha256": input_adapter_semantic_sha256,
        "reference_sft_adapter": str(reference_sft_adapter.expanduser().resolve()),
        "reference_sft_adapter_sha256": directory_sha256(reference_sft_adapter),
        "source_group_content_sha256": [group["content_sha256"] for group in baseline["groups"]],
        "optimizer_steps": 0,
    }
