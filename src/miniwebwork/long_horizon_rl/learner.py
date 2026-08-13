"""Shared hierarchical clipped-policy learner for both focused online methods."""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..m4_long_horizon_protocol import GROUP_SIZE, MAX_SEQUENCE_LENGTH, ONLINE_LEARNER_CONFIG
from .adapter_view import build_vllm_adapter_view
from .contracts import (
    RunIdentity,
    directory_sha256,
    sha256_file,
    sha256_json,
    token_ids_sha256,
    validate_committed_group,
)
from .credit import assign_group_credit

LEARNER_BATCH_SCHEMA = "m4_long_horizon_learner_batch_v1"
LEARNER_OPTIMIZER_SCHEMA = "m4_long_horizon_optimizer_state_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TurnTrainingExample:
    group_id: str
    trajectory_id: str
    trajectory_index: int
    turn_index: int
    turns_in_trajectory: int
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    sampling_logprobs: tuple[float, ...]
    advantage: float

    @property
    def completion_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def forward_tokens(self) -> int:
        return len(self.prompt_token_ids) + self.completion_tokens

    @property
    def token_loss_weight(self) -> float:
        """Weight whose total is one across the exact K4 hierarchy."""

        return 1.0 / (GROUP_SIZE * self.turns_in_trajectory * self.completion_tokens)


def prepare_group_training_examples(
    group: Mapping[str, Any],
    method: str,
    *,
    identity: RunIdentity | None = None,
) -> dict[str, Any]:
    """Bind credit output to the exact completion/log-prob evidence."""

    validated = validate_committed_group(group, identity)
    credit = assign_group_credit(validated, method)
    examples: list[TurnTrainingExample] = []
    effective_tokens = 0
    for trajectory_index, (trajectory, trajectory_credit) in enumerate(
        zip(validated["trajectories"], credit["turn_credit"])
    ):
        turn_count = len(trajectory["turns"])
        _require(turn_count > 0, "learner trajectory has no generated turns")
        _require(len(trajectory_credit) == turn_count, "credit/trajectory turn-count mismatch")
        for turn, turn_credit in zip(trajectory["turns"], trajectory_credit):
            prompt_ids = tuple(int(value) for value in turn["prompt_token_ids"])
            generated_ids = tuple(int(value) for value in turn["generated_token_ids"])
            behavior = tuple(float(value) for value in turn["behavior_logprobs"])
            sampling = tuple(float(value) for value in turn["sampling_logprobs"])
            _require(bool(prompt_ids) and bool(generated_ids), "learner turn has empty token evidence")
            _require(
                len(generated_ids) == len(behavior) == len(sampling),
                "learner completion/log-prob length mismatch",
            )
            _require(
                token_ids_sha256(generated_ids) == turn["generated_token_sha256"],
                "learner generated-token hash drift",
            )
            _require(
                len(prompt_ids) + len(generated_ids) <= MAX_SEQUENCE_LENGTH,
                "learner replay exceeds frozen context length",
            )
            advantage = float(turn_credit["turn_advantage"])
            _require(math.isfinite(advantage), "learner advantage is not finite")
            example = TurnTrainingExample(
                group_id=validated["group_id"],
                trajectory_id=trajectory["trajectory_id"],
                trajectory_index=trajectory_index,
                turn_index=turn["turn_index"],
                turns_in_trajectory=turn_count,
                prompt_token_ids=prompt_ids,
                generated_token_ids=generated_ids,
                behavior_logprobs=behavior,
                sampling_logprobs=sampling,
                advantage=advantage,
            )
            examples.append(example)
            if abs(advantage) > 0:
                effective_tokens += example.completion_tokens
    expected_tokens = sum(example.completion_tokens for example in examples)
    _require(
        expected_tokens == validated["generated_action_tokens"],
        "learner group token count drift",
    )
    total_weight = sum(example.token_loss_weight * example.completion_tokens for example in examples)
    _require(math.isclose(total_weight, 1.0, abs_tol=1e-12), "hierarchical weights do not sum to one")
    return {
        "schema_version": LEARNER_BATCH_SCHEMA,
        "group_id": validated["group_id"],
        "method": method,
        "examples": tuple(examples),
        "credit": credit,
        "generated_action_tokens": expected_tokens,
        "effective_optimizer_action_tokens": effective_tokens,
        "zero_advantage_group": effective_tokens == 0,
        "hierarchical_weight_sum": total_weight,
    }


def collate_turn_training_examples(
    examples: Sequence[TurnTrainingExample],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    """Left-pad full replay sequences and align tail-only completion logits."""

    _require(bool(examples), "cannot collate an empty learner microbatch")
    _require(isinstance(pad_token_id, int) and pad_token_id >= 0, "invalid learner pad token")
    maximum_forward = max(example.forward_tokens for example in examples)
    maximum_completion = max(example.completion_tokens for example in examples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    behavior_logprobs: list[list[float]] = []
    completion_mask: list[list[bool]] = []
    generated_ids: list[list[int]] = []
    advantages: list[float] = []
    token_weights: list[float] = []
    for example in examples:
        full = list(example.prompt_token_ids + example.generated_token_ids)
        left_padding = maximum_forward - len(full)
        completion_padding = maximum_completion - example.completion_tokens
        input_ids.append([pad_token_id] * left_padding + full)
        attention_mask.append([0] * left_padding + [1] * len(full))
        behavior_logprobs.append(
            list(example.behavior_logprobs) + [0.0] * completion_padding
        )
        generated_ids.append(
            list(example.generated_token_ids) + [pad_token_id] * completion_padding
        )
        completion_mask.append(
            [True] * example.completion_tokens + [False] * completion_padding
        )
        advantages.append(example.advantage)
        token_weights.append(example.token_loss_weight)
    attention = torch.tensor(attention_mask, dtype=torch.long)
    positions = (attention.cumsum(dim=-1) - 1).clamp_min(0)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": attention,
        "position_ids": positions,
        "generated_token_ids": torch.tensor(generated_ids, dtype=torch.long),
        "behavior_logprobs": torch.tensor(behavior_logprobs, dtype=torch.float32),
        "completion_mask": torch.tensor(completion_mask, dtype=torch.bool),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "token_loss_weights": torch.tensor(token_weights, dtype=torch.float32),
        "completion_lengths": torch.tensor(
            [example.completion_tokens for example in examples], dtype=torch.long
        ),
        "logits_to_keep": maximum_completion + 1,
        "maximum_forward_tokens": maximum_forward,
        "examples": tuple(examples),
    }


def extract_tail_completion_logprobs(
    logits: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    logprob_precision: str = "float32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract chosen-token log-probs from left-padded, tail-only model logits."""

    _require(logprob_precision in {"float32", "model"}, "learner logprob precision drift")
    _require(logits.ndim == 3, f"learner logits must be [batch,seq,vocab], got {tuple(logits.shape)}")
    generated_ids = batch["generated_token_ids"].to(logits.device)
    mask = batch["completion_mask"].to(logits.device)
    lengths = batch["completion_lengths"].tolist()
    maximum_completion = int(generated_ids.shape[1])
    expected_keep = maximum_completion + 1
    _require(logits.shape[0] == generated_ids.shape[0], "learner logits batch drift")
    _require(
        logits.shape[1] in {expected_keep, int(batch["maximum_forward_tokens"])},
        "learner logits sequence dimension is neither tail-only nor full",
    )
    if logits.shape[1] == int(batch["maximum_forward_tokens"]):
        tail = logits[:, -expected_keep:, :]
    else:
        tail = logits
    output = torch.zeros(
        generated_ids.shape,
        dtype=torch.float32,
        device=logits.device,
    )
    entropy = torch.zeros_like(output)
    for row, completion_length in enumerate(lengths):
        _require(completion_length > 0, "learner completion length must be positive")
        start = maximum_completion - completion_length
        token_logits = tail[row, start:maximum_completion, :]
        if logprob_precision == "float32":
            token_logits = token_logits.float()
        token_ids = generated_ids[row, :completion_length]
        _require(
            bool((token_ids >= 0).all()) and bool((token_ids < token_logits.shape[-1]).all()),
            "learner generated token id outside vocabulary",
        )
        log_probs = torch.log_softmax(token_logits, dim=-1)
        chosen = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        output[row, :completion_length] = chosen
        probabilities = log_probs.exp()
        entropy[row, :completion_length] = -(probabilities * log_probs).sum(dim=-1)
    _require(bool(torch.isfinite(output[mask]).all()), "learner replay log-probs are non-finite")
    _require(bool(torch.isfinite(entropy[mask]).all()), "learner entropy is non-finite")
    return output, entropy


def hierarchical_clipped_policy_loss(
    replay_logprobs: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    clip_epsilon: float = ONLINE_LEARNER_CONFIG["clip_epsilon"],
    entropy: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute the exact token→turn→trajectory→K4 normalized PPO objective."""

    _require(0 < clip_epsilon < 1, "invalid learner clip epsilon")
    old = batch["behavior_logprobs"].to(replay_logprobs.device)
    mask = batch["completion_mask"].to(replay_logprobs.device)
    advantages = batch["advantages"].to(replay_logprobs.device).unsqueeze(1)
    weights = batch["token_loss_weights"].to(replay_logprobs.device).unsqueeze(1)
    _require(replay_logprobs.shape == old.shape == mask.shape, "learner loss tensor shape drift")
    log_ratio = replay_logprobs - old
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    token_objective = torch.minimum(unclipped, clipped)
    loss = -(token_objective * weights * mask).sum()
    selected = mask
    token_count = int(selected.sum().item())
    _require(token_count > 0, "learner loss contains no completion tokens")
    clip_fraction = ((ratio != clipped_ratio) & selected).float().sum() / token_count
    approx_kl = (((ratio - 1.0) - log_ratio) * selected).sum() / token_count
    result: dict[str, Any] = {
        "loss": loss,
        "token_count": token_count,
        "mean_ratio": ratio[selected].mean(),
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "maximum_absolute_log_ratio": log_ratio[selected].abs().max(),
    }
    if entropy is not None:
        _require(entropy.shape == replay_logprobs.shape, "learner entropy shape drift")
        result["entropy"] = entropy[selected].mean()
    return result


def summarize_logprob_parity(
    reference: Sequence[float],
    candidate: Sequence[float],
) -> dict[str, float | int]:
    """Return transparent parity statistics without silently choosing a threshold."""

    _require(len(reference) == len(candidate) and bool(reference), "parity vectors must match and be non-empty")
    differences = sorted(abs(float(left) - float(right)) for left, right in zip(reference, candidate))
    _require(all(math.isfinite(value) for value in differences), "parity difference is non-finite")
    count = len(differences)
    p95_index = min(count - 1, max(0, math.ceil(0.95 * count) - 1))
    p99_index = min(count - 1, max(0, math.ceil(0.99 * count) - 1))
    p999_index = min(count - 1, max(0, math.ceil(0.999 * count) - 1))
    log_ratios = [float(right) - float(left) for left, right in zip(reference, candidate)]
    ratios = [math.exp(min(80.0, max(-80.0, value))) for value in log_ratios]
    clip_epsilon = float(ONLINE_LEARNER_CONFIG["clip_epsilon"])
    lower_log_ratio = math.log(1.0 - clip_epsilon)
    upper_log_ratio = math.log(1.0 + clip_epsilon)
    initial_ratio_clip_count = sum(
        value < lower_log_ratio or value > upper_log_ratio for value in log_ratios
    )
    return {
        "token_count": count,
        "mean_absolute_logprob_difference": sum(differences) / count,
        "p95_absolute_logprob_difference": differences[p95_index],
        "p99_absolute_logprob_difference": differences[p99_index],
        "p999_absolute_logprob_difference": differences[p999_index],
        "maximum_absolute_logprob_difference": differences[-1],
        "mean_importance_ratio": sum(ratios) / count,
        "maximum_absolute_log_ratio": max(abs(value) for value in log_ratios),
        "initial_ratio_clip_epsilon": clip_epsilon,
        "initial_ratio_clip_count": initial_ratio_clip_count,
        "initial_ratio_clip_fraction": initial_ratio_clip_count / count,
    }


def audit_group_behavior_sampling_parity(
    group: Mapping[str, Any],
    *,
    maximum_absolute_difference: float,
) -> dict[str, Any]:
    """Verify that stored behavior probabilities match the no-warp sampling distribution."""

    _require(maximum_absolute_difference > 0, "parity threshold must be positive")
    validated = validate_committed_group(group)
    behavior: list[float] = []
    sampling: list[float] = []
    for trajectory in validated["trajectories"]:
        for turn in trajectory["turns"]:
            behavior.extend(float(value) for value in turn["behavior_logprobs"])
            sampling.extend(float(value) for value in turn["sampling_logprobs"])
    report = summarize_logprob_parity(behavior, sampling)
    report.update(
        group_id=validated["group_id"],
        threshold=maximum_absolute_difference,
        passed=(
            report["maximum_absolute_logprob_difference"]
            <= maximum_absolute_difference
        ),
    )
    if not report["passed"]:
        raise ValueError(
            "behavior/sampling log-prob parity failed: "
            f"maximum={report['maximum_absolute_logprob_difference']}, "
            f"threshold={maximum_absolute_difference}"
        )
    return report


def _move_replay_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for field in (
        "input_ids",
        "attention_mask",
        "position_ids",
        "generated_token_ids",
        "behavior_logprobs",
        "completion_mask",
        "advantages",
        "token_loss_weights",
        "completion_lengths",
    ):
        moved[field] = batch[field].to(device, non_blocking=True)
    return moved


def _forward_replay(
    model: Any,
    batch: Mapping[str, Any],
    *,
    logprob_precision: str = "float32",
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
        logits_to_keep=int(batch["logits_to_keep"]),
    )
    logits = output.logits if hasattr(output, "logits") else output[0]
    return extract_tail_completion_logprobs(
        logits,
        batch,
        logprob_precision=logprob_precision,
    )


def _chunked(sequence: Sequence[Any], size: int):
    _require(size > 0, "learner microbatch must be positive")
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def replay_examples(
    *,
    model: Any,
    examples: Sequence[TurnTrainingExample],
    pad_token_id: int,
    microbatch_size: int,
    device: torch.device,
    logprob_precision: str = "float32",
) -> list[float]:
    """Teacher-force exact prompt+completion IDs under the current adapter."""

    values: list[float] = []
    was_training = bool(model.training)
    model.eval()
    with torch.inference_mode():
        for chunk in _chunked(examples, microbatch_size):
            batch = _move_replay_batch(
                collate_turn_training_examples(chunk, pad_token_id=pad_token_id),
                device,
            )
            replay, _ = _forward_replay(
                model,
                batch,
                logprob_precision=logprob_precision,
            )
            for row, example in enumerate(chunk):
                values.extend(
                    float(value)
                    for value in replay[row, : example.completion_tokens].detach().cpu().tolist()
                )
    if was_training:
        model.train()
    return values


def audit_initial_replay_parity(
    *,
    model: Any,
    groups: Sequence[Mapping[str, Any]],
    method: str,
    identity: RunIdentity,
    pad_token_id: int,
    microbatch_size: int,
    device: torch.device,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Gate vLLM behavior evidence against HF replay before the first update."""

    required = {
        "behavior_sampling_maximum_absolute_difference",
        "replay_mean_absolute_difference",
        "replay_p95_absolute_difference",
        "replay_p99_absolute_difference",
        "replay_p999_absolute_difference",
        "replay_initial_ratio_clip_fraction",
        "mean_importance_ratio_absolute_deviation",
    }
    _require(set(thresholds) == required, "replay parity threshold set drift")
    _require(all(float(value) > 0 for value in thresholds.values()), "invalid replay parity threshold")
    reference: list[float] = []
    candidate: list[float] = []
    behavior_sampling_reports = []
    for group in groups:
        behavior_sampling_reports.append(
            audit_group_behavior_sampling_parity(
                group,
                maximum_absolute_difference=float(
                    thresholds["behavior_sampling_maximum_absolute_difference"]
                ),
            )
        )
        prepared = prepare_group_training_examples(group, method, identity=identity)
        examples = prepared["examples"]
        reference.extend(
            value for example in examples for value in example.behavior_logprobs
        )
        candidate.extend(
            replay_examples(
                model=model,
                examples=examples,
                pad_token_id=pad_token_id,
                microbatch_size=microbatch_size,
                device=device,
            )
        )
    report = summarize_logprob_parity(reference, candidate)
    checks = {
        "mean_absolute_difference": (
            report["mean_absolute_logprob_difference"]
            <= thresholds["replay_mean_absolute_difference"]
        ),
        "p95_absolute_difference": (
            report["p95_absolute_logprob_difference"]
            <= thresholds["replay_p95_absolute_difference"]
        ),
        "p99_absolute_difference": (
            report["p99_absolute_logprob_difference"]
            <= thresholds["replay_p99_absolute_difference"]
        ),
        "p999_absolute_difference": (
            report["p999_absolute_logprob_difference"]
            <= thresholds["replay_p999_absolute_difference"]
        ),
        "initial_ratio_clip_fraction": (
            report["initial_ratio_clip_fraction"]
            <= thresholds["replay_initial_ratio_clip_fraction"]
        ),
        "mean_importance_ratio": (
            abs(report["mean_importance_ratio"] - 1.0)
            <= thresholds["mean_importance_ratio_absolute_deviation"]
        ),
    }
    passed = all(checks.values())
    result = {
        **report,
        "thresholds": dict(thresholds),
        "checks": checks,
        "behavior_sampling": behavior_sampling_reports,
        "passed": passed,
    }
    if not passed:
        raise ValueError(f"initial behavior/replay parity failed: {result}")
    return result


def _trainable_parameters(model: Any) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(bool(parameters), "online learner has no trainable adapter parameters")
    return parameters


def _parameter_snapshot(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _parameter_change_norm(model: Any, before: Mapping[str, torch.Tensor]) -> float:
    squared = 0.0
    seen: set[str] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        _require(name in before, f"trainable parameter appeared during learner update: {name}")
        difference = parameter.detach().float().cpu() - before[name]
        squared += float(torch.sum(difference * difference).item())
        seen.add(name)
    _require(seen == set(before), "trainable parameter disappeared during learner update")
    return math.sqrt(squared)


def summarize_credit_assignment(prepared_groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Expose the credit signal and its horizon position without changing loss."""

    _require(bool(prepared_groups), "credit summary requires prepared groups")
    position = {
        name: {"turn_count": 0, "action_tokens": 0, "effective_action_tokens": 0, "absolute_advantage_sum": 0.0}
        for name in ("early", "middle", "late")
    }
    macro_values: list[float] = []
    micro_values: list[float] = []
    turn_values: list[float] = []
    shared_anchors = informative_anchors = informative_micro_turns = 0
    mixed_reward_groups = 0
    unique_anchors: set[str] = set()
    for prepared in prepared_groups:
        credit = prepared["credit"]
        rewards = [float(value) for value in credit["rewards"]]
        mixed_reward_groups += len(set(rewards)) > 1
        macro_values.extend(float(value) for value in credit["macro_advantages"])
        metrics = credit["metrics"]
        shared_anchors += metrics["shared_anchor_count"]
        informative_anchors += metrics["informative_anchor_count"]
        informative_micro_turns += metrics["informative_micro_turn_count"]
        credit_turns = tuple(entry for trajectory in credit["turn_credit"] for entry in trajectory)
        _require(len(credit_turns) == len(prepared["examples"]), "credit summary turn/example drift")
        for example, item in zip(prepared["examples"], credit_turns):
            micro = float(item["micro_advantage"])
            advantage = float(item["turn_advantage"])
            micro_values.append(micro)
            turn_values.append(advantage)
            unique_anchors.add(item["anchor_signature"])
            slot = min(2, ((example.turn_index - 1) * 3) // example.turns_in_trajectory)
            bucket = ("early", "middle", "late")[slot]
            summary = position[bucket]
            summary["turn_count"] += 1
            summary["action_tokens"] += example.completion_tokens
            summary["effective_action_tokens"] += example.completion_tokens if abs(advantage) > 0 else 0
            summary["absolute_advantage_sum"] += abs(advantage)
    for summary in position.values():
        turns = summary["turn_count"]
        summary["mean_absolute_turn_advantage"] = summary.pop("absolute_advantage_sum") / turns if turns else 0.0
    return {
        "group_count": len(prepared_groups),
        "mixed_reward_group_count": mixed_reward_groups,
        "zero_variance_group_count": len(prepared_groups) - mixed_reward_groups,
        "unique_public_anchor_count": len(unique_anchors),
        "shared_anchor_count": shared_anchors,
        "informative_anchor_count": informative_anchors,
        "informative_micro_turn_count": informative_micro_turns,
        "mean_absolute_macro_advantage": sum(abs(value) for value in macro_values) / len(macro_values),
        "mean_absolute_micro_advantage": sum(abs(value) for value in micro_values) / len(micro_values),
        "mean_absolute_turn_advantage": sum(abs(value) for value in turn_values) / len(turn_values),
        "nonzero_turn_advantage_count": sum(abs(value) > 0 for value in turn_values),
        "turn_position": position,
    }


def train_policy_groups(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    groups: Sequence[Mapping[str, Any]],
    method: str,
    identity: RunIdentity,
    pad_token_id: int,
    microbatch_size: int,
    device: torch.device,
    collection_sha256: str,
    all_generated_action_tokens: int,
    parity_thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Run two policy epochs with one exact K4 group per optimizer minibatch."""

    identity.validate()
    _require(bool(groups), "online learner received no committed groups")
    _require(all_generated_action_tokens > 0, "online learner all-token ledger is empty")
    _require(
        isinstance(collection_sha256, str) and len(collection_sha256) == 64,
        "online learner collection hash drift",
    )
    ordered_groups = sorted(groups, key=lambda group: str(group.get("group_id", "")))
    prepared_groups = [
        prepare_group_training_examples(group, method, identity=identity)
        for group in ordered_groups
    ]
    credit_assignment = summarize_credit_assignment(prepared_groups)
    committed_tokens = sum(item["generated_action_tokens"] for item in prepared_groups)
    _require(
        all_generated_action_tokens >= committed_tokens,
        "all-token ledger excludes committed action tokens",
    )
    effective_unique_tokens = sum(
        item["effective_optimizer_action_tokens"] for item in prepared_groups
    )
    parity = audit_initial_replay_parity(
        model=model,
        groups=ordered_groups,
        method=method,
        identity=identity,
        pad_token_id=pad_token_id,
        microbatch_size=microbatch_size,
        device=device,
        thresholds=parity_thresholds,
    )
    before = _parameter_snapshot(model)
    parameters = _trainable_parameters(model)
    optimizer_updates = 0
    metric_tokens = 0
    ratio_sum = 0.0
    clip_sum = 0.0
    kl_sum = 0.0
    entropy_sum = 0.0
    maximum_absolute_log_ratio = 0.0
    gradient_norms: list[float] = []
    zero_advantage_groups = sum(item["zero_advantage_group"] for item in prepared_groups)
    model.train()
    for _policy_epoch in range(ONLINE_LEARNER_CONFIG["policy_epochs"]):
        for prepared in prepared_groups:
            if prepared["zero_advantage_group"]:
                continue
            optimizer.zero_grad(set_to_none=True)
            examples = sorted(
                prepared["examples"],
                key=lambda example: (example.forward_tokens, example.trajectory_index, example.turn_index),
            )
            for chunk in _chunked(examples, microbatch_size):
                batch = _move_replay_batch(
                    collate_turn_training_examples(chunk, pad_token_id=pad_token_id),
                    device,
                )
                replay, entropy = _forward_replay(model, batch)
                loss = hierarchical_clipped_policy_loss(replay, batch, entropy=entropy)
                loss["loss"].backward()
                count = loss["token_count"]
                metric_tokens += count
                ratio_sum += float(loss["mean_ratio"].detach().cpu()) * count
                clip_sum += float(loss["clip_fraction"].detach().cpu()) * count
                kl_sum += float(loss["approx_kl"].detach().cpu()) * count
                entropy_sum += float(loss["entropy"].detach().cpu()) * count
                maximum_absolute_log_ratio = max(
                    maximum_absolute_log_ratio,
                    float(loss["maximum_absolute_log_ratio"].detach().cpu()),
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                ONLINE_LEARNER_CONFIG["gradient_clip"],
            )
            _require(bool(torch.isfinite(gradient_norm)), "online learner gradient norm is non-finite")
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            optimizer.step()
            optimizer_updates += 1
    parameter_change = _parameter_change_norm(model, before)
    _require(
        (optimizer_updates == 0 and parameter_change == 0.0)
        or (optimizer_updates > 0 and parameter_change > 0.0),
        "online learner optimizer/parameter change mismatch",
    )
    _require(
        optimizer_updates
        == (len(prepared_groups) - zero_advantage_groups)
        * ONLINE_LEARNER_CONFIG["policy_epochs"],
        "online learner optimizer update count drift",
    )
    denominator = max(1, metric_tokens)
    return {
        "schema_version": "m4_long_horizon_learner_report_v2",
        "identity_sha256": identity.sha256,
        "collection_sha256": collection_sha256,
        "method": method,
        "policy_epochs": ONLINE_LEARNER_CONFIG["policy_epochs"],
        "trajectory_minibatch_size": ONLINE_LEARNER_CONFIG["trajectory_minibatch_size"],
        "turn_microbatch_size": microbatch_size,
        "group_count": len(prepared_groups),
        "zero_advantage_group_count": zero_advantage_groups,
        "credit_assignment": credit_assignment,
        "optimizer_updates": optimizer_updates,
        "committed_group_action_tokens": committed_tokens,
        "effective_optimizer_action_tokens": effective_unique_tokens,
        "optimizer_evaluated_action_tokens": (
            effective_unique_tokens * ONLINE_LEARNER_CONFIG["policy_epochs"]
        ),
        "all_generated_action_tokens": all_generated_action_tokens,
        "effective_optimizer_action_token_fraction": (
            effective_unique_tokens / all_generated_action_tokens
        ),
        "mean_ratio": ratio_sum / denominator if metric_tokens else 1.0,
        "clip_fraction": clip_sum / denominator,
        "approx_kl": kl_sum / denominator,
        "entropy": entropy_sum / denominator,
        "maximum_absolute_log_ratio": maximum_absolute_log_ratio,
        "gradient_norm": (
            sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
        ),
        "maximum_gradient_norm": max(gradient_norms, default=0.0),
        "parameter_change_norm": parameter_change,
        "initial_replay_parity": parity,
        "behavior_policy_staleness": 0,
        "complete": True,
    }


def load_trainable_policy_model(
    *,
    base_model: Path,
    adapter_path: Path,
    device: str = "cuda:0",
) -> tuple[Any, Any]:
    """Load the exact frozen adapter without merging it into base weights."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        local_files_only=True,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        is_trainable=True,
    )
    model.train()
    _trainable_parameters(model)
    return model, tokenizer


def build_or_load_policy_optimizer(
    *,
    model: Any,
    identity: RunIdentity,
    optimizer_artifact: Path | None,
) -> torch.optim.Optimizer:
    parameters = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=ONLINE_LEARNER_CONFIG["learning_rate"],
        weight_decay=0.0,
    )
    if optimizer_artifact is not None:
        payload = torch.load(
            Path(optimizer_artifact).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        _require(
            isinstance(payload, dict)
            and payload.get("schema_version") == LEARNER_OPTIMIZER_SCHEMA,
            "online optimizer artifact schema drift",
        )
        _require(
            payload.get("adapter_sha256") == identity.input_adapter_sha256,
            "online optimizer/adapter lineage mismatch",
        )
        _require(
            payload.get("rollout_adapter_sha256")
            == identity.input_rollout_adapter_sha256,
            "online optimizer/rollout-adapter lineage mismatch",
        )
        _require(
            payload.get("adapter_semantic_sha256")
            == identity.input_adapter_semantic_sha256,
            "online optimizer/adapter-semantic lineage mismatch",
        )
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    return optimizer


def create_bootstrap_optimizer_artifact(
    *,
    path: Path,
    identity: RunIdentity,
) -> dict[str, Any]:
    """Create an audited fresh-AdamW marker before the first collection."""

    identity.validate()
    destination = Path(path).expanduser().resolve()
    _require(not destination.exists(), "bootstrap optimizer artifact already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LEARNER_OPTIMIZER_SCHEMA,
        "input_identity_sha256": identity.sha256,
        "adapter_sha256": identity.input_adapter_sha256,
        "rollout_adapter_sha256": identity.input_rollout_adapter_sha256,
        "adapter_semantic_sha256": identity.input_adapter_semantic_sha256,
        "optimizer_state_dict": None,
        "optimizer_steps": 0,
        "initialization": "fresh_adamw_at_first_learner_phase",
    }
    torch.save(payload, destination)
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return {"path": str(destination), "sha256": sha256_file(destination), "payload": payload}


def save_policy_update_artifacts(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    identity: RunIdentity,
    learner_report: Mapping[str, Any],
    output_adapter: Path,
    output_rollout_adapter: Path,
    output_optimizer: Path,
    base_model: Path,
    input_adapter: Path,
    input_optimizer: Path,
) -> dict[str, Any]:
    """Write learner outputs inside an uncommitted iteration stage."""

    adapter_destination = Path(output_adapter).expanduser().resolve()
    rollout_adapter_destination = Path(output_rollout_adapter).expanduser().resolve()
    optimizer_destination = Path(output_optimizer).expanduser().resolve()
    _require(not adapter_destination.exists(), "output adapter path already exists")
    _require(
        not rollout_adapter_destination.exists(),
        "output rollout adapter path already exists",
    )
    _require(not optimizer_destination.exists(), "output optimizer path already exists")
    updates = int(learner_report.get("optimizer_updates", -1))
    if updates == 0:
        shutil.copytree(Path(input_adapter).expanduser().resolve(), adapter_destination)
    else:
        model.save_pretrained(adapter_destination, safe_serialization=True)
        tokenizer.save_pretrained(adapter_destination)
    rollout_audit = build_vllm_adapter_view(
        source_adapter=adapter_destination,
        destination=rollout_adapter_destination,
        base_model=Path(base_model).expanduser().resolve(),
    )
    artifact_audit = {
        "output_adapter_sha256": directory_sha256(adapter_destination),
        "output_rollout_adapter_sha256": rollout_audit[
            "view_directory_sha256"
        ],
        "output_adapter_semantic_sha256": rollout_audit[
            "semantic_tensor_sha256"
        ],
    }
    if updates == 0:
        source_optimizer = Path(input_optimizer).expanduser().resolve()
        if source_optimizer.is_dir():
            shutil.copytree(source_optimizer, optimizer_destination)
        else:
            shutil.copy2(source_optimizer, optimizer_destination)
    else:
        payload = {
            "schema_version": LEARNER_OPTIMIZER_SCHEMA,
            "input_identity_sha256": identity.sha256,
            "adapter_sha256": artifact_audit["output_adapter_sha256"],
            "rollout_adapter_sha256": artifact_audit[
                "output_rollout_adapter_sha256"
            ],
            "adapter_semantic_sha256": artifact_audit[
                "output_adapter_semantic_sha256"
            ],
            "learner_report_sha256": sha256_json(
                dict(learner_report) | artifact_audit
            ),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(payload, optimizer_destination)
        descriptor = os.open(optimizer_destination, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_descriptor = os.open(optimizer_destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    return {
        **artifact_audit,
        "output_optimizer_sha256": (
            directory_sha256(optimizer_destination)
            if optimizer_destination.is_dir()
            else sha256_file(optimizer_destination)
        ),
    }
