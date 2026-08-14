"""Small full-horizon K4x4 trajectory-GRPO updates for the Phase4 experiment.

This deliberately avoids the legacy M6 K8/short-horizon protocol.  It keeps
only the checks needed to know that the update is on-policy and comparable.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..long_horizon_rl.adapter_view import build_vllm_adapter_view
from ..long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_file, sha256_json
from .m6_online_training import strict_grpo_kl_loss, validate_committed_group

GROUP_SIZE = 4
TASK_GROUPS_PER_UPDATE = 4
REFERENCE_ADAPTER_NAME = "phase4_frozen_sft_reference"
OPTIMIZER_SCHEMA = "m6_phase4_online_optimizer_v1"
REPORT_SCHEMA = "m6_phase4_online_update_v1"
POLICY_CREDIT_WINDOWS = {"full", "tail2"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def standardized_binary_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    _require(len(rewards) == GROUP_SIZE, "Phase4 advantages require K4")
    values = tuple(float(value) for value in rewards)
    _require(all(value in {0.0, 1.0} for value in values), "Phase4 reward must be strict binary")
    mean = sum(values) / GROUP_SIZE
    variance = sum((value - mean) ** 2 for value in values) / GROUP_SIZE
    if variance == 0.0:
        return (0.0,) * GROUP_SIZE
    scale = math.sqrt(variance)
    return tuple((value - mean) / scale for value in values)


@dataclass(frozen=True)
class Phase4TurnExample:
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
    token_loss_weight: float
    policy_token_loss_weight: float

    @property
    def completion_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def forward_tokens(self) -> int:
        return len(self.prompt_token_ids) + self.completion_tokens


def trajectory_policy_turn_weights(
    completion_tokens: Sequence[int],
    *,
    policy_credit_window: str,
) -> tuple[float, ...]:
    """Return per-token turn weights with unit mass inside one trajectory."""

    _require(policy_credit_window in POLICY_CREDIT_WINDOWS, "unsupported Phase4 policy credit window")
    lengths = tuple(int(value) for value in completion_tokens)
    _require(bool(lengths) and all(value > 0 for value in lengths), "Phase4 turn token count is invalid")
    selected = len(lengths) if policy_credit_window == "full" else min(2, len(lengths))
    first = len(lengths) - selected
    return tuple(0.0 if index < first else 1.0 / (selected * length) for index, length in enumerate(lengths))


def prepare_k4x4_examples(
    groups: Sequence[Mapping[str, Any]],
    *,
    policy_credit_window: str = "full",
) -> dict[str, Any]:
    """Build one equal-task-weight update; homogeneous attempted tasks get zero gradient."""

    _require(len(groups) == TASK_GROUPS_PER_UPDATE, "Phase4 update requires four task groups")
    validated = [validate_committed_group(group, require_k=GROUP_SIZE) for group in groups]
    _require(len({group["task_id"] for group in validated}) == len(validated), "Phase4 update repeated a task")
    _require(all(group.get("training_updates_allowed") is True for group in validated), "Phase4 group is not update-authorized")
    active: list[tuple[dict[str, Any], tuple[float, ...]]] = []
    rows: list[dict[str, Any]] = []
    for group in validated:
        rewards = tuple(float(trajectory["binary_reward"]) for trajectory in group["trajectories"])
        advantages = standardized_binary_advantages(rewards)
        mixed = len(set(rewards)) > 1
        rows.append({
            "group_id": group["group_id"],
            "task_id": group["task_id"],
            "strict_success_count": int(sum(rewards)),
            "mixed": mixed,
        })
        if mixed:
            active.append((group, advantages))
    _require(active, "Phase4 K4x4 batch has no mixed strict-reward task")

    examples: list[Phase4TurnExample] = []
    active_group_weight = 1.0 / len(active)
    for group, advantages in active:
        for trajectory_index, (trajectory, advantage) in enumerate(zip(group["trajectories"], advantages)):
            turns = trajectory["turns"]
            _require(turns, "Phase4 trajectory has no generated turn")
            policy_turn_weights = trajectory_policy_turn_weights(
                [len(turn["generated_token_ids"]) for turn in turns],
                policy_credit_window=policy_credit_window,
            )
            for turn, policy_turn_weight in zip(turns, policy_turn_weights):
                generated = tuple(int(value) for value in turn["generated_token_ids"])
                examples.append(
                    Phase4TurnExample(
                        group_id=group["group_id"],
                        trajectory_id=trajectory["trajectory_id"],
                        trajectory_index=trajectory_index,
                        turn_index=int(turn["turn_index"]),
                        turns_in_trajectory=len(turns),
                        prompt_token_ids=tuple(int(value) for value in turn["prompt_token_ids"]),
                        generated_token_ids=generated,
                        behavior_logprobs=tuple(float(value) for value in turn["behavior_logprobs"]),
                        sampling_logprobs=tuple(float(value) for value in turn["sampling_logprobs"]),
                        advantage=float(advantage),
                        token_loss_weight=active_group_weight / (GROUP_SIZE * len(turns) * len(generated)),
                        policy_token_loss_weight=active_group_weight * policy_turn_weight / GROUP_SIZE,
                    )
                )
    total_weight = sum(example.token_loss_weight * example.completion_tokens for example in examples)
    policy_total_weight = sum(example.policy_token_loss_weight * example.completion_tokens for example in examples)
    _require(math.isclose(total_weight, 1.0, abs_tol=1e-10), "Phase4 K4x4 hierarchy does not sum to one")
    _require(math.isclose(policy_total_weight, 1.0, abs_tol=1e-10), "Phase4 policy credit hierarchy does not sum to one")
    return {
        "groups": validated,
        "examples": tuple(sorted(examples, key=lambda item: (item.forward_tokens, item.group_id, item.trajectory_index, item.turn_index))),
        "group_rows": rows,
        "attempted_group_count": len(validated),
        "active_mixed_group_count": len(active),
        "hierarchical_weight_sum": total_weight,
        "policy_hierarchical_weight_sum": policy_total_weight,
        "policy_credit_window": policy_credit_window,
    }


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    moved = dict(batch)
    for field in (
        "input_ids", "attention_mask", "position_ids", "generated_token_ids",
        "behavior_logprobs", "completion_mask", "advantages", "token_loss_weights",
        "completion_lengths",
    ):
        moved[field] = batch[field].to(device, non_blocking=True)
    if "policy_token_loss_weights" in batch:
        moved["policy_token_loss_weights"] = batch["policy_token_loss_weights"].to(device, non_blocking=True)
    return moved


def collate_phase4_examples(examples: Sequence[Phase4TurnExample], *, pad_token_id: int) -> dict[str, Any]:
    """Use the shared replay collator while keeping policy and KL weights separate."""

    import torch
    from ..long_horizon_rl.learner import collate_turn_training_examples

    batch = collate_turn_training_examples(examples, pad_token_id=pad_token_id)
    batch["policy_token_loss_weights"] = torch.tensor(
        [example.policy_token_loss_weight for example in examples], dtype=torch.float32
    )
    return batch


def _forward_logprobs(model: Any, batch: Mapping[str, Any]) -> Any:
    from ..long_horizon_rl.learner import extract_tail_completion_logprobs

    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
        logits_to_keep=int(batch["logits_to_keep"]),
    )
    logits = output.logits if hasattr(output, "logits") else output[0]
    replay, _ = extract_tail_completion_logprobs(logits, batch)
    return replay


def _active_adapter_name(model: Any) -> str:
    active = getattr(model, "active_adapters", None)
    active = active() if callable(active) else active
    if isinstance(active, (list, tuple)):
        _require(len(active) == 1, "Phase4 learner requires one active adapter")
        return str(active[0])
    value = str(getattr(model, "active_adapter", "default"))
    _require(value, "Phase4 learner active adapter is empty")
    return value


def train_phase4_iteration(
    *,
    groups: Sequence[Mapping[str, Any]],
    base_model: Path,
    input_adapter: Path,
    input_adapter_semantic_sha256: str,
    reference_sft_adapter: Path,
    output_root: Path,
    iteration_index: int,
    seed: int,
    learning_rate: float = 3e-6,
    kl_coefficient: float = 0.03,
    kl_hard_stop: float = 0.01,
    microbatch_size: int = 4,
    input_optimizer: Path | None = None,
) -> dict[str, Any]:
    """Apply exactly one optimizer step from four on-policy K4 task groups."""

    import torch
    from ..long_horizon_rl.learner import load_trainable_policy_model

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Phase4 learner requires one GPU")
    _require(iteration_index >= 0 and seed >= 0, "Phase4 iteration identity drift")
    _require(learning_rate == 3e-6, "Phase4 learning rate must remain 3e-6")
    _require(kl_coefficient > 0 and kl_hard_stop == 0.01, "Phase4 KL configuration drift")
    _require(microbatch_size in {1, 2, 4, 8}, "Phase4 microbatch size drift")
    prepared = prepare_k4x4_examples(groups)
    root = output_root.expanduser().resolve()
    report_path = root / "learner_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _require(report.get("content_sha256") == _self_hash(report), "Phase4 recovered report hash drift")
        return report
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        interrupted = root.parent / "interrupted_updates"
        interrupted.mkdir(exist_ok=True)
        os.rename(root, interrupted / f"{root.name}_{time.time_ns()}")
    staging = root.parent / f".{root.name}.staging-{os.getpid()}-{time.time_ns()}"
    staging.mkdir()

    torch.manual_seed(seed + iteration_index)
    torch.cuda.manual_seed_all(seed + iteration_index)
    device = torch.device("cuda:0")
    model, tokenizer = load_trainable_policy_model(base_model=base_model, adapter_path=input_adapter)
    current = _active_adapter_name(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(parameters, "Phase4 learner has no trainable parameters")
    model.load_adapter(str(reference_sft_adapter.expanduser().resolve()), adapter_name=REFERENCE_ADAPTER_NAME, is_trainable=False)
    for name, parameter in model.named_parameters():
        if REFERENCE_ADAPTER_NAME in name:
            parameter.requires_grad_(False)
    dropout_modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    _require(dropout_modules, "Phase4 learner found no dropout modules")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    cumulative_updates = 0
    input_optimizer_sha256 = None
    if input_optimizer is not None:
        checkpoint = torch.load(input_optimizer.expanduser().resolve(), map_location="cpu", weights_only=False)
        _require(checkpoint.get("schema_version") == OPTIMIZER_SCHEMA, "Phase4 optimizer schema drift")
        _require(checkpoint.get("iteration_index") == iteration_index - 1, "Phase4 optimizer iteration drift")
        _require(checkpoint.get("learning_rate") == learning_rate, "Phase4 optimizer learning-rate drift")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cumulative_updates = int(checkpoint["cumulative_optimizer_updates"])
        input_optimizer_sha256 = sha256_file(input_optimizer)

    examples = prepared["examples"]
    behavior: list[float] = []
    replay_before: list[float] = []
    model.set_adapter(current)
    model.eval()
    with torch.inference_mode():
        for chunk in _chunks(examples, microbatch_size):
            batch = _move_batch(collate_phase4_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
            replay = _forward_logprobs(model, batch)
            for row, example in enumerate(chunk):
                behavior.extend(example.behavior_logprobs)
                replay_before.extend(float(value) for value in replay[row, : example.completion_tokens].cpu().tolist())
    log_ratios = [current_value - old_value for current_value, old_value in zip(replay_before, behavior)]
    initial_clip_fraction = sum(abs(value) > math.log(1.2) for value in log_ratios) / len(log_ratios)
    _require(initial_clip_fraction <= 0.10, "Phase4 initial behavior/replay clip fraction exceeded 10%")

    optimizer.zero_grad(set_to_none=True)
    metric_tokens = 0
    policy_loss = reference_kl = total_loss = 0.0
    for chunk in _chunks(examples, microbatch_size):
        batch = _move_batch(collate_phase4_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
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
            replay, reference, batch, clip_epsilon=0.2, kl_coefficient=kl_coefficient
        )
        result["loss"].backward()
        count = int(result["token_count"])
        metric_tokens += count
        policy_loss += float(result["policy_loss"].detach().cpu())
        reference_kl += float(result["mean_token_reference_kl"].detach().cpu()) * count
        total_loss += float(result["loss"].detach().cpu())
    mean_reference_kl = reference_kl / metric_tokens
    _require(math.isfinite(mean_reference_kl) and mean_reference_kl <= kl_hard_stop, "Phase4 reference KL exceeded hard stop")
    gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    _require(bool(torch.isfinite(gradient)), "Phase4 gradient is non-finite")
    optimizer.step()

    model.set_adapter(current)
    if hasattr(model, "delete_adapter"):
        model.delete_adapter(REFERENCE_ADAPTER_NAME)
    adapter = staging / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    view = staging / "rollout_adapter"
    view_audit = build_vllm_adapter_view(source_adapter=adapter, destination=view, base_model=base_model)
    _require(
        view_audit["semantic_tensor_sha256"] != input_adapter_semantic_sha256,
        "Phase4 optimizer step produced no parameter change",
    )
    optimizer_path = staging / "optimizer.pt"
    torch.save(
        {
            "schema_version": OPTIMIZER_SCHEMA,
            "iteration_index": iteration_index,
            "learning_rate": learning_rate,
            "cumulative_optimizer_updates": cumulative_updates + 1,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        optimizer_path,
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "complete": True,
        "training_performed": True,
        "method": "single_epoch_trajectory_group_normalized_policy_gradient_with_sft_kl",
        "reward": "strict_binary",
        "iteration_index": iteration_index,
        "seed": seed,
        "group_size": GROUP_SIZE,
        "attempted_task_groups": prepared["attempted_group_count"],
        "active_mixed_task_groups": prepared["active_mixed_group_count"],
        "group_rows": prepared["group_rows"],
        "task_groups_per_optimizer_step": TASK_GROUPS_PER_UPDATE,
        "optimizer_steps": 1,
        "cumulative_optimizer_updates": cumulative_updates + 1,
        "learning_rate": learning_rate,
        "lora_dropout_during_rl": 0.0,
        "kl_coefficient": kl_coefficient,
        "reference_kl": mean_reference_kl,
        "kl_hard_stop": kl_hard_stop,
        "initial_replay_clip_fraction": initial_clip_fraction,
        "gradient_norm_before_clip": float(gradient.detach().cpu()),
        "policy_loss": policy_loss,
        "total_loss": total_loss,
        "optimizer_action_tokens": metric_tokens,
        "input_adapter": str(input_adapter.expanduser().resolve()),
        "input_adapter_sha256": directory_sha256(input_adapter),
        "input_adapter_semantic_sha256": input_adapter_semantic_sha256,
        "reference_sft_adapter": str(reference_sft_adapter.expanduser().resolve()),
        "reference_sft_adapter_sha256": directory_sha256(reference_sft_adapter),
        "input_optimizer_sha256": input_optimizer_sha256,
        "output_adapter": str(root / "adapter"),
        "output_adapter_sha256": directory_sha256(adapter),
        "output_adapter_semantic_sha256": view_audit["semantic_tensor_sha256"],
        "output_optimizer": str(root / "optimizer.pt"),
        "output_optimizer_sha256": sha256_file(optimizer_path),
        "source_group_content_sha256": [group["content_sha256"] for group in prepared["groups"]],
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(staging / "learner_report.json", report)
    os.rename(staging, root)
    return report
