"""Fail-closed M5 WebShop online rollout and learner primitives.

This module deliberately owns an M5 evidence schema instead of reusing the M4
browser-study identity.  Only generic tensor replay and vLLM engine mechanics
are shared; task identity, public-state anchors, context length and artifact
lineage remain bound to the frozen WebShop protocol.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..long_horizon_rl.adapter_view import build_vllm_adapter_view
from ..long_horizon_rl.contracts import (
    SHA256_PATTERN,
    atomic_write_json,
    directory_sha256,
    sha256_file,
    sha256_json,
    token_ids_sha256,
)
from ..long_horizon_rl.learner import (
    collate_turn_training_examples,
    extract_tail_completion_logprobs,
    hierarchical_clipped_policy_loss,
    load_trainable_policy_model,
    summarize_logprob_parity,
)
from ..long_horizon_rl.vllm_backend import VLLMBackendConfig
from .credit import (
    ANCHOR_METHOD,
    BASELINE_METHOD,
    GROUP_SIZE,
    assign_group_credit,
    policy_context_signature,
    public_state_anchor_signature,
)

GROUP_SCHEMA = "m5_webshop_k4_group_v2"
LEARNER_REPORT_SCHEMA = "m5_webshop_online_learner_report_v2"
OPTIMIZER_SCHEMA = "m5_webshop_online_optimizer_v2"
FORMAL_ITERATION_LEARNER_SCHEMA = "m5_webshop_formal_iteration_learner_v1"
FORMAL_OPTIMIZER_SCHEMA = "m5_webshop_formal_optimizer_v1"
METHODS = (BASELINE_METHOD, ANCHOR_METHOD)
MAX_SEQUENCE_TOKENS = 8192
MAX_NEW_TOKENS = 128


class ReplayParityError(ValueError):
    """Fail-closed parity error that preserves the complete numeric audit."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__(
            "M5 behavior/replay parity failed: "
            + json.dumps(self.report, sort_keys=True, separators=(",", ":"))
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


@dataclass(frozen=True)
class M5VLLMBackendConfig(VLLMBackendConfig):
    """M5 specialization of the shared vLLM engine contract."""

    max_model_len: int = MAX_SEQUENCE_TOKENS
    max_new_tokens: int = MAX_NEW_TOKENS

    def validate(self, *, check_adapter_files: bool = True) -> None:
        _require(Path(self.base_model).is_absolute(), "vLLM base model path must be absolute")
        _require(Path(self.adapter_path).is_absolute(), "vLLM adapter path must be absolute")
        _require(Path(self.rollout_adapter_path).is_absolute(), "vLLM rollout adapter path must be absolute")
        for field in ("adapter_sha256", "rollout_adapter_sha256", "adapter_semantic_sha256"):
            value = getattr(self, field)
            _require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None, f"invalid {field}")
        _require(isinstance(self.seed, int) and self.seed >= 0, "invalid vLLM seed")
        _require(self.dtype == "bfloat16", "M5 vLLM dtype drift")
        _require(self.max_model_len == MAX_SEQUENCE_TOKENS, "M5 vLLM context drift")
        _require(self.max_new_tokens == MAX_NEW_TOKENS, "M5 vLLM turn-token cap drift")
        _require(math.isclose(self.gpu_memory_utilization, 0.5), "M5 vLLM memory fraction drift")
        _require(self.max_num_seqs == 32, "M5 vLLM concurrency drift")
        _require(self.enforce_eager is True, "M5 vLLM eager-mode gate disabled")
        _require(self.adapter_id == 1 and self.stream_interval == 8, "M5 vLLM adapter/stream contract drift")
        if check_adapter_files:
            from ..long_horizon_rl.adapter_view import validate_vllm_adapter_view

            canonical = Path(self.adapter_path).expanduser().resolve()
            view = Path(self.rollout_adapter_path).expanduser().resolve()
            _require(canonical.is_dir() and directory_sha256(canonical) == self.adapter_sha256, "M5 adapter hash drift")
            audit = validate_vllm_adapter_view(
                source_adapter=canonical,
                view_directory=view,
                base_model=Path(self.base_model),
            )
            _require(audit["view_directory_sha256"] == self.rollout_adapter_sha256, "M5 rollout view hash drift")
            _require(audit["semantic_tensor_sha256"] == self.adapter_semantic_sha256, "M5 semantic adapter hash drift")


def trajectory_from_episode(
    episode: Mapping[str, Any],
    *,
    trajectory_id: str,
    rollout_index: int,
    adapter_sha256: str,
    rollout_adapter_sha256: str,
    adapter_semantic_sha256: str,
) -> dict[str, Any]:
    """Convert one valid generic episode into exact M5 learner evidence."""

    _require(episode.get("rollout_valid") is True, "cannot commit an infrastructure-invalid trajectory")
    binary_reward = episode.get("reward")
    _require(
        isinstance(binary_reward, (int, float))
        and not isinstance(binary_reward, bool)
        and float(binary_reward) in (0.0, 1.0),
        "M5 binary success reward is invalid",
    )
    task_score = episode.get("task_score", binary_reward)
    _require(
        isinstance(task_score, (int, float))
        and not isinstance(task_score, bool)
        and math.isfinite(float(task_score))
        and 0.0 <= float(task_score) <= 1.0,
        "M5 official task score is invalid",
    )
    success = episode.get("success")
    _require(isinstance(success, bool), "M5 success flag is invalid")
    _require(success == (float(task_score) >= 0.999), "M5 success/task-score disagreement")
    source_turns = episode.get("turns")
    _require(isinstance(source_turns, list) and source_turns, "M5 trajectory has no generated turns")
    turns = []
    for index, source in enumerate(source_turns, start=1):
        _require(isinstance(source, Mapping), "M5 generated turn is malformed")
        prompt_ids = [int(value) for value in source.get("prompt_token_ids", [])]
        generated_ids = [int(value) for value in source.get("generated_token_ids", [])]
        behavior = [float(value) for value in source.get("token_logprobs", [])]
        sampling = [float(value) for value in source.get("sampling_logprobs", [])]
        _require(prompt_ids and generated_ids, "M5 generated turn has empty token evidence")
        _require(len(generated_ids) == len(behavior) == len(sampling), "M5 logprob/token length drift")
        _require(all(math.isfinite(value) for value in behavior + sampling), "M5 logprob is non-finite")
        _require(len(prompt_ids) + len(generated_ids) <= MAX_SEQUENCE_TOKENS, "M5 replay exceeds 8192 tokens")
        observation = source.get("observation")
        _require(isinstance(observation, Mapping), "M5 turn lacks public observation")
        _require(source.get("adapter_sha256") == adapter_sha256, "M5 behavior adapter drift")
        _require(source.get("rollout_adapter_sha256") == rollout_adapter_sha256, "M5 rollout adapter drift")
        _require(source.get("adapter_semantic_sha256") == adapter_semantic_sha256, "M5 semantic adapter drift")
        turns.append(
            {
                "turn_index": index,
                "public_state_anchor_sha256": public_state_anchor_signature(observation),
                "policy_context_token_sha256": policy_context_signature(prompt_ids),
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "generated_token_sha256": token_ids_sha256(generated_ids),
                "behavior_logprobs": behavior,
                "sampling_logprobs": sampling,
                "request_id": str(source.get("request_id", "")),
                "sampling_seed": int(source.get("sampling_seed", 0)),
                "generation_backend": str(source.get("generation_backend", "")),
                "schema_valid": bool(source.get("schema_valid", False)),
                "action": source.get("action"),
                "raw_output": str(source.get("raw_output", ""))[:4096],
                "observation": dict(observation),
            }
        )
    return {
        "trajectory_id": trajectory_id,
        "rollout_index": rollout_index,
        "task_id": str(episode.get("task_id", "")),
        "reward": float(task_score),
        "binary_reward": float(binary_reward),
        "success": success,
        "termination_reason": str(episode.get("termination_reason", "")),
        "environment_steps": int(episode.get("environment_steps", 0)),
        "generated_action_tokens": sum(len(turn["generated_token_ids"]) for turn in turns),
        "turns": turns,
    }


def build_committed_group(
    *,
    task_id: str,
    group_id: str,
    attempt_index: int,
    trajectories: Sequence[Mapping[str, Any]],
    git_sha: str,
    protocol_sha256: str,
    adapter_sha256: str,
    rollout_adapter_sha256: str,
    adapter_semantic_sha256: str,
) -> dict[str, Any]:
    _require(len(trajectories) == GROUP_SIZE, "M5 group is not K=4")
    payload = {
        "schema_version": GROUP_SCHEMA,
        "complete": True,
        "git_sha": git_sha,
        "protocol_sha256": protocol_sha256,
        "task_id": task_id,
        "group_id": group_id,
        "attempt_index": attempt_index,
        "adapter_sha256": adapter_sha256,
        "rollout_adapter_sha256": rollout_adapter_sha256,
        "adapter_semantic_sha256": adapter_semantic_sha256,
        "generated_action_tokens": sum(int(item["generated_action_tokens"]) for item in trajectories),
        "trajectories": [dict(item) for item in trajectories],
    }
    payload["content_sha256"] = _self_hash(payload)
    return validate_committed_group(payload)


def validate_committed_group(group: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(group)
    _require(value.get("schema_version") == GROUP_SCHEMA and value.get("complete") is True, "M5 group schema drift")
    _require(value.get("content_sha256") == _self_hash(value), "M5 group self-hash drift")
    for field in ("protocol_sha256", "adapter_sha256", "rollout_adapter_sha256", "adapter_semantic_sha256"):
        _require(isinstance(value.get(field), str) and SHA256_PATTERN.fullmatch(value[field]) is not None, f"M5 group {field} drift")
    trajectories = value.get("trajectories")
    _require(isinstance(trajectories, list) and len(trajectories) == GROUP_SIZE, "M5 group K drift")
    task_id = value.get("task_id")
    generated = 0
    rollout_indices = set()
    for trajectory in trajectories:
        _require(trajectory.get("task_id") == task_id, "M5 group crosses tasks")
        reward = trajectory.get("reward")
        _require(
            isinstance(reward, (int, float))
            and math.isfinite(float(reward))
            and 0.0 <= float(reward) <= 1.0,
            "M5 group official task-score drift",
        )
        success = trajectory.get("success")
        binary_reward = trajectory.get("binary_reward")
        _require(isinstance(success, bool), "M5 group success flag drift")
        _require(
            isinstance(binary_reward, (int, float))
            and not isinstance(binary_reward, bool)
            and float(binary_reward) in (0.0, 1.0)
            and bool(binary_reward) is success,
            "M5 group binary reward drift",
        )
        _require(success == (float(reward) >= 0.999), "M5 group success/task-score drift")
        turns = trajectory.get("turns")
        _require(isinstance(turns, list) and turns, "M5 group trajectory has no turns")
        rollout_indices.add(trajectory.get("rollout_index"))
        trajectory_tokens = 0
        for turn_index, turn in enumerate(turns, start=1):
            _require(turn.get("turn_index") == turn_index, "M5 turn order drift")
            prompt_ids = turn.get("prompt_token_ids")
            generated_ids = turn.get("generated_token_ids")
            behavior = turn.get("behavior_logprobs")
            sampling = turn.get("sampling_logprobs")
            _require(isinstance(prompt_ids, list) and prompt_ids and isinstance(generated_ids, list) and generated_ids, "M5 turn token evidence drift")
            _require(len(generated_ids) == len(behavior) == len(sampling), "M5 turn logprob length drift")
            _require(len(prompt_ids) + len(generated_ids) <= MAX_SEQUENCE_TOKENS, "M5 turn context overflow")
            _require(token_ids_sha256(generated_ids) == turn.get("generated_token_sha256"), "M5 generated token hash drift")
            _require(policy_context_signature(prompt_ids) == turn.get("policy_context_token_sha256"), "M5 prompt token hash drift")
            _require(public_state_anchor_signature(turn["observation"]) == turn.get("public_state_anchor_sha256"), "M5 public-state anchor drift")
            trajectory_tokens += len(generated_ids)
        _require(trajectory_tokens == trajectory.get("generated_action_tokens"), "M5 trajectory token ledger drift")
        generated += trajectory_tokens
    _require(rollout_indices == set(range(GROUP_SIZE)), "M5 rollout index roster drift")
    _require(generated == value.get("generated_action_tokens"), "M5 group token ledger drift")
    return value


@dataclass(frozen=True)
class M5TurnTrainingExample:
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
        return 1.0 / (GROUP_SIZE * self.turns_in_trajectory * self.completion_tokens)


def prepare_group_training_examples(group: Mapping[str, Any], method: str) -> dict[str, Any]:
    _require(method in METHODS, "unsupported M5 learner method")
    validated = validate_committed_group(group)
    credit = assign_group_credit(validated["trajectories"], method)
    examples = []
    effective = 0
    for trajectory_index, (trajectory, trajectory_credit) in enumerate(zip(validated["trajectories"], credit["turn_credit"])):
        turns = trajectory["turns"]
        _require(len(turns) == len(trajectory_credit), "M5 credit/turn count drift")
        for turn, assigned in zip(turns, trajectory_credit):
            advantage = float(assigned["turn_advantage"])
            example = M5TurnTrainingExample(
                group_id=validated["group_id"],
                trajectory_id=trajectory["trajectory_id"],
                trajectory_index=trajectory_index,
                turn_index=turn["turn_index"],
                turns_in_trajectory=len(turns),
                prompt_token_ids=tuple(turn["prompt_token_ids"]),
                generated_token_ids=tuple(turn["generated_token_ids"]),
                behavior_logprobs=tuple(turn["behavior_logprobs"]),
                sampling_logprobs=tuple(turn["sampling_logprobs"]),
                advantage=advantage,
            )
            examples.append(example)
            effective += example.completion_tokens if abs(advantage) > 0 else 0
    _require(math.isclose(sum(item.token_loss_weight * item.completion_tokens for item in examples), 1.0, abs_tol=1e-12), "M5 hierarchical weight drift")
    return {
        "group": validated,
        "credit": credit,
        "examples": tuple(examples),
        "generated_action_tokens": validated["generated_action_tokens"],
        "effective_optimizer_action_tokens": effective,
        "zero_advantage_group": effective == 0,
    }


def audit_collection(groups: Sequence[Mapping[str, Any]], *, all_generated_action_tokens: int) -> dict[str, Any]:
    _require(bool(groups), "M5 collection audit requires at least one K4 group")
    validated = [validate_committed_group(group) for group in groups]
    rewards = [float(trajectory["reward"]) for group in validated for trajectory in group["trajectories"]]
    successes = [bool(trajectory["success"]) for group in validated for trajectory in group["trajectories"]]
    mixed = sum(len({float(item["reward"]) for item in group["trajectories"]}) > 1 for group in validated)
    anchor_reports = [assign_group_credit(group["trajectories"], ANCHOR_METHOD) for group in validated]
    total_turns = sum(report["metrics"]["turn_count"] for report in anchor_reports)
    informative = sum(report["metrics"]["informative_micro_turn_count"] for report in anchor_reports)
    noninitial_groups = sum(report["metrics"]["non_initial_shared_anchor_count"] > 0 for report in anchor_reports)
    initial_shared = sum(len({trajectory["turns"][0]["public_state_anchor_sha256"] for trajectory in group["trajectories"]}) == 1 for group in validated)
    committed_tokens = sum(group["generated_action_tokens"] for group in validated)
    _require(all_generated_action_tokens >= committed_tokens > 0, "M5 all-token ledger excludes committed tokens")
    return {
        "group_count": len(validated),
        "trajectory_count": len(rewards),
        "success_count": sum(successes),
        "success_rate": sum(successes) / len(successes),
        "mean_official_task_score": sum(rewards) / len(rewards),
        "nonzero_task_score_count": sum(value > 0 for value in rewards),
        "mixed_reward_group_count": mixed,
        "mixed_reward_group_fraction": mixed / len(validated),
        "initial_shared_anchor_group_count": initial_shared,
        "initial_shared_anchor_group_fraction": initial_shared / len(validated),
        "informative_micro_turn_count": informative,
        "informative_micro_turn_fraction": informative / total_turns,
        "shared_noninitial_group_count": noninitial_groups,
        "shared_noninitial_group_fraction": noninitial_groups / len(validated),
        "committed_group_action_tokens": committed_tokens,
        "all_generated_action_tokens": all_generated_action_tokens,
    }


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    value = dict(batch)
    for field in ("input_ids", "attention_mask", "position_ids", "generated_token_ids", "behavior_logprobs", "completion_mask", "advantages", "token_loss_weights", "completion_lengths"):
        value[field] = batch[field].to(device, non_blocking=True)
    return value


def _forward(model: Any, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
        logits_to_keep=int(batch["logits_to_keep"]),
    )
    return extract_tail_completion_logprobs(output.logits, batch)


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _optimizer_examples(item: Mapping[str, Any]) -> list[M5TurnTrainingExample]:
    """Return the exact per-group order used by both parity and optimization."""

    return sorted(
        item["examples"],
        key=lambda example: (
            example.forward_tokens,
            example.trajectory_index,
            example.turn_index,
        ),
    )


def _replay_logprobs(model: Any, examples: Sequence[M5TurnTrainingExample], tokenizer: Any, device: torch.device, microbatch_size: int) -> list[float]:
    output = []
    model.eval()
    with torch.inference_mode():
        for chunk in _chunks(examples, microbatch_size):
            batch = _move_batch(collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
            replay, _ = _forward(model, batch)
            for row, example in enumerate(chunk):
                output.extend(float(value) for value in replay[row, : example.completion_tokens].detach().cpu().tolist())
    return output


def audit_initial_replay_parity(
    *,
    model: Any,
    prepared: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    device: torch.device,
    microbatch_size: int,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    behavior: list[float] = []
    sampling: list[float] = []
    replay: list[float] = []
    for item in prepared:
        examples = _optimizer_examples(item)
        behavior.extend(value for example in examples for value in example.behavior_logprobs)
        sampling.extend(value for example in examples for value in example.sampling_logprobs)
        replay.extend(_replay_logprobs(model, examples, tokenizer, device, microbatch_size))
    sampling_report = summarize_logprob_parity(behavior, sampling)
    _require(sampling_report["maximum_absolute_logprob_difference"] <= thresholds["behavior_sampling_maximum_absolute_difference"], "M5 behavior/sampling parity failed")
    report = summarize_logprob_parity(behavior, replay)
    checks = {
        "mean": report["mean_absolute_logprob_difference"] <= thresholds["replay_mean_absolute_difference"],
        "p95": report["p95_absolute_logprob_difference"] <= thresholds["replay_p95_absolute_difference"],
        "p99": report["p99_absolute_logprob_difference"] <= thresholds["replay_p99_absolute_difference"],
        "p999": report["p999_absolute_logprob_difference"] <= thresholds["replay_p999_absolute_difference"],
        "clip_fraction": report["initial_ratio_clip_fraction"] <= thresholds["replay_initial_ratio_clip_fraction"],
        "mean_ratio": abs(report["mean_importance_ratio"] - 1.0) <= thresholds["mean_importance_ratio_absolute_deviation"],
    }
    audit = {
        **report,
        "behavior_sampling": sampling_report,
        "checks": checks,
        "thresholds": dict(thresholds),
        "passed": all(checks.values()),
    }
    if not audit["passed"]:
        raise ReplayParityError(audit)
    return audit


def train_policy_preflight(
    *,
    method: str,
    groups: Sequence[Mapping[str, Any]],
    all_generated_action_tokens: int,
    base_model: Path,
    initial_adapter: Path,
    output_root: Path,
    protocol: Mapping[str, Any],
    git_sha: str,
    protocol_sha256: str,
    microbatch_size: int = 4,
) -> dict[str, Any]:
    """Train one disposable method from the shared SFT adapter."""

    _require(method in METHODS, "unsupported M5 method")
    prepared_all = [prepare_group_training_examples(group, method) for group in groups]
    candidates = [item for item in prepared_all if not item["zero_advantage_group"]]
    prepared = []
    selected_effective_tokens = 0
    for item in candidates:
        prepared.append(item)
        selected_effective_tokens += item["effective_optimizer_action_tokens"]
        if selected_effective_tokens / all_generated_action_tokens >= 0.15:
            break
    effective_tokens = sum(item["effective_optimizer_action_tokens"] for item in prepared)
    effective_fraction = effective_tokens / all_generated_action_tokens
    _require(len(prepared) >= 1 and effective_fraction >= 0.15, "M5 effective optimizer token gate failed before update")
    parent = Path(output_root).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / method
    report_path = root / "learner_report.json"
    if report_path.is_file():
        return validate_learner_report(report_path)
    interrupted = parent / "interrupted_stages"
    stale = ([root] if root.exists() else []) + sorted(parent.glob(f".{method}.staging-*"))
    for index, path in enumerate(stale):
        interrupted.mkdir(exist_ok=True)
        destination = interrupted / f"{method}_{time.time_ns()}_{index}"
        os.rename(path, destination)
    staging = parent / f".{method}.staging-{os.getpid()}-{time.time_ns()}"
    staging.mkdir()
    device = torch.device("cuda:0")
    model, tokenizer = load_trainable_policy_model(base_model=base_model, adapter_path=initial_adapter)
    learner = protocol["online"]["learner"]
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(learner["learning_rate"]), weight_decay=0.0)
    parity = audit_initial_replay_parity(
        model=model,
        prepared=prepared,
        tokenizer=tokenizer,
        device=device,
        microbatch_size=microbatch_size,
        thresholds=protocol["online"]["parity_contract"],
    )
    updates = 0
    losses = []
    gradients = []
    metric_tokens = 0
    model.train()
    for _epoch in range(int(learner["policy_epochs"])):
        for item in prepared:
            optimizer.zero_grad(set_to_none=True)
            examples = _optimizer_examples(item)
            for chunk in _chunks(examples, microbatch_size):
                batch = _move_batch(collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
                replay, entropy = _forward(model, batch)
                result = hierarchical_clipped_policy_loss(replay, batch, clip_epsilon=float(learner["clip_epsilon"]), entropy=entropy)
                _require(bool(torch.isfinite(result["loss"])), "M5 learner loss is non-finite")
                result["loss"].backward()
                losses.append(float(result["loss"].detach().cpu()))
                metric_tokens += int(result["token_count"])
            gradient = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], float(learner["gradient_clip"]))
            _require(bool(torch.isfinite(gradient)), "M5 learner gradient is non-finite")
            gradients.append(float(gradient.detach().cpu()))
            optimizer.step()
            updates += 1
    _require(updates >= 2, "M5 learner completed fewer than two nonzero updates")
    adapter = staging / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    view = staging / "rollout_adapter"
    view_audit = build_vllm_adapter_view(source_adapter=adapter, destination=view, base_model=base_model)
    optimizer_path = staging / "optimizer.pt"
    torch.save(
        {
            "schema_version": OPTIMIZER_SCHEMA,
            "method": method,
            "git_sha": git_sha,
            "protocol_sha256": protocol_sha256,
            "adapter_sha256": directory_sha256(adapter),
            "optimizer_updates": updates,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        optimizer_path,
    )
    report = {
        "schema_version": LEARNER_REPORT_SCHEMA,
        "complete": True,
        "method": method,
        "git_sha": git_sha,
        "protocol_sha256": protocol_sha256,
        "group_count": len(groups),
        "available_nonzero_group_count": len(candidates),
        "selected_nonzero_group_count": len(prepared),
        "optimizer_updates": updates,
        "all_generated_action_tokens": all_generated_action_tokens,
        "effective_optimizer_action_tokens": effective_tokens,
        "effective_optimizer_action_token_fraction": effective_fraction,
        "optimizer_evaluated_action_tokens": metric_tokens,
        "mean_loss": sum(losses) / len(losses),
        "maximum_absolute_loss": max(abs(value) for value in losses),
        "mean_gradient_norm": sum(gradients) / len(gradients),
        "maximum_gradient_norm": max(gradients),
        "initial_replay_parity": parity,
        "input_adapter_sha256": directory_sha256(initial_adapter),
        "base_model": str(Path(base_model).expanduser().resolve()),
        "output_adapter": str(root / "adapter"),
        "output_adapter_sha256": directory_sha256(adapter),
        "output_rollout_adapter": str(root / "rollout_adapter"),
        "output_rollout_adapter_sha256": view_audit["view_directory_sha256"],
        "output_adapter_semantic_sha256": view_audit["semantic_tensor_sha256"],
        "output_optimizer": str(root / "optimizer.pt"),
        "output_optimizer_sha256": sha256_file(optimizer_path),
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(staging / "learner_report.json", report)
    os.rename(staging, root)
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    del model, optimizer
    torch.cuda.empty_cache()
    return report


def validate_learner_report(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(report.get("schema_version") == LEARNER_REPORT_SCHEMA and report.get("complete") is True, "M5 learner report schema drift")
    _require(report.get("content_sha256") == _self_hash(report), "M5 learner report self-hash drift")
    _require(directory_sha256(Path(report["output_adapter"])) == report["output_adapter_sha256"], "M5 learner adapter hash drift")
    from ..long_horizon_rl.adapter_view import validate_vllm_adapter_view

    rollout = validate_vllm_adapter_view(
        source_adapter=Path(report["output_adapter"]),
        view_directory=Path(report["output_rollout_adapter"]),
        base_model=Path(report["base_model"]),
    )
    _require(rollout["view_directory_sha256"] == report["output_rollout_adapter_sha256"], "M5 learner rollout adapter hash drift")
    _require(rollout["semantic_tensor_sha256"] == report["output_adapter_semantic_sha256"], "M5 learner semantic adapter hash drift")
    _require(sha256_file(Path(report["output_optimizer"])) == report["output_optimizer_sha256"], "M5 learner optimizer hash drift")
    return report


def train_policy_iteration(
    *,
    method: str,
    groups: Sequence[Mapping[str, Any]],
    iteration_generated_action_tokens: int,
    base_model: Path,
    initial_adapter: Path,
    input_adapter_semantic_sha256: str,
    input_optimizer: Path | None,
    output_root: Path,
    protocol: Mapping[str, Any],
    git_sha: str,
    protocol_sha256: str,
    iteration_index: int,
    seed: int,
    microbatch_size: int = 4,
) -> dict[str, Any]:
    """Apply one atomic formal on-policy update and persist Adam state.

    Every non-zero K4 group from the current behavior-policy snapshot is used.
    A zero-signal iteration is still committed with an unchanged policy so its
    rollout cost remains visible and the next iteration can continue safely.
    """

    _require(method in METHODS, "unsupported M5 formal method")
    _require(groups and iteration_generated_action_tokens > 0, "M5 formal iteration is empty")
    _require(iteration_index >= 0 and seed >= 0, "M5 formal iteration identity drift")
    prepared_all = [prepare_group_training_examples(group, method) for group in groups]
    prepared = [item for item in prepared_all if not item["zero_advantage_group"]]
    effective_tokens = sum(item["effective_optimizer_action_tokens"] for item in prepared)
    effective_fraction = effective_tokens / iteration_generated_action_tokens
    parent = Path(output_root).expanduser().resolve()
    parent.parent.mkdir(parents=True, exist_ok=True)
    report_path = parent / "learner_report.json"
    if report_path.is_file():
        return validate_formal_iteration_learner_report(report_path)
    interrupted = parent.parent / "interrupted_learner_stages"
    stale = ([parent] if parent.exists() else []) + sorted(parent.parent.glob(f".{parent.name}.staging-*"))
    for index, path in enumerate(stale):
        interrupted.mkdir(exist_ok=True)
        os.rename(path, interrupted / f"{parent.name}_{time.time_ns()}_{index}")
    staging = parent.parent / f".{parent.name}.staging-{os.getpid()}-{time.time_ns()}"
    staging.mkdir()
    device = torch.device("cuda:0")
    torch.manual_seed(seed + iteration_index)
    torch.cuda.manual_seed_all(seed + iteration_index)
    model, tokenizer = load_trainable_policy_model(base_model=base_model, adapter_path=initial_adapter)
    learner = protocol["online"]["learner"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learner["learning_rate"]),
        weight_decay=0.0,
    )
    cumulative_updates_before = 0
    input_optimizer_sha256 = None
    if input_optimizer is not None:
        checkpoint_path = Path(input_optimizer).expanduser().resolve()
        _require(checkpoint_path.is_file(), "M5 formal input optimizer is missing")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _require(checkpoint.get("schema_version") == FORMAL_OPTIMIZER_SCHEMA, "M5 formal optimizer schema drift")
        _require(checkpoint.get("method") == method, "M5 formal optimizer method drift")
        _require(checkpoint.get("git_sha") == git_sha, "M5 formal optimizer Git drift")
        _require(checkpoint.get("protocol_sha256") == protocol_sha256, "M5 formal optimizer protocol drift")
        _require(
            checkpoint.get("adapter_sha256") == directory_sha256(initial_adapter),
            "M5 formal optimizer/adapter lineage drift",
        )
        cumulative_updates_before = int(checkpoint.get("cumulative_optimizer_updates", -1))
        _require(cumulative_updates_before >= 0, "M5 formal optimizer update count drift")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        input_optimizer_sha256 = sha256_file(checkpoint_path)

    parity_source = prepared if prepared else prepared_all
    parity = audit_initial_replay_parity(
        model=model,
        prepared=parity_source,
        tokenizer=tokenizer,
        device=device,
        microbatch_size=microbatch_size,
        thresholds=protocol["online"]["parity_contract"],
    )
    updates = 0
    losses: list[float] = []
    gradients: list[float] = []
    metric_tokens = 0
    model.train()
    for _epoch in range(int(learner["policy_epochs"])):
        for item in prepared:
            optimizer.zero_grad(set_to_none=True)
            examples = _optimizer_examples(item)
            for chunk in _chunks(examples, microbatch_size):
                batch = _move_batch(
                    collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id),
                    device,
                )
                replay, entropy = _forward(model, batch)
                result = hierarchical_clipped_policy_loss(
                    replay,
                    batch,
                    clip_epsilon=float(learner["clip_epsilon"]),
                    entropy=entropy,
                )
                _require(bool(torch.isfinite(result["loss"])), "M5 formal learner loss is non-finite")
                result["loss"].backward()
                losses.append(float(result["loss"].detach().cpu()))
                metric_tokens += int(result["token_count"])
            gradient = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(learner["gradient_clip"]),
            )
            _require(bool(torch.isfinite(gradient)), "M5 formal learner gradient is non-finite")
            gradients.append(float(gradient.detach().cpu()))
            optimizer.step()
            updates += 1

    adapter = staging / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    view = staging / "rollout_adapter"
    view_audit = build_vllm_adapter_view(source_adapter=adapter, destination=view, base_model=base_model)
    if updates:
        _require(
            view_audit["semantic_tensor_sha256"] != input_adapter_semantic_sha256,
            "M5 formal optimizer reported updates without a parameter change",
        )
    else:
        _require(
            view_audit["semantic_tensor_sha256"] == input_adapter_semantic_sha256,
            "M5 zero-signal iteration changed policy parameters",
        )
    cumulative_updates_after = cumulative_updates_before + updates
    optimizer_path = staging / "optimizer.pt"
    torch.save(
        {
            "schema_version": FORMAL_OPTIMIZER_SCHEMA,
            "method": method,
            "seed": seed,
            "iteration_index": iteration_index,
            "git_sha": git_sha,
            "protocol_sha256": protocol_sha256,
            "adapter_sha256": directory_sha256(adapter),
            "iteration_optimizer_updates": updates,
            "cumulative_optimizer_updates": cumulative_updates_after,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        optimizer_path,
    )
    report = {
        "schema_version": FORMAL_ITERATION_LEARNER_SCHEMA,
        "complete": True,
        "formal_training": True,
        "method": method,
        "seed": seed,
        "iteration_index": iteration_index,
        "git_sha": git_sha,
        "protocol_sha256": protocol_sha256,
        "group_count": len(groups),
        "nonzero_group_count": len(prepared),
        "iteration_optimizer_updates": updates,
        "cumulative_optimizer_updates_before": cumulative_updates_before,
        "cumulative_optimizer_updates_after": cumulative_updates_after,
        "iteration_generated_action_tokens": iteration_generated_action_tokens,
        "effective_optimizer_action_tokens": effective_tokens,
        "effective_optimizer_action_token_fraction": effective_fraction,
        "optimizer_evaluated_action_tokens": metric_tokens,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "maximum_absolute_loss": max(abs(value) for value in losses) if losses else None,
        "mean_gradient_norm": sum(gradients) / len(gradients) if gradients else None,
        "maximum_gradient_norm": max(gradients) if gradients else None,
        "initial_replay_parity": parity,
        "input_adapter": str(Path(initial_adapter).expanduser().resolve()),
        "input_adapter_sha256": directory_sha256(initial_adapter),
        "input_adapter_semantic_sha256": input_adapter_semantic_sha256,
        "input_optimizer": str(Path(input_optimizer).expanduser().resolve()) if input_optimizer else None,
        "input_optimizer_sha256": input_optimizer_sha256,
        "base_model": str(Path(base_model).expanduser().resolve()),
        "output_adapter": str(parent / "adapter"),
        "output_adapter_sha256": directory_sha256(adapter),
        "output_adapter_semantic_sha256": view_audit["semantic_tensor_sha256"],
        "output_rollout_adapter": str(parent / "rollout_adapter"),
        "output_rollout_adapter_sha256": view_audit["view_directory_sha256"],
        "output_optimizer": str(parent / "optimizer.pt"),
        "output_optimizer_sha256": sha256_file(optimizer_path),
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(staging / "learner_report.json", report)
    os.rename(staging, parent)
    descriptor = os.open(parent.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    del model, optimizer
    torch.cuda.empty_cache()
    return report


def validate_formal_iteration_learner_report(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        report.get("schema_version") == FORMAL_ITERATION_LEARNER_SCHEMA
        and report.get("complete") is True
        and report.get("formal_training") is True,
        "M5 formal learner report schema drift",
    )
    _require(report.get("content_sha256") == _self_hash(report), "M5 formal learner report self-hash drift")
    _require(directory_sha256(Path(report["output_adapter"])) == report["output_adapter_sha256"], "M5 formal learner adapter hash drift")
    from ..long_horizon_rl.adapter_view import validate_vllm_adapter_view

    rollout = validate_vllm_adapter_view(
        source_adapter=Path(report["output_adapter"]),
        view_directory=Path(report["output_rollout_adapter"]),
        base_model=Path(report["base_model"]),
    )
    _require(rollout["view_directory_sha256"] == report["output_rollout_adapter_sha256"], "M5 formal learner rollout adapter hash drift")
    _require(rollout["semantic_tensor_sha256"] == report["output_adapter_semantic_sha256"], "M5 formal learner semantic hash drift")
    _require(sha256_file(Path(report["output_optimizer"])) == report["output_optimizer_sha256"], "M5 formal learner optimizer hash drift")
    _require(
        report["cumulative_optimizer_updates_after"]
        == report["cumulative_optimizer_updates_before"] + report["iteration_optimizer_updates"],
        "M5 formal learner cumulative update drift",
    )
    for field in ("mean_loss", "maximum_absolute_loss", "mean_gradient_norm", "maximum_gradient_norm"):
        value = report[field]
        _require(value is None or (isinstance(value, (int, float)) and math.isfinite(float(value))), f"M5 formal learner non-finite {field}")
    return report
