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

GROUP_SCHEMA = "m5_webshop_k4_group_v1"
LEARNER_REPORT_SCHEMA = "m5_webshop_online_learner_report_v1"
OPTIMIZER_SCHEMA = "m5_webshop_online_optimizer_v1"
METHODS = (BASELINE_METHOD, ANCHOR_METHOD)
MAX_SEQUENCE_TOKENS = 8192
MAX_NEW_TOKENS = 128


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
    reward = episode.get("reward")
    _require(isinstance(reward, (int, float)) and not isinstance(reward, bool) and float(reward) in (0.0, 1.0), "M5 reward is not binary")
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
        "reward": float(reward),
        "success": bool(episode.get("success", False)),
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
        _require(isinstance(reward, (int, float)) and float(reward) in (0.0, 1.0), "M5 group reward drift")
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
        "success_rate": sum(rewards) / len(rewards),
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
    behavior = [value for item in prepared for example in item["examples"] for value in example.behavior_logprobs]
    sampling = [value for item in prepared for example in item["examples"] for value in example.sampling_logprobs]
    sampling_report = summarize_logprob_parity(behavior, sampling)
    _require(sampling_report["maximum_absolute_logprob_difference"] <= thresholds["behavior_sampling_maximum_absolute_difference"], "M5 behavior/sampling parity failed")
    examples = [example for item in prepared for example in item["examples"]]
    replay = _replay_logprobs(model, examples, tokenizer, device, microbatch_size)
    report = summarize_logprob_parity(behavior, replay)
    checks = {
        "mean": report["mean_absolute_logprob_difference"] <= thresholds["replay_mean_absolute_difference"],
        "p95": report["p95_absolute_logprob_difference"] <= thresholds["replay_p95_absolute_difference"],
        "p99": report["p99_absolute_logprob_difference"] <= thresholds["replay_p99_absolute_difference"],
        "p999": report["p999_absolute_logprob_difference"] <= thresholds["replay_p999_absolute_difference"],
        "clip_fraction": report["initial_ratio_clip_fraction"] <= thresholds["replay_initial_ratio_clip_fraction"],
        "mean_ratio": abs(report["mean_importance_ratio"] - 1.0) <= thresholds["mean_importance_ratio_absolute_deviation"],
    }
    _require(all(checks.values()), f"M5 behavior/replay parity failed: {checks}")
    return {**report, "behavior_sampling": sampling_report, "checks": checks, "thresholds": dict(thresholds), "passed": True}


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
            examples = sorted(item["examples"], key=lambda example: (example.forward_tokens, example.trajectory_index, example.turn_index))
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
