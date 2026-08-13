"""M6 K=8 strict-GRPO learner and auditable rollout contracts.

The module is intentionally independent from the M5 K=4 schemas.  Generic
tensor collation and replay are reused, while group size, terminal reward,
verifier-TD evidence and the frozen-SFT KL reference remain M6-owned.
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
from ..long_horizon_rl.contracts import (
    SHA256_PATTERN,
    atomic_write_json,
    directory_sha256,
    sha256_file,
    sha256_json,
    token_ids_sha256,
)
from ..m6_posttraining_protocol import (
    audit_behavior_sampling_logprobs,
    load_protocol,
    validate_protocol,
)
from .credit import policy_context_signature, public_state_anchor_signature
from .verifier_td import (
    ANCHOR_METHOD,
    BASELINE_METHOD,
    FORMULA_VERSION_BY_METHOD,
    GROUP_SIZE,
    METHODS,
    assign_group_credit,
    strict_terminal_reward,
    validate_credit_assignment,
)

GROUP_SCHEMA = "m6_webshop_group_v1"
LEARNER_SCHEMA = "m6_webshop_mini_rl_iteration_v1"
OPTIMIZER_SCHEMA = "m6_webshop_mini_rl_optimizer_v1"
MAX_SEQUENCE_TOKENS = 8192
REFERENCE_ADAPTER_NAME = "m6_frozen_sft_reference"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def trajectory_from_episode(
    episode: Mapping[str, Any],
    *,
    trajectory_id: str,
    rollout_index: int,
    adapter_sha256: str,
    rollout_adapter_sha256: str,
    adapter_semantic_sha256: str,
) -> dict[str, Any]:
    """Convert an annotated episode into exact M6 replay evidence."""

    _require(episode.get("rollout_valid") is True, "cannot commit an infrastructure-invalid M6 trajectory")
    task_score = episode.get("task_score")
    _require(
        isinstance(task_score, (int, float))
        and not isinstance(task_score, bool)
        and math.isfinite(float(task_score))
        and 0.0 <= float(task_score) <= 1.0,
        "M6 official task score is invalid",
    )
    binary = strict_terminal_reward(task_score)
    _require(episode.get("success") is bool(binary), "M6 episode success/task-score disagreement")
    source_turns = episode.get("turns")
    _require(isinstance(source_turns, list) and source_turns, "M6 trajectory has no generated turns")
    turns: list[dict[str, Any]] = []
    for index, source in enumerate(source_turns, start=1):
        _require(isinstance(source, Mapping), "M6 generated turn is malformed")
        prompt_ids = [int(value) for value in source.get("prompt_token_ids", [])]
        generated_ids = [int(value) for value in source.get("generated_token_ids", [])]
        behavior = [float(value) for value in source.get("token_logprobs", [])]
        sampling = [float(value) for value in source.get("sampling_logprobs", [])]
        _require(prompt_ids and generated_ids, "M6 generated turn has empty token evidence")
        _require(len(generated_ids) == len(behavior) == len(sampling), "M6 logprob/token length drift")
        _require(all(math.isfinite(value) for value in behavior + sampling), "M6 logprob is non-finite")
        _require(len(prompt_ids) + len(generated_ids) <= MAX_SEQUENCE_TOKENS, "M6 replay exceeds 8192 tokens")
        observation = source.get("observation")
        post = source.get("post_action_observation")
        evidence = source.get("verifier_progress_evidence")
        _require(isinstance(observation, Mapping), "M6 turn lacks pre-action public observation")
        _require(isinstance(post, Mapping), "M6 turn lacks post-action public observation")
        _require(isinstance(evidence, Mapping), "M6 turn lacks verifier progress evidence")
        _require(evidence.get("content_sha256") == _self_hash(evidence), "M6 verifier evidence self-hash drift")
        _require(
            evidence.get("public_state_sha256") == public_state_anchor_signature(post),
            "M6 verifier post-action state hash drift",
        )
        progress = evidence.get("potential")
        _require(
            isinstance(progress, (int, float))
            and not isinstance(progress, bool)
            and math.isfinite(float(progress)),
            "M6 verifier progress is invalid",
        )
        _require(source.get("adapter_sha256") == adapter_sha256, "M6 behavior adapter drift")
        _require(source.get("rollout_adapter_sha256") == rollout_adapter_sha256, "M6 rollout adapter drift")
        _require(source.get("adapter_semantic_sha256") == adapter_semantic_sha256, "M6 semantic adapter drift")
        turns.append(
            {
                "turn_index": index,
                "pre_action_public_state_sha256": public_state_anchor_signature(observation),
                "post_action_public_state_sha256": public_state_anchor_signature(post),
                "verifier_public_state_sha256": evidence["public_state_sha256"],
                "verifier_progress_after_action": float(progress),
                "verifier_progress_evidence": dict(evidence),
                "policy_context_token_sha256": policy_context_signature(prompt_ids),
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "generated_token_sha256": token_ids_sha256(generated_ids),
                "behavior_logprobs": behavior,
                "sampling_logprobs": sampling,
                "rendered_prompt_sha256": str(source.get("rendered_prompt_sha256", "")),
                "request_id": str(source.get("request_id", "")),
                "sampling_seed": int(source.get("sampling_seed", 0)),
                "generation_backend": str(source.get("generation_backend", "")),
                "schema_valid": bool(source.get("schema_valid", False)),
                "action": source.get("action"),
                "raw_output": str(source.get("raw_output", ""))[:4096],
                "action_result": source.get("action_result"),
                "observation": dict(observation),
                "post_action_observation": dict(post),
            }
        )
    return {
        "trajectory_id": trajectory_id,
        "rollout_index": rollout_index,
        "task_id": str(episode.get("task_id", "")),
        "task_score": float(task_score),
        "reward": binary,
        "binary_reward": binary,
        "success": bool(binary),
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
    expected_k: int,
    git_sha: str,
    protocol_sha256: str,
    adapter_sha256: str,
    rollout_adapter_sha256: str,
    adapter_semantic_sha256: str,
    training_updates_allowed: bool,
) -> dict[str, Any]:
    _require(expected_k in {4, GROUP_SIZE}, "M6 group K must be evaluation K4 or training K8")
    _require(len(trajectories) == expected_k, "M6 group trajectory count drift")
    payload = {
        "schema_version": GROUP_SCHEMA,
        "complete": True,
        "development_only": True,
        "training_updates_allowed": bool(training_updates_allowed),
        "git_sha": git_sha,
        "protocol_sha256": protocol_sha256,
        "task_id": task_id,
        "group_id": group_id,
        "attempt_index": attempt_index,
        "K": expected_k,
        "adapter_sha256": adapter_sha256,
        "rollout_adapter_sha256": rollout_adapter_sha256,
        "adapter_semantic_sha256": adapter_semantic_sha256,
        "generated_action_tokens": sum(int(item["generated_action_tokens"]) for item in trajectories),
        "trajectories": [dict(item) for item in trajectories],
    }
    payload["content_sha256"] = _self_hash(payload)
    return validate_committed_group(payload, require_k=expected_k)


def validate_committed_group(
    group: Mapping[str, Any],
    *,
    require_k: int | None = None,
) -> dict[str, Any]:
    value = dict(group)
    _require(value.get("schema_version") == GROUP_SCHEMA and value.get("complete") is True, "M6 group schema drift")
    _require(value.get("development_only") is True, "M6 mini group lost development-only marker")
    _require(isinstance(value.get("training_updates_allowed"), bool), "M6 group training-purpose flag drift")
    _require(isinstance(value.get("git_sha"), str) and len(value["git_sha"]) == 40, "M6 group Git SHA drift")
    _require(value.get("content_sha256") == _self_hash(value), "M6 group self-hash drift")
    for field in ("protocol_sha256", "adapter_sha256", "rollout_adapter_sha256", "adapter_semantic_sha256"):
        _require(SHA256_PATTERN.fullmatch(str(value.get(field, ""))) is not None, f"M6 group {field} drift")
    k = value.get("K")
    _require(k in {4, GROUP_SIZE} and (require_k is None or k == require_k), "M6 group K drift")
    trajectories = value.get("trajectories")
    _require(isinstance(trajectories, list) and len(trajectories) == k, "M6 group trajectory roster drift")
    task_id = value.get("task_id")
    generated = 0
    rollout_indices: set[int] = set()
    for trajectory in trajectories:
        _require(isinstance(trajectory, Mapping) and trajectory.get("task_id") == task_id, "M6 group crosses tasks")
        binary = strict_terminal_reward(trajectory.get("task_score"))
        _require(
            trajectory.get("reward") == binary
            and trajectory.get("binary_reward") == binary
            and trajectory.get("success") is bool(binary),
            "M6 strict terminal fields drift",
        )
        turns = trajectory.get("turns")
        _require(isinstance(turns, list) and turns, "M6 group trajectory has no turns")
        rollout_indices.add(int(trajectory.get("rollout_index", -1)))
        trajectory_tokens = 0
        for turn_index, turn in enumerate(turns, start=1):
            _require(turn.get("turn_index") == turn_index, "M6 turn order drift")
            prompt_ids = turn.get("prompt_token_ids")
            generated_ids = turn.get("generated_token_ids")
            behavior = turn.get("behavior_logprobs")
            sampling = turn.get("sampling_logprobs")
            _require(isinstance(prompt_ids, list) and prompt_ids and isinstance(generated_ids, list) and generated_ids, "M6 turn token evidence drift")
            _require(len(generated_ids) == len(behavior) == len(sampling), "M6 turn logprob length drift")
            _require(
                all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in [*prompt_ids, *generated_ids]),
                "M6 turn token id drift",
            )
            _require(
                all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in [*behavior, *sampling]),
                "M6 turn logprob value drift",
            )
            _require(len(prompt_ids) + len(generated_ids) <= MAX_SEQUENCE_TOKENS, "M6 turn context overflow")
            _require(token_ids_sha256(generated_ids) == turn.get("generated_token_sha256"), "M6 generated token hash drift")
            _require(policy_context_signature(prompt_ids) == turn.get("policy_context_token_sha256"), "M6 prompt token hash drift")
            _require(public_state_anchor_signature(turn["observation"]) == turn.get("pre_action_public_state_sha256"), "M6 pre-action state hash drift")
            _require(public_state_anchor_signature(turn["post_action_observation"]) == turn.get("post_action_public_state_sha256"), "M6 post-action state hash drift")
            evidence = turn.get("verifier_progress_evidence")
            _require(isinstance(evidence, Mapping) and evidence.get("content_sha256") == _self_hash(evidence), "M6 verifier evidence drift")
            _require(evidence.get("public_state_sha256") == turn.get("post_action_public_state_sha256"), "M6 verifier state binding drift")
            _require(float(evidence.get("potential")) == float(turn.get("verifier_progress_after_action")), "M6 verifier progress binding drift")
            trajectory_tokens += len(generated_ids)
        _require(trajectory_tokens == trajectory.get("generated_action_tokens"), "M6 trajectory token ledger drift")
        generated += trajectory_tokens
    _require(rollout_indices == set(range(k)), "M6 rollout index roster drift")
    _require(generated == value.get("generated_action_tokens"), "M6 group token ledger drift")
    if k == GROUP_SIZE:
        validate_credit_assignment(assign_group_credit(trajectories))
    return value


@dataclass(frozen=True)
class M6TurnTrainingExample:
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


def prepare_group_training_examples(
    group: Mapping[str, Any],
    method: str = ANCHOR_METHOD,
) -> dict[str, Any]:
    validated = validate_committed_group(group, require_k=GROUP_SIZE)
    _require(method in METHODS, "unsupported M6 mini RL method")
    credit = assign_group_credit(validated["trajectories"], method=method)
    examples: list[M6TurnTrainingExample] = []
    effective = 0
    for trajectory_index, (trajectory, trajectory_credit) in enumerate(
        zip(validated["trajectories"], credit["turn_credit"])
    ):
        _require(len(trajectory["turns"]) == len(trajectory_credit), "M6 credit/turn count drift")
        for turn, assigned in zip(trajectory["turns"], trajectory_credit):
            example = M6TurnTrainingExample(
                group_id=validated["group_id"],
                trajectory_id=trajectory["trajectory_id"],
                trajectory_index=trajectory_index,
                turn_index=turn["turn_index"],
                turns_in_trajectory=len(trajectory["turns"]),
                prompt_token_ids=tuple(turn["prompt_token_ids"]),
                generated_token_ids=tuple(turn["generated_token_ids"]),
                behavior_logprobs=tuple(turn["behavior_logprobs"]),
                sampling_logprobs=tuple(turn["sampling_logprobs"]),
                advantage=float(assigned["turn_advantage"]),
            )
            examples.append(example)
            effective += example.completion_tokens if abs(example.advantage) > 0 else 0
    weight = sum(example.token_loss_weight * example.completion_tokens for example in examples)
    _require(math.isclose(weight, 1.0, abs_tol=1e-12), "M6 hierarchical K8 weights do not sum to one")
    return {
        "group": validated,
        "credit": credit,
        "examples": tuple(examples),
        "generated_action_tokens": validated["generated_action_tokens"],
        "effective_optimizer_action_tokens": effective,
        "zero_advantage_group": effective == 0,
        "hierarchical_weight_sum": weight,
    }


def strict_grpo_kl_loss(
    replay_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    clip_epsilon: float,
    kl_coefficient: float,
) -> dict[str, Any]:
    """Clipped K8 policy loss plus a non-negative k3 reference-KL estimate."""

    # Keep torch and the shared learner runtime out of module import.  Group,
    # curriculum and final-audit validation are CPU-only protocol work and
    # must not require the GPU training dependency merely to parse artifacts.
    import torch

    _require(0.0 < clip_epsilon < 1.0, "M6 clip epsilon is invalid")
    _require(math.isfinite(kl_coefficient) and kl_coefficient > 0.0, "M6 KL coefficient is invalid")
    old = batch["behavior_logprobs"].to(replay_logprobs.device)
    mask = batch["completion_mask"].to(replay_logprobs.device)
    advantages = batch["advantages"].to(replay_logprobs.device).unsqueeze(1)
    weights = batch["token_loss_weights"].to(replay_logprobs.device).unsqueeze(1)
    _require(replay_logprobs.shape == reference_logprobs.shape == old.shape == mask.shape, "M6 learner tensor shape drift")
    log_ratio = replay_logprobs - old
    ratio = torch.exp(log_ratio)
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_objective = torch.minimum(ratio * advantages, clipped * advantages)
    policy_loss = -(policy_objective * weights * mask).sum()
    reference_log_ratio = reference_logprobs - replay_logprobs
    k3 = torch.exp(reference_log_ratio) - reference_log_ratio - 1.0
    reference_kl = (k3 * weights * mask).sum()
    total = policy_loss + kl_coefficient * reference_kl
    selected = mask
    count = int(selected.sum().item())
    _require(count > 0, "M6 learner loss contains no completion tokens")
    for tensor, label in ((policy_loss, "policy loss"), (reference_kl, "reference KL"), (total, "total loss")):
        _require(bool(torch.isfinite(tensor)), f"M6 {label} is non-finite")
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "reference_kl": reference_kl,
        "mean_token_reference_kl": k3[selected].mean(),
        "token_count": count,
        "clip_fraction": ((ratio != clipped) & selected).float().sum() / count,
        "maximum_absolute_log_ratio": log_ratio[selected].abs().max(),
    }


def update_adaptive_kl_coefficient(
    coefficient: float,
    observed_kl: float,
    *,
    target_minimum: float,
    target_maximum: float,
) -> float:
    _require(0.0 < target_minimum < target_maximum, "M6 adaptive KL target is invalid")
    _require(math.isfinite(coefficient) and coefficient > 0.0, "M6 adaptive KL coefficient is invalid")
    _require(math.isfinite(observed_kl) and observed_kl >= 0.0, "M6 observed KL is invalid")
    if observed_kl < target_minimum:
        return max(1e-5, coefficient / 2.0)
    if observed_kl > target_maximum:
        return min(10.0, coefficient * 2.0)
    return coefficient


def audit_collection(
    groups: Sequence[Mapping[str, Any]],
    *,
    all_generated_action_tokens: int,
    method: str = ANCHOR_METHOD,
) -> dict[str, Any]:
    _require(bool(groups), "M6 collection audit requires K8 groups")
    prepared = [prepare_group_training_examples(group, method=method) for group in groups]
    credits = [item["credit"] for item in prepared]
    committed = sum(item["generated_action_tokens"] for item in prepared)
    _require(all_generated_action_tokens >= committed > 0, "M6 collection token ledger excludes committed tokens")
    turns = sum(int(item["metrics"]["turn_count"]) for item in credits)
    nonzero_td = sum(int(item["metrics"]["nonzero_td_turn_count"]) for item in credits)
    nonzero_optimizer = sum(
        int(item["metrics"]["nonzero_optimizer_turn_count"]) for item in credits
    )
    mixed = sum(bool(item["metrics"]["mixed_strict_reward_signal"]) for item in credits)
    return {
        "group_count": len(groups),
        "trajectory_count": len(groups) * GROUP_SIZE,
        "mixed_strict_reward_group_count": mixed,
        "mixed_strict_reward_group_fraction": mixed / len(groups),
        "turn_count": turns,
        "method": method,
        "credit_formula_version": FORMULA_VERSION_BY_METHOD[method],
        "nonzero_td_turn_count": nonzero_td,
        "nonzero_td_turn_fraction": nonzero_td / turns,
        "nonzero_optimizer_turn_count": nonzero_optimizer,
        "nonzero_optimizer_turn_fraction": nonzero_optimizer / turns,
        "committed_group_action_tokens": committed,
        "all_generated_action_tokens": all_generated_action_tokens,
        "credit_assignment_content_sha256": [item["content_sha256"] for item in credits],
    }


def validate_replay_parity(
    parity: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, bool]:
    """Apply every frozen vLLM-to-HF replay threshold, never one headline max."""

    checks = replay_parity_checks(parity, contract=contract)
    if not all(checks.values()):
        failure = {
            "parity": dict(parity),
            "thresholds": dict(contract),
            "checks": checks,
        }
        raise ValueError(
            "M6 initial behavior/HF replay parity failed: "
            f"{json.dumps(failure, sort_keys=True)}"
        )
    return checks


def replay_parity_checks(
    parity: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, bool]:
    """Return the complete frozen threshold decision without hiding metrics."""

    checks = {
        "mean_absolute_difference": float(parity["mean_absolute_logprob_difference"])
        <= float(contract["replay_mean_absolute_difference"]),
        "p95_absolute_difference": float(parity["p95_absolute_logprob_difference"])
        <= float(contract["replay_p95_absolute_difference"]),
        "p99_absolute_difference": float(parity["p99_absolute_logprob_difference"])
        <= float(contract["replay_p99_absolute_difference"]),
        "p999_absolute_difference": float(parity["p999_absolute_logprob_difference"])
        <= float(contract["replay_p999_absolute_difference"]),
        "initial_ratio_clip_fraction": float(parity["initial_ratio_clip_fraction"])
        <= float(contract["replay_initial_ratio_clip_fraction"]),
        "mean_importance_ratio": abs(float(parity["mean_importance_ratio"]) - 1.0)
        <= float(contract["mean_importance_ratio_absolute_deviation"]),
    }
    return checks


def _chunks(values: Sequence[Any], size: int):
    _require(size > 0, "M6 learner microbatch must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
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


def _forward_logprobs(model: Any, batch: Mapping[str, Any]) -> torch.Tensor:
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


def _optimizer_examples(item: Mapping[str, Any]) -> list[M6TurnTrainingExample]:
    return sorted(
        item["examples"],
        key=lambda example: (example.forward_tokens, example.trajectory_index, example.turn_index),
    )


def train_mini_policy_iteration(
    *,
    groups: Sequence[Mapping[str, Any]],
    all_generated_action_tokens: int,
    base_model: Path,
    input_adapter: Path,
    input_adapter_semantic_sha256: str,
    reference_sft_adapter: Path,
    input_optimizer: Path | None,
    output_root: Path,
    git_sha: str,
    protocol_sha256: str,
    iteration_index: int,
    seed: int,
    method: str = ANCHOR_METHOD,
    pilot_authorization_sha256: str | None = None,
    microbatch_size: int = 4,
) -> dict[str, Any]:
    """Apply one recoverable M6-mini update from same-policy K8 groups."""

    import torch

    from ..long_horizon_rl.learner import (
        collate_turn_training_examples,
        load_trainable_policy_model,
        summarize_logprob_parity,
    )

    contract = load_protocol()["payload"]
    validate_protocol(contract)
    _require(groups and all_generated_action_tokens > 0, "M6 mini learner input is empty")
    _require(iteration_index >= 0 and seed >= 0, "M6 mini learner identity drift")
    _require(method in METHODS, "unsupported M6 mini RL method")
    _require(microbatch_size in {1, 2, 4, 8}, "M6 mini learner microbatch drift")
    prepared_all = [prepare_group_training_examples(group, method=method) for group in groups]
    prepared = [item for item in prepared_all if not item["zero_advantage_group"]]
    _require(prepared, "M6 mini learner iteration has no nonzero policy credit")
    behavior_sampling_parity = audit_behavior_sampling_logprobs(
        [
            value
            for item in prepared_all
            for example in item["examples"]
            for value in example.behavior_logprobs
        ],
        [
            value
            for item in prepared_all
            for example in item["examples"]
            for value in example.sampling_logprobs
        ],
        maximum_absolute_difference=float(
            contract["rl"]["parity_contract"][
                "behavior_sampling_maximum_absolute_difference"
            ]
        ),
    )
    effective_tokens = sum(item["effective_optimizer_action_tokens"] for item in prepared)
    root = Path(output_root).expanduser().resolve()
    report_path = root / "learner_report.json"
    if report_path.is_file():
        report = validate_learner_report(report_path, expected_method=method)
        _require(report.get("git_sha") == git_sha, "M6 recovered learner Git drift")
        _require(report.get("protocol_sha256") == protocol_sha256, "M6 recovered learner protocol drift")
        _require(report.get("iteration_index") == iteration_index and report.get("seed") == seed, "M6 recovered learner identity drift")
        _require(report.get("input_adapter_sha256") == directory_sha256(input_adapter), "M6 recovered learner input-adapter drift")
        _require(report.get("input_adapter_semantic_sha256") == input_adapter_semantic_sha256, "M6 recovered learner semantic input drift")
        _require(report.get("reference_sft_adapter_sha256") == directory_sha256(reference_sft_adapter), "M6 recovered learner reference drift")
        _require(report.get("pilot_authorization_sha256") == pilot_authorization_sha256, "M6 recovered learner pilot authorization drift")
        expected_optimizer_sha = sha256_file(input_optimizer) if input_optimizer is not None else None
        _require(report.get("input_optimizer_sha256") == expected_optimizer_sha, "M6 recovered learner optimizer drift")
        return report
    root.parent.mkdir(parents=True, exist_ok=True)
    interrupted = root.parent / "interrupted_learner_stages"
    stale = ([root] if root.exists() else []) + sorted(root.parent.glob(f".{root.name}.staging-*"))
    for index, path in enumerate(stale):
        interrupted.mkdir(exist_ok=True)
        os.rename(path, interrupted / f"{root.name}_{time.time_ns()}_{index}")
    staging = root.parent / f".{root.name}.staging-{os.getpid()}-{time.time_ns()}"
    staging.mkdir()
    device = torch.device("cuda:0")
    torch.manual_seed(seed + iteration_index)
    torch.cuda.manual_seed_all(seed + iteration_index)
    model, tokenizer = load_trainable_policy_model(base_model=base_model, adapter_path=input_adapter)
    active_adapter = getattr(model, "active_adapters", None)
    active_adapter = active_adapter() if callable(active_adapter) else active_adapter
    if isinstance(active_adapter, (list, tuple)):
        _require(len(active_adapter) == 1, "M6 learner requires one active policy adapter")
        current_adapter_name = str(active_adapter[0])
    else:
        current_adapter_name = str(getattr(model, "active_adapter", "default"))
    _require(current_adapter_name and not current_adapter_name.startswith("["), "M6 active adapter identity drift")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(trainable_parameters, "M6 mini learner has no trainable parameters")
    model.load_adapter(
        str(Path(reference_sft_adapter).expanduser().resolve()),
        adapter_name=REFERENCE_ADAPTER_NAME,
        is_trainable=False,
    )
    for name, parameter in model.named_parameters():
        if REFERENCE_ADAPTER_NAME in name:
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(contract["rl"]["learning_rate"]),
        weight_decay=0.0,
    )
    cumulative_updates_before = 0
    kl_coefficient = 0.03
    input_optimizer_sha256 = None
    if input_optimizer is not None:
        checkpoint_path = Path(input_optimizer).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _require(checkpoint.get("schema_version") == OPTIMIZER_SCHEMA, "M6 mini optimizer schema drift")
        _require(checkpoint.get("method") == method, "M6 mini optimizer method drift")
        _require(
            checkpoint.get("credit_formula_version") == FORMULA_VERSION_BY_METHOD[method],
            "M6 mini optimizer credit-formula drift",
        )
        _require(
            checkpoint.get("pilot_authorization_sha256") == pilot_authorization_sha256,
            "M6 mini optimizer pilot-authorization drift",
        )
        _require(checkpoint.get("seed") == seed, "M6 mini optimizer seed drift")
        _require(
            checkpoint.get("iteration_index") == iteration_index - 1,
            "M6 mini optimizer iteration lineage drift",
        )
        _require(checkpoint.get("protocol_sha256") == protocol_sha256, "M6 mini optimizer protocol drift")
        _require(checkpoint.get("adapter_sha256") == directory_sha256(input_adapter), "M6 mini optimizer/adapter drift")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cumulative_updates_before = int(checkpoint["cumulative_optimizer_updates"])
        kl_coefficient = float(checkpoint["adaptive_kl_coefficient"])
        input_optimizer_sha256 = sha256_file(checkpoint_path)

    behavior: list[float] = []
    replay_before: list[float] = []
    model.set_adapter(current_adapter_name)
    model.eval()
    with torch.inference_mode():
        for item in prepared:
            for chunk in _chunks(_optimizer_examples(item), microbatch_size):
                batch = _move_batch(collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
                replay = _forward_logprobs(model, batch)
                for row, example in enumerate(chunk):
                    behavior.extend(example.behavior_logprobs)
                    replay_before.extend(float(value) for value in replay[row, : example.completion_tokens].cpu().tolist())
    parity = summarize_logprob_parity(behavior, replay_before)
    parity_checks = validate_replay_parity(
        parity,
        contract=contract["rl"]["parity_contract"],
    )

    losses: list[float] = []
    policy_losses: list[float] = []
    observed_kl_sums: list[float] = []
    gradients: list[float] = []
    updates = 0
    metric_tokens = 0
    coefficient_before = kl_coefficient
    for item in prepared:
        optimizer.zero_grad(set_to_none=True)
        for chunk in _chunks(_optimizer_examples(item), microbatch_size):
            batch = _move_batch(collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
            model.set_adapter(REFERENCE_ADAPTER_NAME)
            model.eval()
            with torch.inference_mode():
                reference = _forward_logprobs(model, batch)
            model.set_adapter(current_adapter_name)
            model.train()
            replay = _forward_logprobs(model, batch)
            result = strict_grpo_kl_loss(
                replay,
                reference,
                batch,
                clip_epsilon=0.2,
                kl_coefficient=kl_coefficient,
            )
            result["loss"].backward()
            losses.append(float(result["loss"].detach().cpu()))
            policy_losses.append(float(result["policy_loss"].detach().cpu()))
            observed_kl_sums.append(
                float(result["mean_token_reference_kl"].detach().cpu())
                * int(result["token_count"])
            )
            metric_tokens += int(result["token_count"])
        gradient = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        _require(bool(torch.isfinite(gradient)), "M6 mini learner gradient is non-finite")
        gradients.append(float(gradient.detach().cpu()))
        optimizer.step()
        updates += 1
    observed_kl = sum(observed_kl_sums) / metric_tokens
    kl_coefficient = update_adaptive_kl_coefficient(
        kl_coefficient,
        observed_kl,
        target_minimum=float(contract["rl"]["adaptive_kl_target_minimum"]),
        target_maximum=float(contract["rl"]["adaptive_kl_target_maximum"]),
    )
    cumulative_updates_after = cumulative_updates_before + updates
    model.set_adapter(current_adapter_name)
    if hasattr(model, "delete_adapter"):
        model.delete_adapter(REFERENCE_ADAPTER_NAME)
    adapter = staging / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    view = staging / "rollout_adapter"
    view_audit = build_vllm_adapter_view(source_adapter=adapter, destination=view, base_model=base_model)
    _require(view_audit["semantic_tensor_sha256"] != input_adapter_semantic_sha256, "M6 optimizer updated without a parameter change")
    optimizer_path = staging / "optimizer.pt"
    torch.save(
        {
            "schema_version": OPTIMIZER_SCHEMA,
            "method": method,
            "credit_formula_version": FORMULA_VERSION_BY_METHOD[method],
            "pilot_authorization_sha256": pilot_authorization_sha256,
            "seed": seed,
            "iteration_index": iteration_index,
            "git_sha": git_sha,
            "protocol_sha256": protocol_sha256,
            "adapter_sha256": directory_sha256(adapter),
            "iteration_optimizer_updates": updates,
            "cumulative_optimizer_updates": cumulative_updates_after,
            "adaptive_kl_coefficient": kl_coefficient,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        optimizer_path,
    )
    collection = audit_collection(
        groups,
        all_generated_action_tokens=all_generated_action_tokens,
        method=method,
    )
    report = {
        "schema_version": LEARNER_SCHEMA,
        "complete": True,
        "development_only": True,
        "formal_checkpoint_reusable": False,
        "method": method,
        "credit_formula_version": FORMULA_VERSION_BY_METHOD[method],
        "verifier_td_lambda": 0.0 if method == BASELINE_METHOD else 0.5,
        "pilot_authorization_sha256": pilot_authorization_sha256,
        "group_size": GROUP_SIZE,
        "seed": seed,
        "iteration_index": iteration_index,
        "git_sha": git_sha,
        "protocol_sha256": protocol_sha256,
        "group_count": len(groups),
        "nonzero_group_count": len(prepared),
        "learner_microbatch_size": microbatch_size,
        "iteration_optimizer_updates": updates,
        "cumulative_optimizer_updates_before": cumulative_updates_before,
        "cumulative_optimizer_updates_after": cumulative_updates_after,
        "all_generated_action_tokens": all_generated_action_tokens,
        "effective_optimizer_action_tokens": effective_tokens,
        "optimizer_evaluated_action_tokens": metric_tokens,
        "mean_loss": sum(losses) / len(losses),
        "mean_policy_loss": sum(policy_losses) / len(policy_losses),
        "mean_gradient_norm": sum(gradients) / len(gradients),
        "observed_reference_kl": observed_kl,
        "adaptive_kl_coefficient_before": coefficient_before,
        "adaptive_kl_coefficient_after": kl_coefficient,
        "initial_replay_parity": parity,
        "initial_replay_parity_checks": parity_checks,
        "behavior_sampling_parity": behavior_sampling_parity,
        "collection_audit": collection,
        "input_adapter": str(Path(input_adapter).expanduser().resolve()),
        "input_adapter_sha256": directory_sha256(input_adapter),
        "input_adapter_semantic_sha256": input_adapter_semantic_sha256,
        "reference_sft_adapter": str(Path(reference_sft_adapter).expanduser().resolve()),
        "reference_sft_adapter_sha256": directory_sha256(reference_sft_adapter),
        "input_optimizer": str(Path(input_optimizer).expanduser().resolve()) if input_optimizer else None,
        "input_optimizer_sha256": input_optimizer_sha256,
        "base_model": str(Path(base_model).expanduser().resolve()),
        "output_adapter": str(root / "adapter"),
        "output_adapter_sha256": directory_sha256(adapter),
        "output_adapter_semantic_sha256": view_audit["semantic_tensor_sha256"],
        "output_rollout_adapter": str(root / "rollout_adapter"),
        "output_rollout_adapter_sha256": view_audit["view_directory_sha256"],
        "output_optimizer": str(root / "optimizer.pt"),
        "output_optimizer_sha256": sha256_file(optimizer_path),
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(staging / "learner_report.json", report)
    os.rename(staging, root)
    descriptor = os.open(root.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    del model, optimizer
    torch.cuda.empty_cache()
    return report


def validate_learner_report(
    path: Path,
    *,
    expected_method: str | None = None,
) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(report.get("schema_version") == LEARNER_SCHEMA and report.get("complete") is True, "M6 learner report schema drift")
    _require(report.get("content_sha256") == _self_hash(report), "M6 learner report self-hash drift")
    _require(report.get("development_only") is True and report.get("formal_checkpoint_reusable") is False, "M6 learner scope drift")
    method = report.get("method")
    _require(method in METHODS and report.get("group_size") == GROUP_SIZE, "M6 learner method/K drift")
    _require(expected_method is None or method == expected_method, "M6 learner expected-method drift")
    _require(report.get("learner_microbatch_size") in {1, 2, 4, 8}, "M6 learner microbatch drift")
    _require(report.get("credit_formula_version") == FORMULA_VERSION_BY_METHOD[method], "M6 learner credit formula drift")
    _require(report.get("verifier_td_lambda") == (0.0 if method == BASELINE_METHOD else 0.5), "M6 learner lambda/method drift")
    _require(report.get("iteration_optimizer_updates", 0) > 0, "M6 learner made no optimizer update")
    _require(report.get("cumulative_optimizer_updates_after", 0) > report.get("cumulative_optimizer_updates_before", -1), "M6 learner update counter drift")
    _require(report.get("input_adapter_semantic_sha256") != report.get("output_adapter_semantic_sha256"), "M6 learner parameters did not change")
    _require(
        isinstance(report.get("initial_replay_parity_checks"), Mapping)
        and all(report["initial_replay_parity_checks"].values()),
        "M6 learner replay parity did not pass",
    )
    _require(
        isinstance(report.get("behavior_sampling_parity"), Mapping)
        and report["behavior_sampling_parity"].get("passed") is True,
        "M6 learner behavior/sampling parity did not pass",
    )
    collection = report.get("collection_audit")
    _require(isinstance(collection, Mapping), "M6 learner collection audit is missing")
    _require(
        collection.get("credit_assignment_content_sha256")
        and int(collection.get("mixed_strict_reward_group_count", 0)) > 0,
        "M6 learner lacks mixed-reward credit evidence",
    )
    _require(directory_sha256(Path(report["output_adapter"])) == report["output_adapter_sha256"], "M6 learner adapter hash drift")
    _require(sha256_file(Path(report["output_optimizer"])) == report["output_optimizer_sha256"], "M6 learner optimizer hash drift")
    return report
