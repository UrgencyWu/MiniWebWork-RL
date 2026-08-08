"""K=4 atomic rollout evidence assembly and token-budget admission."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..m4_long_horizon_protocol import (
    ACTION_TOKEN_CAP,
    GROUP_SIZE,
    MAX_MODEL_TURNS,
    MAX_NEW_TOKENS,
)
from .contracts import (
    TRAJECTORY_SCHEMA,
    TURN_SCHEMA,
    RunIdentity,
    public_anchor_signature,
    token_ids_sha256,
    validate_trajectory_evidence,
)
from .journal import CollectionStore, build_committed_group
from .vllm_backend import GENERATION_BACKEND


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class GroupBudgetAdmission:
    allowed: bool
    global_tokens_before: int
    current_iteration_tokens: int
    reserved_inflight_groups: int
    maximum_group_reserve: int
    total_inflight_reserve: int
    token_cap: int
    reason: str


def admit_next_group(
    *,
    global_tokens_before_iteration: int,
    current_iteration_tokens: int,
    reserved_inflight_groups: int = 0,
    token_cap: int = ACTION_TOKEN_CAP,
) -> GroupBudgetAdmission:
    """Reserve the worst-case K4 group so the formal cap cannot be exceeded."""

    for name, value in (
        ("global_tokens_before_iteration", global_tokens_before_iteration),
        ("current_iteration_tokens", current_iteration_tokens),
        ("reserved_inflight_groups", reserved_inflight_groups),
        ("token_cap", token_cap),
    ):
        _require(isinstance(value, int) and value >= 0, f"invalid {name}")
    reserve = GROUP_SIZE * MAX_MODEL_TURNS * MAX_NEW_TOKENS
    consumed = global_tokens_before_iteration + current_iteration_tokens
    total_inflight_reserve = (reserved_inflight_groups + 1) * reserve
    allowed = consumed + total_inflight_reserve <= token_cap
    return GroupBudgetAdmission(
        allowed=allowed,
        global_tokens_before=global_tokens_before_iteration,
        current_iteration_tokens=current_iteration_tokens,
        reserved_inflight_groups=reserved_inflight_groups,
        maximum_group_reserve=reserve,
        total_inflight_reserve=total_inflight_reserve,
        token_cap=token_cap,
        reason="reserved_complete_group" if allowed else "insufficient_worst_case_group_reserve",
    )


class RolloutEvidenceWriter:
    """Convert one browser episode's two callbacks into immutable v3 evidence."""

    def __init__(
        self,
        *,
        store: CollectionStore,
        identity: RunIdentity,
        group_id: str,
        attempt_index: int,
        trajectory_id: str,
        task_id: str,
        rollout_index: int,
    ):
        identity.validate()
        _require(store.identity.sha256 == identity.sha256, "writer/store identity mismatch")
        _require(0 <= rollout_index < GROUP_SIZE, "writer rollout index outside K4")
        self.store = store
        self.identity = identity
        self.group_id = group_id
        self.attempt_index = attempt_index
        self.trajectory_id = trajectory_id
        self.task_id = task_id
        self.rollout_index = rollout_index
        self._charged_turns: set[int] = set()
        self._completed_turns: list[dict[str, Any]] = []

    def _validate_raw_turn_identity(self, turn: Mapping[str, Any]) -> int:
        turn_index = int(turn.get("model_turn_index", 0))
        _require(turn_index > 0, "raw rollout turn index is invalid")
        _require(
            turn_index == len(self._completed_turns) + 1
            or turn_index in self._charged_turns,
            "raw rollout turns are not contiguous",
        )
        _require(
            turn.get("generation_backend") == GENERATION_BACKEND,
            "formal rollout used a non-vLLM generation backend",
        )
        _require(isinstance(turn.get("request_id"), str) and turn["request_id"], "turn request id is missing")
        _require(
            isinstance(turn.get("sampling_seed"), int) and turn["sampling_seed"] >= 0,
            "turn sampling seed is invalid",
        )
        _require(
            turn.get("adapter_sha256") == self.identity.input_adapter_sha256,
            "raw rollout canonical adapter mismatch",
        )
        _require(
            turn.get("rollout_adapter_sha256")
            == self.identity.input_rollout_adapter_sha256,
            "raw rollout adapter view mismatch",
        )
        _require(
            turn.get("adapter_semantic_sha256")
            == self.identity.input_adapter_semantic_sha256,
            "raw rollout adapter semantic mismatch",
        )
        return turn_index

    def on_turn_generated(self, raw_turn: Mapping[str, Any]) -> dict[str, Any]:
        turn = dict(raw_turn)
        turn_index = self._validate_raw_turn_identity(turn)
        _require(turn_index not in self._charged_turns, "rollout turn was charged twice")
        generated_ids = turn.get("generated_token_ids")
        _require(isinstance(generated_ids, list) and bool(generated_ids), "generated turn has no token IDs")
        event = self.store.journal.append_turn_generated(
            group_id=self.group_id,
            iteration_index=self.identity.iteration_index,
            attempt_index=self.attempt_index,
            trajectory_id=self.trajectory_id,
            rollout_index=self.rollout_index,
            turn_index=turn_index,
            request_id=turn["request_id"],
            sampling_seed=turn["sampling_seed"],
            policy_version=self.identity.policy_version,
            adapter_sha256=self.identity.input_adapter_sha256,
            rollout_adapter_sha256=self.identity.input_rollout_adapter_sha256,
            adapter_semantic_sha256=self.identity.input_adapter_semantic_sha256,
            generated_token_ids=generated_ids,
        )
        self._charged_turns.add(turn_index)
        return event

    def _build_turn(self, raw_turn: Mapping[str, Any]) -> dict[str, Any]:
        turn = dict(raw_turn)
        turn_index = self._validate_raw_turn_identity(turn)
        _require(turn_index in self._charged_turns, "completed turn lacks durable token charge")
        _require(
            turn_index == len(self._completed_turns) + 1,
            "completed rollout turns are not contiguous",
        )
        prompt_ids = turn.get("prompt_token_ids")
        generated_ids = turn.get("generated_token_ids")
        behavior = turn.get("token_logprobs")
        sampling = turn.get("sampling_logprobs")
        _require(isinstance(prompt_ids, list) and bool(prompt_ids), "completed turn lacks prompt IDs")
        _require(isinstance(generated_ids, list) and bool(generated_ids), "completed turn lacks generated IDs")
        _require(
            isinstance(behavior, list)
            and isinstance(sampling, list)
            and len(behavior) == len(sampling) == len(generated_ids),
            "completed turn log-prob evidence is incomplete",
        )
        _require(
            all(math.isfinite(float(value)) for value in behavior + sampling),
            "completed turn contains non-finite log-probs",
        )
        observation = turn.get("observation")
        _require(isinstance(observation, Mapping), "completed turn lacks public observation")
        evidence = {
            "schema_version": TURN_SCHEMA,
            "study_id": self.identity.study_id,
            "method": self.identity.method,
            "seed": self.identity.seed,
            "iteration_index": self.identity.iteration_index,
            "attempt_index": self.attempt_index,
            "policy_version": self.identity.policy_version,
            "group_id": self.group_id,
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "rollout_index": self.rollout_index,
            "turn_index": turn_index,
            "request_id": turn["request_id"],
            "sampling_seed": turn["sampling_seed"],
            "generation_backend": turn["generation_backend"],
            "adapter_sha256": self.identity.input_adapter_sha256,
            "rollout_adapter_sha256": self.identity.input_rollout_adapter_sha256,
            "adapter_semantic_sha256": self.identity.input_adapter_semantic_sha256,
            "rendered_prompt_sha256": turn["rendered_prompt_sha256"],
            "prompt_token_ids": list(prompt_ids),
            "prompt_token_sha256": token_ids_sha256(prompt_ids),
            "generated_token_ids": list(generated_ids),
            "generated_token_sha256": token_ids_sha256(generated_ids),
            "behavior_logprobs": [float(value) for value in behavior],
            "sampling_logprobs": [float(value) for value in sampling],
            "observation": dict(observation),
            "anchor_signature": public_anchor_signature(
                observation,
                prompt_token_ids=prompt_ids,
            ),
            "raw_output": str(turn.get("raw_output", "")),
            "strict_json_success": bool(turn.get("strict_json_success", False)),
            "fallback_used": bool(turn.get("fallback_used", False)),
            "schema_valid": bool(turn.get("schema_valid", False)),
            "parsed_action": turn.get("action"),
            "errors": list(turn.get("errors") or []),
            "action_result": turn.get("action_result"),
            "turn_reward": float(turn.get("reward", 0.0)),
            "terminated": bool(turn.get("terminated", False)),
            "truncated": bool(turn.get("truncated", False)),
            "latency_ms": float(turn.get("latency_ms", 0.0)),
            "queue_wait_ms": float(turn.get("queue_wait_ms", 0.0)),
            "first_token_latency_ms": float(turn.get("first_token_latency_ms", 0.0)),
            "generation_time_ms": float(turn.get("generation_time_ms", 0.0)),
        }
        return evidence

    def on_turn_completed(self, raw_turn: Mapping[str, Any]) -> dict[str, Any]:
        evidence = self._build_turn(raw_turn)
        artifact = self.store.write_turn_artifact(evidence)
        self._completed_turns.append(evidence)
        return artifact

    def finalize(self, task_result: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(task_result)
        _require(result.get("task_id") == self.task_id, "trajectory task result mismatch")
        _require(result.get("rollout_valid") is True, "cannot complete infrastructure-invalid trajectory")
        _require(
            len(self._completed_turns) == len(self._charged_turns) > 0,
            "trajectory lacks complete charged turn artifacts",
        )
        reward = float(result.get("reward"))
        _require(reward in (0.0, 1.0), "valid trajectory reward must be binary")
        trajectory = {
            "schema_version": TRAJECTORY_SCHEMA,
            "study_id": self.identity.study_id,
            "method": self.identity.method,
            "seed": self.identity.seed,
            "iteration_index": self.identity.iteration_index,
            "attempt_index": self.attempt_index,
            "trajectory_id": self.trajectory_id,
            "group_id": self.group_id,
            "task_id": self.task_id,
            "policy_version": self.identity.policy_version,
            "adapter_sha256": self.identity.input_adapter_sha256,
            "rollout_adapter_sha256": self.identity.input_rollout_adapter_sha256,
            "adapter_semantic_sha256": self.identity.input_adapter_semantic_sha256,
            "rollout_index": self.rollout_index,
            "rollout_valid": True,
            "success": reward == 1.0,
            "reward": reward,
            "failure_origin": "none" if reward == 1.0 else "policy",
            "termination_reason": str(result.get("termination_reason", ""))[:200],
            "failure_reasons": list(result.get("failure_reasons") or []),
            "episode_id": str(result.get("episode_id", "")),
            "model_turns": int(result.get("model_turns", len(self._completed_turns))),
            "environment_steps": int(result.get("environment_steps", 0)),
            "elapsed_s": float(result.get("elapsed_s", 0.0)),
            "turns": list(self._completed_turns),
            "generated_action_tokens": sum(
                len(turn["generated_token_ids"]) for turn in self._completed_turns
            ),
        }
        validated = validate_trajectory_evidence(
            trajectory,
            self.identity,
            require_infra_valid=True,
        )
        self.store.mark_trajectory_completed(
            group_id=self.group_id,
            attempt_index=self.attempt_index,
            trajectory=validated,
        )
        return validated


RolloutWorker = Callable[[int, RolloutEvidenceWriter], Mapping[str, Any]]


def run_atomic_k4_group(
    *,
    store: CollectionStore,
    identity: RunIdentity,
    group_id: str,
    task_id: str,
    worker: RolloutWorker,
    maximum_workers: int = GROUP_SIZE,
) -> dict[str, Any]:
    """Run four candidates concurrently and commit only the exact valid set."""

    _require(
        isinstance(maximum_workers, int)
        and not isinstance(maximum_workers, bool)
        and 1 <= maximum_workers <= GROUP_SIZE,
        "K4 worker count must be within 1..K",
    )

    attempt_index = store.start_group(
        group_id=group_id,
        task_id=task_id,
        iteration_index=identity.iteration_index,
    )
    writers = [
        RolloutEvidenceWriter(
            store=store,
            identity=identity,
            group_id=group_id,
            attempt_index=attempt_index,
            trajectory_id=f"{group_id}.a{attempt_index}.r{rollout_index}",
            task_id=task_id,
            rollout_index=rollout_index,
        )
        for rollout_index in range(GROUP_SIZE)
    ]
    task_results: dict[int, Mapping[str, Any]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = {
            executor.submit(worker, rollout_index, writers[rollout_index]): rollout_index
            for rollout_index in range(GROUP_SIZE)
        }
        for future in as_completed(futures):
            rollout_index = futures[future]
            try:
                result = future.result()
                task_results[rollout_index] = result
                if result.get("rollout_valid") is not True:
                    errors.append(
                        f"rollout-{rollout_index}: infrastructure-invalid "
                        f"{result.get('termination_reason')}"
                    )
            except Exception as exc:
                errors.append(f"rollout-{rollout_index}: {type(exc).__name__}: {exc}"[:500])
    trajectories: list[dict[str, Any]] = []
    if not errors and len(task_results) == GROUP_SIZE:
        for rollout_index in range(GROUP_SIZE):
            try:
                trajectories.append(writers[rollout_index].finalize(task_results[rollout_index]))
            except Exception as exc:
                errors.append(f"finalize-{rollout_index}: {type(exc).__name__}: {exc}"[:500])
                break
    if errors:
        store.mark_group_invalid(
            group_id=group_id,
            attempt_index=attempt_index,
            reason=" | ".join(errors)[:500],
        )
        store.archive_invalid_attempt(
            group_id=group_id,
            attempt_index=attempt_index,
        )
        return {
            "committed": False,
            "group_id": group_id,
            "attempt_index": attempt_index,
            "errors": errors,
            "retained_generated_action_tokens": store.journal.generated_action_tokens,
        }
    group = build_committed_group(
        identity=identity,
        iteration_index=identity.iteration_index,
        attempt_index=attempt_index,
        group_id=group_id,
        task_id=task_id,
        policy_version=identity.policy_version,
        trajectories=trajectories,
    )
    committed = store.commit_group(group, attempt_index=attempt_index)
    return {
        "committed": True,
        "group_id": group_id,
        "attempt_index": attempt_index,
        "group": committed["group"],
        "path": committed["path"],
    }
