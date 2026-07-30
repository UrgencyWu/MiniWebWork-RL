"""Typed rollout contracts shared by probes and future Agentic RL training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

POLICY_FAILURE = "policy"
INFRASTRUCTURE_FAILURE = "infrastructure"
NO_FAILURE = "none"


@dataclass
class RolloutStep:
    """One model decision and its environment consequence.

    ``token_logprobs`` are raw model-policy log-probabilities. The optional
    ``sampling_logprobs`` are produced after generation processors such as
    temperature and top-p. They answer different questions and must never be
    silently interchanged.
    """

    turn: int
    page_type: str
    prompt_hash: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    raw_model_output: str = ""
    generated_token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    sampling_logprobs: list[float] = field(default_factory=list)
    strict_json_success: bool = False
    fallback_used: bool = False
    schema_valid: bool = False
    schema_errors: list[str] = field(default_factory=list)
    parsed_action: Optional[dict] = None
    env_action_success: Optional[bool] = None
    env_error_code: str = ""
    skipped: bool = False
    terminated: bool = False
    truncated: bool = False

    def validate(self) -> None:
        token_count = len(self.generated_token_ids)
        if self.token_logprobs and len(self.token_logprobs) != token_count:
            raise ValueError(
                f"turn {self.turn}: {len(self.token_logprobs)} raw logprobs for "
                f"{token_count} generated tokens"
            )
        if self.sampling_logprobs and len(self.sampling_logprobs) != token_count:
            raise ValueError(
                f"turn {self.turn}: {len(self.sampling_logprobs)} sampling logprobs for "
                f"{token_count} generated tokens"
            )
        if self.schema_valid and self.parsed_action is None:
            raise ValueError(f"turn {self.turn}: schema-valid step has no parsed action")
        if self.skipped and self.env_action_success is not None:
            raise ValueError(f"turn {self.turn}: skipped step cannot have environment result")
        if self.fallback_used and self.strict_json_success:
            raise ValueError(f"turn {self.turn}: strict JSON and fallback cannot both be true")


@dataclass
class RolloutRecord:
    task_id: str
    task_type: str
    episode_id: str
    rollout_index: int
    rollout_seed: int
    policy: str
    temperature: float
    success: bool = False
    reward: Optional[float] = 0.0
    rollout_valid: bool = True
    failure_origin: str = POLICY_FAILURE
    termination_reason: str = ""
    model_turns: int = 0
    environment_steps: int = 0
    schema_valid_count: int = 0
    schema_invalid_count: int = 0
    verification: dict[str, Any] = field(default_factory=dict)
    steps: list[RolloutStep] = field(default_factory=list)

    def validate(self) -> None:
        if self.failure_origin not in {POLICY_FAILURE, INFRASTRUCTURE_FAILURE, NO_FAILURE}:
            raise ValueError(f"Unknown failure_origin: {self.failure_origin}")
        if not self.rollout_valid:
            if self.failure_origin != INFRASTRUCTURE_FAILURE:
                raise ValueError("Invalid rollout must be classified as infrastructure")
            if self.reward is not None:
                raise ValueError("Infrastructure failure must have reward=None")
        if self.success:
            if not self.rollout_valid or self.reward != 1.0:
                raise ValueError("Successful rollout must be valid with reward=1")
            if self.failure_origin != NO_FAILURE:
                raise ValueError("Successful rollout must have failure_origin='none'")
        if self.rollout_valid and not self.success and self.reward != 0.0:
            raise ValueError("Valid unsuccessful rollout must have reward=0")
        if self.schema_valid_count + self.schema_invalid_count != self.model_turns:
            raise ValueError(
                "Schema counts do not equal model turns: "
                f"{self.schema_valid_count}+{self.schema_invalid_count}!={self.model_turns}"
            )
        if len(self.steps) != self.model_turns:
            raise ValueError(f"Expected {self.model_turns} step events, got {len(self.steps)}")
        for step in self.steps:
            step.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class RolloutGroupSummary:
    task_id: str
    task_type: str
    policy: str
    temperature: float
    requested_k: int
    total_trajectories: int
    valid_trajectories: int
    infrastructure_errors: int
    success_count: int
    reward_sequence: list[float]
    reward_mean: float
    reward_std: float
    has_reward_variance: bool
    has_learning_signal: bool
    update_distribution_compatible: bool
    valid_for_grpo_update: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_rollout_seed(master_seed: int, task_id: str, rollout_index: int) -> int:
    """Derive a stable seed independent of Python's randomized hash()."""
    payload = f"{master_seed}:{task_id}:{rollout_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def summarize_group(
    records: Iterable[RolloutRecord],
    requested_k: int,
    *,
    update_distribution_compatible: bool = False,
) -> RolloutGroupSummary:
    """Summarize one same-task rollout group.

    A diagnostic group can have a useful mixed reward signal without being
    immediately eligible for a policy update. For example, a top-p readiness
    probe demonstrates exploration but is not the strict first on-policy
    training distribution. The caller must explicitly attest that rollout and
    training log-probability conventions match before
    ``valid_for_grpo_update`` can become true.
    """
    records = list(records)
    if not records:
        raise ValueError("Cannot summarize an empty rollout group")
    if requested_k <= 0:
        raise ValueError("requested_k must be positive")
    first = records[0]
    if any(record.task_id != first.task_id for record in records):
        raise ValueError("A rollout group cannot mix task IDs")
    if any(record.policy != first.policy for record in records):
        raise ValueError("A rollout group cannot mix policies")
    if any(record.temperature != first.temperature for record in records):
        raise ValueError("A rollout group cannot mix temperatures")

    for record in records:
        record.validate()

    valid = [record for record in records if record.rollout_valid]
    rewards = [float(record.reward) for record in valid if record.reward is not None]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards) if rewards else 0.0
    std = variance**0.5
    successes = sum(1 for record in valid if record.success)
    has_variance = std > 0.0
    has_learning_signal = len(valid) >= 2 and has_variance

    return RolloutGroupSummary(
        task_id=first.task_id,
        task_type=first.task_type,
        policy=first.policy,
        temperature=first.temperature,
        requested_k=requested_k,
        total_trajectories=len(records),
        valid_trajectories=len(valid),
        infrastructure_errors=len(records) - len(valid),
        success_count=successes,
        reward_sequence=rewards,
        reward_mean=mean,
        reward_std=std,
        has_reward_variance=has_variance,
        has_learning_signal=has_learning_signal,
        update_distribution_compatible=bool(update_distribution_compatible),
        valid_for_grpo_update=(
            has_learning_signal and bool(update_distribution_compatible)
        ),
    )


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-serializable value with stable key and whitespace rules."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
