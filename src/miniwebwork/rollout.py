"""Typed rollout contracts shared by probes and Agentic RL training."""

from __future__ import annotations

import hashlib
import json
import math
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
    temperature, top-p, and top-k. They answer different questions and must
    never be silently interchanged.
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

    def validate(self, *, require_complete_evidence: bool = True) -> None:
        token_count = len(self.generated_token_ids)
        if require_complete_evidence:
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
            if any(not math.isfinite(value) for value in self.token_logprobs):
                raise ValueError(f"turn {self.turn}: raw policy logprobs contain NaN or Inf")
            if any(not math.isfinite(value) for value in self.sampling_logprobs):
                raise ValueError(f"turn {self.turn}: sampling logprobs contain NaN or Inf")
        if self.schema_valid and self.parsed_action is None:
            raise ValueError(f"turn {self.turn}: schema-valid step has no parsed action")
        if self.skipped and self.env_action_success is not None:
            raise ValueError(f"turn {self.turn}: skipped step cannot have environment result")
        if self.fallback_used and self.strict_json_success:
            raise ValueError(f"turn {self.turn}: strict JSON and fallback cannot both be true")

    def diagnostic_dict(self) -> dict[str, Any]:
        """Return JSON-safe evidence for an infrastructure-invalid rollout."""
        value = asdict(self)
        token_count = len(self.generated_token_ids)
        for field_name in ("token_logprobs", "sampling_logprobs"):
            values = value[field_name]
            if len(values) != token_count or any(
                not math.isfinite(item) for item in values
            ):
                value[field_name] = []
                marker = f"invalid_{field_name}_evidence"
                if marker not in value["schema_errors"]:
                    value["schema_errors"].append(marker)
        return value


@dataclass
class RolloutRecord:
    task_id: str
    task_type: str
    episode_id: str
    rollout_index: int
    rollout_seed: int
    policy: str
    temperature: float
    top_p: float = 1.0
    top_k: int = 0
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
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
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
            step.validate(require_complete_evidence=self.rollout_valid)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        if not self.rollout_valid:
            value["steps"] = [step.diagnostic_dict() for step in self.steps]
        return value


@dataclass
class RolloutGroupSummary:
    task_id: str
    task_type: str
    policy: str
    temperature: float
    top_p: float
    top_k: int
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


def strict_raw_policy_distribution(temperature: float, top_p: float, top_k: int) -> bool:
    """Return whether sampling parameters match the raw categorical policy."""
    return (
        math.isclose(temperature, 1.0, rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(top_p, 1.0, rel_tol=0.0, abs_tol=1e-8)
        and top_k == 0
    )


def summarize_group(
    records: Iterable[RolloutRecord],
    requested_k: int,
    *,
    update_distribution_compatible: bool = False,
) -> RolloutGroupSummary:
    """Summarize one same-task, same-policy, same-distribution group.

    A diagnostic group can have a useful mixed reward signal without being
    eligible for a policy update. In the first implementation a caller cannot
    mark a warped distribution as update-compatible; future scaled/truncated
    distributions require a separate versioned probability contract.
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
    if any(record.top_p != first.top_p for record in records):
        raise ValueError("A rollout group cannot mix top_p values")
    if any(record.top_k != first.top_k for record in records):
        raise ValueError("A rollout group cannot mix top_k values")

    for record in records:
        record.validate()

    if update_distribution_compatible and not strict_raw_policy_distribution(
        first.temperature,
        first.top_p,
        first.top_k,
    ):
        raise ValueError(
            "update_distribution_compatible requires temperature=1, top_p=1, top_k=0"
        )

    valid = [record for record in records if record.rollout_valid]
    rewards = [float(record.reward) for record in valid if record.reward is not None]
    if any(not math.isfinite(reward) for reward in rewards):
        raise ValueError("rollout rewards contain NaN or Inf")
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
        top_p=first.top_p,
        top_k=first.top_k,
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
