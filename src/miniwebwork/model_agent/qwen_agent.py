"""Qwen browser policy used by evaluation and future RL rollout collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..agent_env.schemas import AgentAction


@dataclass
class ModelActionAttempt:
    """Complete evidence for one model decision.

    The token fields are intentionally retained here rather than reconstructed
    from decoded JSON.  M3.0 policy updates require the exact generated token
    sequence and the corresponding old-policy log-probabilities.
    """

    model_turn_index: int = 0
    raw_output: str = ""
    strict_json_success: bool = False
    fallback_used: bool = False
    parsed_payload: Optional[dict] = None
    schema_valid: bool = False
    action: Optional[AgentAction] = None
    errors: list[str] = field(default_factory=list)
    prompt_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    generated_token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    sampling_logprobs: list[float] = field(default_factory=list)

    def validate_rollout_evidence(self) -> None:
        """Raise when token-level evidence is internally inconsistent."""
        if self.generated_token_ids and len(self.generated_token_ids) != self.output_tokens:
            raise ValueError(
                f"generated_token_ids={len(self.generated_token_ids)} but output_tokens={self.output_tokens}"
            )
        if self.token_logprobs and len(self.token_logprobs) != self.output_tokens:
            raise ValueError(
                f"token_logprobs={len(self.token_logprobs)} but output_tokens={self.output_tokens}"
            )
        if self.sampling_logprobs and len(self.sampling_logprobs) != self.output_tokens:
            raise ValueError(
                f"sampling_logprobs={len(self.sampling_logprobs)} but output_tokens={self.output_tokens}"
            )


class QwenBrowserAgent:
    """Build a canonical prompt, invoke Qwen, and parse one JSON action."""

    def __init__(self, backend, prompt_builder, output_parser):
        self._backend = backend
        self._prompt_builder = prompt_builder
        self._parser = output_parser
        self._history: list[dict] = []
        self._model_turn = 0
        self._task_id = ""
        self._instruction = ""

    def reset(self, task_id: str, instruction: str) -> None:
        self._history = []
        self._model_turn = 0
        self._task_id = task_id
        self._instruction = instruction

    def act(self, observation) -> ModelActionAttempt:
        """Generate and parse the next action from the current observation."""
        self._model_turn += 1
        attempt = ModelActionAttempt(model_turn_index=self._model_turn)

        messages = self._prompt_builder.build_messages(observation, self._history)
        attempt.prompt_hash = self._prompt_builder.compute_message_hash(messages)

        generation = self._backend.generate(messages)
        attempt.raw_output = generation.raw_text
        attempt.input_tokens = generation.input_tokens
        attempt.output_tokens = generation.new_tokens
        attempt.latency_ms = generation.latency_ms
        attempt.generated_token_ids = list(getattr(generation, "generated_token_ids", []))
        attempt.token_logprobs = list(getattr(generation, "logprobs", []))
        attempt.sampling_logprobs = list(getattr(generation, "sampling_logprobs", []))

        try:
            attempt.validate_rollout_evidence()
        except ValueError as exc:
            attempt.errors.append(f"rollout_evidence_error: {exc}")
            return attempt

        if generation.error:
            attempt.errors.append(f"generation_error: {generation.error}")
            return attempt

        parser = getattr(self._parser, "parse", self._parser)
        parsed = parser(generation.raw_text) if callable(parser) else self._parser.parse(generation.raw_text)
        attempt.strict_json_success = parsed.strict_json_success
        attempt.fallback_used = parsed.fallback_used
        attempt.parsed_payload = parsed.parsed_payload
        attempt.schema_valid = parsed.schema_valid
        attempt.errors.extend(parsed.errors)

        if parsed.schema_valid and parsed.parsed_payload:
            try:
                attempt.action = AgentAction.from_dict(parsed.parsed_payload)
            except Exception as exc:
                attempt.schema_valid = False
                attempt.errors.append(f"action_construction_error: {exc}")

        return attempt

    def record_feedback(self, attempt: ModelActionAttempt, action_result, page_type: str) -> None:
        """Record compact action feedback for the bounded prompt history."""
        self._history.append(
            {
                "model_turn_index": attempt.model_turn_index,
                "action": attempt.action.to_dict() if attempt.action else None,
                "parse_ok": attempt.schema_valid,
                "result": action_result.to_dict() if action_result else {"success": False},
                "page_type": page_type,
            }
        )

    @property
    def model_turn(self) -> int:
        return self._model_turn
