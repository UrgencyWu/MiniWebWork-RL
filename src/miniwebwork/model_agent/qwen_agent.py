"""QwenBrowserAgent: model-based procurement agent."""

from dataclasses import dataclass, field
from typing import Optional

from ..agent_env.schemas import AgentAction


@dataclass
class ModelActionAttempt:
    """Full record of one model turn."""
    model_turn_index: int = 0
    raw_output: str = ""
    strict_json_success: bool = False
    fallback_used: bool = False
    parsed_payload: Optional[dict] = None
    schema_valid: bool = False
    action: Optional[AgentAction] = None
    errors: list = field(default_factory=list)
    prompt_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class QwenBrowserAgent:
    """Agent that uses Qwen3.5-4B to decide actions from observations."""

    def __init__(self, backend, prompt_builder, output_parser):
        self._backend = backend
        self._prompt_builder = prompt_builder
        self._parser = output_parser
        self._history = []
        self._model_turn = 0
        self._task_id = ""
        self._instruction = ""

    def reset(self, task_id: str, instruction: str):
        self._history = []
        self._model_turn = 0
        self._task_id = task_id
        self._instruction = instruction

    def act(self, observation) -> ModelActionAttempt:
        """Build prompt, call model, parse output."""
        self._model_turn += 1
        attempt = ModelActionAttempt(model_turn_index=self._model_turn)

        # Build messages
        messages = self._prompt_builder.build_messages(observation, self._history)
        attempt.prompt_hash = self._prompt_builder.compute_message_hash(messages)

        # Generate
        gen = self._backend.generate(messages)
        attempt.raw_output = gen.raw_text
        attempt.input_tokens = gen.input_tokens
        attempt.output_tokens = gen.new_tokens
        attempt.latency_ms = gen.latency_ms

        if gen.error:
            attempt.errors.append(f"generation_error: {gen.error}")
            return attempt

        # Parse — support both module.parse() and direct function
        parser = getattr(self._parser, "parse", self._parser)
        parsed = parser(gen.raw_text) if callable(parser) else self._parser.parse(gen.raw_text)
        attempt.strict_json_success = parsed.strict_json_success
        attempt.fallback_used = parsed.fallback_used
        attempt.parsed_payload = parsed.parsed_payload
        attempt.schema_valid = parsed.schema_valid
        attempt.errors.extend(parsed.errors)

        # Build AgentAction
        if parsed.schema_valid and parsed.parsed_payload:
            try:
                attempt.action = AgentAction.from_dict(parsed.parsed_payload)
            except Exception as e:
                attempt.errors.append(f"action_construction_error: {e}")

        return attempt

    def record_feedback(self, attempt: ModelActionAttempt, action_result, page_type: str):
        """Record turn result in history."""
        self._history.append({
            "model_turn_index": attempt.model_turn_index,
            "action": attempt.action.to_dict() if attempt.action else None,
            "parse_ok": attempt.schema_valid,
            "result": action_result.to_dict() if action_result else {"success": False},
            "page_type": page_type,
        })

    @property
    def model_turn(self) -> int:
        return self._model_turn
