"""Qwen-compatible WebShop agent preserving token-level rollout evidence."""

from __future__ import annotations

from typing import Any

from ..model_agent.qwen_agent import ModelActionAttempt
from . import prompt
from .actions import parse_command_output


class QwenWebShopAgent:
    def __init__(self, backend: Any):
        self._backend = backend
        self._history: list[dict[str, Any]] = []
        self._model_turn = 0

    def reset(self, task_id: str, instruction: str) -> None:
        self._history = []
        self._model_turn = 0

    def act(self, observation: Any) -> ModelActionAttempt:
        self._model_turn += 1
        attempt = ModelActionAttempt(model_turn_index=self._model_turn)
        messages = prompt.build_messages(observation, self._history)
        attempt.prompt_hash = prompt.compute_message_hash(messages)
        generation = self._backend.generate(messages)
        attempt.raw_output = generation.raw_text
        attempt.input_tokens = generation.input_tokens
        attempt.output_tokens = generation.new_tokens
        attempt.latency_ms = generation.latency_ms
        attempt.prompt_token_ids = list(getattr(generation, "prompt_token_ids", []))
        attempt.generated_token_ids = list(getattr(generation, "generated_token_ids", []))
        attempt.token_logprobs = list(getattr(generation, "logprobs", []))
        attempt.sampling_logprobs = list(getattr(generation, "sampling_logprobs", []))
        attempt.request_id = str(getattr(generation, "request_id", ""))
        attempt.sampling_seed = int(getattr(generation, "sampling_seed", 0))
        attempt.generation_backend = str(getattr(generation, "generation_backend", "unknown"))
        attempt.adapter_sha256 = str(getattr(generation, "adapter_sha256", ""))
        attempt.rollout_adapter_sha256 = str(getattr(generation, "rollout_adapter_sha256", ""))
        attempt.adapter_semantic_sha256 = str(getattr(generation, "adapter_semantic_sha256", ""))
        attempt.queue_wait_ms = float(getattr(generation, "queue_wait_ms", 0.0))
        attempt.first_token_latency_ms = float(getattr(generation, "first_token_latency_ms", 0.0))
        attempt.generation_time_ms = float(getattr(generation, "generation_time_ms", 0.0))
        try:
            attempt.validate_rollout_evidence()
        except ValueError as exc:
            attempt.errors.append(f"rollout_evidence_error: {exc}")
            return attempt
        if generation.error:
            attempt.errors.append(f"generation_error: {generation.error}")
            return attempt
        parsed = parse_command_output(generation.raw_text)
        attempt.strict_json_success = parsed.strict_json_success
        attempt.fallback_used = parsed.fallback_used
        attempt.parsed_payload = parsed.parsed_payload
        attempt.schema_valid = parsed.schema_valid
        attempt.action = parsed.action  # type: ignore[assignment]
        attempt.errors.extend(parsed.errors)
        return attempt

    def record_feedback(self, attempt: ModelActionAttempt, action_result: Any, page_type: str) -> None:
        info = getattr(action_result, "info", {})
        payload = info.get("action_result", {}) if isinstance(info, dict) else {}
        self._history.append(
            {
                "turn": attempt.model_turn_index,
                "command": attempt.action.command if attempt.action is not None else "",  # type: ignore[attr-defined]
                "success": bool(payload.get("success", False)),
                "error_code": str(payload.get("error_code", ""))[:100],
                "page_type": str(page_type),
            }
        )

    @property
    def model_turn(self) -> int:
        return self._model_turn

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._history)
