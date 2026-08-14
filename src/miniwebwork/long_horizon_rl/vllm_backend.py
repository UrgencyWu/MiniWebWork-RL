"""Async vLLM 0.17 backend and synchronous browser-worker bridge."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..m4_long_horizon_protocol import (
    MAX_NEW_TOKENS,
    MAX_SEQUENCE_LENGTH,
    SFT_LORA_CONFIG,
)
from ..model_agent.model_backend import GenerationResult
from .adapter_view import validate_vllm_adapter_view
from .contracts import SHA256_PATTERN, directory_sha256

SAFE_REQUEST_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
GENERATION_BACKEND = "vllm_async"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    _require(
        isinstance(value, list)
        and bool(value)
        and all(isinstance(token_id, int) and token_id >= 0 for token_id in value),
        "vLLM prompt template did not produce one non-empty token-id list",
    )
    return [int(token_id) for token_id in value]


@dataclass(frozen=True)
class VLLMBackendConfig:
    base_model: str
    adapter_path: str
    adapter_sha256: str
    rollout_adapter_path: str
    rollout_adapter_sha256: str
    adapter_semantic_sha256: str
    seed: int
    dtype: str = "bfloat16"
    max_model_len: int = MAX_SEQUENCE_LENGTH
    max_new_tokens: int = MAX_NEW_TOKENS
    gpu_memory_utilization: float = 0.5
    max_num_seqs: int = 32
    enforce_eager: bool = True
    adapter_id: int = 1
    stream_interval: int = 8

    def validate(self, *, check_adapter_files: bool = True) -> None:
        _require(Path(self.base_model).is_absolute(), "vLLM base model path must be absolute")
        _require(Path(self.adapter_path).is_absolute(), "vLLM adapter path must be absolute")
        _require(
            Path(self.rollout_adapter_path).is_absolute(),
            "vLLM rollout adapter path must be absolute",
        )
        for field in (
            "adapter_sha256",
            "rollout_adapter_sha256",
            "adapter_semantic_sha256",
        ):
            value = getattr(self, field)
            _require(
                isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
                f"vLLM {field.replace('_sha256', '').replace('_', ' ')} hash drift",
            )
        _require(isinstance(self.seed, int) and self.seed >= 0, "vLLM seed drift")
        _require(self.dtype == "bfloat16", "vLLM dtype drift")
        _require(self.max_model_len == MAX_SEQUENCE_LENGTH, "vLLM model length drift")
        _require(self.max_new_tokens == MAX_NEW_TOKENS, "vLLM turn token cap drift")
        _require(math.isclose(self.gpu_memory_utilization, 0.5), "vLLM memory fraction drift")
        _require(self.max_num_seqs == 32, "vLLM maximum sequence count drift")
        _require(self.enforce_eager is True, "vLLM Qwen3.5 LoRA eager gate disabled")
        _require(self.adapter_id == 1, "vLLM adapter id drift")
        _require(self.stream_interval == 8, "vLLM stream interval drift")
        if check_adapter_files:
            path = Path(self.adapter_path).expanduser().resolve()
            _require(path.is_dir(), "vLLM adapter directory is missing")
            _require(directory_sha256(path) == self.adapter_sha256, "vLLM adapter hash mismatch")
            view = Path(self.rollout_adapter_path).expanduser().resolve()
            audit = validate_vllm_adapter_view(
                source_adapter=path,
                view_directory=view,
                base_model=Path(self.base_model),
            )
            _require(
                audit["view_directory_sha256"] == self.rollout_adapter_sha256,
                "vLLM rollout adapter view hash mismatch",
            )
            _require(
                audit["semantic_tensor_sha256"] == self.adapter_semantic_sha256,
                "vLLM adapter semantic hash mismatch",
            )

    def engine_kwargs(self) -> dict[str, Any]:
        self.validate(check_adapter_files=False)
        return {
            "model": self.base_model,
            "tokenizer": self.base_model,
            "dtype": self.dtype,
            "seed": self.seed,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            # vLLM 0.17 dummy-LoRA CUDA-graph warm-up treats Qwen3.5's four
            # in_proj_qkvz slices as the two logical HF checkpoint modules and
            # fails before engine startup. Eager mode bypasses only that graph
            # capture; real LoRA loading, async batching, and sleep/wake remain.
            "enforce_eager": self.enforce_eager,
            "max_logprobs": 1,
            "logprobs_mode": "raw_logprobs",
            "language_model_only": True,
            "enable_lora": True,
            "max_loras": 1,
            "max_lora_rank": SFT_LORA_CONFIG["r"],
            # vLLM 0.17 labels Qwen3.5 Mamba align-mode prefix caching as
            # experimental. Keep it off while preserving chunked prefill so
            # replay-tail correctness is tested without cross-request state
            # reuse as a confounder.
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "enable_sleep_mode": True,
            "generation_config": "vllm",
            "stream_interval": self.stream_interval,
            "trust_remote_code": True,
            "disable_log_stats": False,
        }


def extract_chosen_token_logprobs(
    generated_token_ids: Sequence[int],
    logprobs: Any,
) -> list[float]:
    """Extract one chosen-token raw log-prob from FlatLogprobs or legacy lists."""

    token_ids = [int(token_id) for token_id in generated_token_ids]
    _require(bool(token_ids), "vLLM completion emitted no tokens")
    _require(logprobs is not None and len(logprobs) == len(token_ids), "vLLM log-prob length drift")
    values: list[float] = []
    for index, token_id in enumerate(token_ids):
        position = logprobs[index]
        _require(isinstance(position, Mapping), f"vLLM log-prob position {index} is malformed")
        _require(token_id in position, f"vLLM chosen token absent from log-probs at {index}")
        value = position[token_id]
        if hasattr(value, "logprob"):
            value = value.logprob
        value = float(value)
        _require(math.isfinite(value), f"vLLM chosen-token log-prob is non-finite at {index}")
        values.append(value)
    return values


def derive_sampling_seed(
    *,
    run_seed: int,
    iteration_index: int,
    group_id: str,
    attempt_index: int,
    rollout_index: int,
    turn_index: int,
) -> int:
    payload = (
        f"{run_seed}|{iteration_index}|{group_id}|{attempt_index}|"
        f"{rollout_index}|{turn_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


@dataclass(frozen=True)
class RolloutRequestContext:
    run_seed: int
    iteration_index: int
    group_id: str
    attempt_index: int
    trajectory_id: str
    rollout_index: int
    shared_prefix_turns: int = 0
    sampling_attempt_index: int | None = None

    def validate(self) -> None:
        for field in ("group_id", "trajectory_id"):
            value = getattr(self, field)
            _require(
                isinstance(value, str)
                and SAFE_REQUEST_COMPONENT.fullmatch(value) is not None,
                f"unsafe rollout request {field}",
            )
        for field in ("run_seed", "iteration_index", "attempt_index", "rollout_index", "shared_prefix_turns"):
            value = getattr(self, field)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"invalid rollout request {field}",
            )
        _require(
            self.sampling_attempt_index is None
            or (
                isinstance(self.sampling_attempt_index, int)
                and not isinstance(self.sampling_attempt_index, bool)
                and self.sampling_attempt_index >= 0
            ),
            "invalid rollout request sampling_attempt_index",
        )


def derive_context_sampling_seed(context: RolloutRequestContext, *, turn_index: int) -> int:
    """Share a stochastic prefix within K, then branch with rollout-specific seeds."""

    context.validate()
    _require(isinstance(turn_index, int) and turn_index > 0, "invalid rollout turn index")
    seed_rollout_index = 0 if turn_index <= context.shared_prefix_turns else context.rollout_index
    return derive_sampling_seed(
        run_seed=context.run_seed,
        iteration_index=context.iteration_index,
        group_id=context.group_id,
        attempt_index=(
            context.attempt_index
            if context.sampling_attempt_index is None
            else context.sampling_attempt_index
        ),
        rollout_index=seed_rollout_index,
        turn_index=turn_index,
    )


class AsyncVLLMGenerationEngine:
    """Own one AsyncLLM and transition it exclusively between generation/learner."""

    def __init__(self, *, config: VLLMBackendConfig, engine: Any, tokenizer: Any):
        config.validate()
        self.config = config
        self._engine = engine
        self._tokenizer = tokenizer
        self._phase = "generation"
        self._inflight = 0
        self._adapter_path = config.adapter_path
        self._adapter_sha256 = config.adapter_sha256
        self._rollout_adapter_path = config.rollout_adapter_path
        self._rollout_adapter_sha256 = config.rollout_adapter_sha256
        self._adapter_semantic_sha256 = config.adapter_semantic_sha256

    @classmethod
    async def create(cls, config: VLLMBackendConfig) -> "AsyncVLLMGenerationEngine":
        config.validate()
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            local_files_only=True,
            trust_remote_code=True,
        )
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config.engine_kwargs()))
        instance = cls(config=config, engine=engine, tokenizer=tokenizer)
        await instance._add_adapter(
            adapter_path=config.adapter_path,
            adapter_sha256=config.adapter_sha256,
            rollout_adapter_path=config.rollout_adapter_path,
            rollout_adapter_sha256=config.rollout_adapter_sha256,
            adapter_semantic_sha256=config.adapter_semantic_sha256,
        )
        return instance

    def _lora_request(
        self,
        path: str,
        rollout_sha256: str,
        adapter_sha256: str,
    ):
        from vllm.lora.request import LoRARequest

        return LoRARequest(
            lora_name=(
                f"policy-{adapter_sha256[:12]}-view-{rollout_sha256[:12]}"
            ),
            lora_int_id=self.config.adapter_id,
            lora_path=path,
        )

    async def _add_adapter(
        self,
        *,
        adapter_path: str,
        adapter_sha256: str,
        rollout_adapter_path: str,
        rollout_adapter_sha256: str,
        adapter_semantic_sha256: str,
    ) -> None:
        resolved = Path(adapter_path).expanduser().resolve()
        view = Path(rollout_adapter_path).expanduser().resolve()
        _require(resolved.is_dir(), "vLLM canonical adapter directory is missing")
        _require(
            directory_sha256(resolved) == adapter_sha256,
            "vLLM canonical adapter swap hash mismatch",
        )
        audit = validate_vllm_adapter_view(
            source_adapter=resolved,
            view_directory=view,
            base_model=Path(self.config.base_model),
        )
        _require(
            audit["view_directory_sha256"] == rollout_adapter_sha256,
            "vLLM rollout adapter swap hash mismatch",
        )
        _require(
            audit["semantic_tensor_sha256"] == adapter_semantic_sha256,
            "vLLM adapter swap semantic hash mismatch",
        )
        loaded = await self._engine.add_lora(
            self._lora_request(
                str(view),
                rollout_adapter_sha256,
                adapter_sha256,
            )
        )
        _require(loaded is True, "vLLM refused the requested LoRA adapter")
        self._adapter_path = str(resolved)
        self._adapter_sha256 = adapter_sha256
        self._rollout_adapter_path = str(view)
        self._rollout_adapter_sha256 = rollout_adapter_sha256
        self._adapter_semantic_sha256 = adapter_semantic_sha256

    async def generate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str,
        sampling_seed: int,
    ) -> GenerationResult:
        _require(self._phase == "generation", "vLLM generation requested outside generation phase")
        _require(SAFE_REQUEST_COMPONENT.fullmatch(request_id) is not None, "unsafe vLLM request id")
        _require(isinstance(sampling_seed, int) and sampling_seed >= 0, "invalid vLLM sampling seed")
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        prompt_ids = _normalize_token_ids(
            self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        _require(
            len(prompt_ids) + self.config.max_new_tokens <= self.config.max_model_len,
            "vLLM request exceeds frozen model length",
        )
        sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            seed=sampling_seed,
            max_tokens=self.config.max_new_tokens,
            logprobs=0,
            flat_logprobs=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
            detokenize=True,
            skip_special_tokens=True,
        )
        started = time.monotonic()
        final = None
        self._inflight += 1
        try:
            async for output in self._engine.generate(
                TokensPrompt(prompt_token_ids=prompt_ids),
                sampling,
                request_id,
                lora_request=self._lora_request(
                    self._rollout_adapter_path,
                    self._rollout_adapter_sha256,
                    self._adapter_sha256,
                ),
            ):
                final = output
        finally:
            self._inflight -= 1
        _require(final is not None and bool(final.finished), "vLLM request did not finish")
        _require(len(final.outputs) == 1, "vLLM request returned non-single completion")
        completion = final.outputs[0]
        generated_ids = [int(token_id) for token_id in completion.token_ids]
        raw_logprobs = extract_chosen_token_logprobs(generated_ids, completion.logprobs)
        metrics = getattr(final, "metrics", None)
        _require(metrics is not None, "vLLM request metrics are missing")
        _require(
            int(metrics.num_generation_tokens) == len(generated_ids),
            "vLLM metrics/token count mismatch",
        )
        queue_wait_ms = max(0.0, float(metrics.scheduled_ts - metrics.queued_ts) * 1000)
        generation_time_ms = max(
            0.0,
            float(metrics.last_token_ts - metrics.first_token_ts) * 1000,
        )
        return GenerationResult(
            raw_text=str(completion.text),
            new_tokens=len(generated_ids),
            input_tokens=len(prompt_ids),
            latency_ms=(time.monotonic() - started) * 1000,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
            logprobs=raw_logprobs,
            # With temperature=1, top_p=1, top_k=0, no penalty, and raw-logprob
            # mode, the behavior sampling distribution is exactly the raw policy.
            sampling_logprobs=list(raw_logprobs),
            request_id=request_id,
            sampling_seed=sampling_seed,
            generation_backend=GENERATION_BACKEND,
            adapter_sha256=self._adapter_sha256,
            rollout_adapter_sha256=self._rollout_adapter_sha256,
            adapter_semantic_sha256=self._adapter_semantic_sha256,
            queue_wait_ms=queue_wait_ms,
            first_token_latency_ms=float(metrics.first_token_latency) * 1000,
            generation_time_ms=generation_time_ms,
        )

    async def switch_to_learner(self) -> dict[str, Any]:
        _require(self._phase == "generation", "vLLM is not in generation phase")
        _require(self._inflight == 0, "cannot sleep vLLM with in-flight requests")
        removed = await self._engine.remove_lora(self.config.adapter_id)
        _require(removed is True, "vLLM failed to remove active adapter")
        await self._engine.sleep(level=2, mode="wait")
        self._phase = "learner"
        return {"phase": self._phase, "adapter_removed": True, "sleep_level": 2}

    async def switch_to_generation(
        self,
        *,
        adapter_path: str,
        adapter_sha256: str,
        rollout_adapter_path: str,
        rollout_adapter_sha256: str,
        adapter_semantic_sha256: str,
    ) -> dict[str, Any]:
        _require(self._phase == "learner", "vLLM is not in learner phase")
        await self._engine.wake_up()
        await self._add_adapter(
            adapter_path=adapter_path,
            adapter_sha256=adapter_sha256,
            rollout_adapter_path=rollout_adapter_path,
            rollout_adapter_sha256=rollout_adapter_sha256,
            adapter_semantic_sha256=adapter_semantic_sha256,
        )
        self._phase = "generation"
        return {
            "phase": self._phase,
            "adapter_path": self._adapter_path,
            "adapter_sha256": self._adapter_sha256,
            "rollout_adapter_path": self._rollout_adapter_path,
            "rollout_adapter_sha256": self._rollout_adapter_sha256,
            "adapter_semantic_sha256": self._adapter_semantic_sha256,
        }

    def shutdown(self) -> None:
        _require(self._inflight == 0, "cannot shut down vLLM with in-flight requests")
        self._engine.shutdown()
        self._phase = "shutdown"


@dataclass(frozen=True)
class RawVLLMBackendConfig:
    """Frozen no-LoRA vLLM identity used only by the raw-base evaluation."""

    base_model: str
    base_model_manifest_sha256: str
    base_model_functional_sha256: str
    seed: int
    dtype: str = "bfloat16"
    max_model_len: int = 8_192
    max_new_tokens: int = MAX_NEW_TOKENS
    gpu_memory_utilization: float = 0.5
    max_num_seqs: int = 32
    enforce_eager: bool = True
    stream_interval: int = 8

    def validate(self) -> None:
        _require(Path(self.base_model).is_absolute(), "raw vLLM base model path must be absolute")
        for field in ("base_model_manifest_sha256", "base_model_functional_sha256"):
            value = getattr(self, field)
            _require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None, f"invalid raw {field}")
        _require(isinstance(self.seed, int) and self.seed >= 0, "raw vLLM seed drift")
        _require(self.dtype == "bfloat16", "raw vLLM dtype drift")
        _require(self.max_model_len == 8_192, "raw M5 vLLM model length drift")
        _require(self.max_new_tokens == MAX_NEW_TOKENS, "raw vLLM turn-token cap drift")
        _require(math.isclose(self.gpu_memory_utilization, 0.5), "raw vLLM memory fraction drift")
        _require(self.max_num_seqs == 32, "raw vLLM maximum sequence count drift")
        _require(self.enforce_eager is True and self.stream_interval == 8, "raw vLLM runtime drift")

    def engine_kwargs(self) -> dict[str, Any]:
        self.validate()
        return {
            "model": self.base_model,
            "tokenizer": self.base_model,
            "dtype": self.dtype,
            "seed": self.seed,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "enforce_eager": self.enforce_eager,
            "max_logprobs": 1,
            "logprobs_mode": "raw_logprobs",
            "language_model_only": True,
            "enable_lora": False,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "generation_config": "vllm",
            "stream_interval": self.stream_interval,
            "trust_remote_code": True,
            "disable_log_stats": False,
        }


class RawAsyncVLLMGenerationEngine:
    """Async vLLM engine that evaluates the frozen base model without LoRA."""

    def __init__(self, *, config: RawVLLMBackendConfig, engine: Any, tokenizer: Any):
        config.validate()
        self.config = config
        self._engine = engine
        self._tokenizer = tokenizer
        self._phase = "generation"
        self._inflight = 0
        # Keep the generic rollout evidence interface explicit. These are base
        # model identities, not adapter identities, and are recorded as such in
        # the frozen evaluation manifest.
        self._adapter_sha256 = config.base_model_manifest_sha256
        self._rollout_adapter_sha256 = config.base_model_functional_sha256
        self._adapter_semantic_sha256 = config.base_model_functional_sha256

    @classmethod
    async def create(cls, config: RawVLLMBackendConfig) -> "RawAsyncVLLMGenerationEngine":
        config.validate()
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            local_files_only=True,
            trust_remote_code=True,
        )
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config.engine_kwargs()))
        return cls(config=config, engine=engine, tokenizer=tokenizer)

    async def generate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str,
        sampling_seed: int,
    ) -> GenerationResult:
        _require(self._phase == "generation", "raw vLLM generation requested outside generation phase")
        _require(SAFE_REQUEST_COMPONENT.fullmatch(request_id) is not None, "unsafe raw vLLM request id")
        _require(isinstance(sampling_seed, int) and sampling_seed >= 0, "invalid raw vLLM sampling seed")
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        prompt_ids = _normalize_token_ids(
            self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        _require(
            len(prompt_ids) + self.config.max_new_tokens <= self.config.max_model_len,
            "raw vLLM request exceeds frozen model length",
        )
        sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            seed=sampling_seed,
            max_tokens=self.config.max_new_tokens,
            logprobs=0,
            flat_logprobs=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
            detokenize=True,
            skip_special_tokens=True,
        )
        started = time.monotonic()
        final = None
        self._inflight += 1
        try:
            async for output in self._engine.generate(
                TokensPrompt(prompt_token_ids=prompt_ids),
                sampling,
                request_id,
            ):
                final = output
        finally:
            self._inflight -= 1
        _require(final is not None and bool(final.finished), "raw vLLM request did not finish")
        _require(len(final.outputs) == 1, "raw vLLM request returned non-single completion")
        completion = final.outputs[0]
        generated_ids = [int(token_id) for token_id in completion.token_ids]
        raw_logprobs = extract_chosen_token_logprobs(generated_ids, completion.logprobs)
        metrics = getattr(final, "metrics", None)
        _require(metrics is not None, "raw vLLM request metrics are missing")
        _require(int(metrics.num_generation_tokens) == len(generated_ids), "raw vLLM metrics/token count mismatch")
        queue_wait_ms = max(0.0, float(metrics.scheduled_ts - metrics.queued_ts) * 1000)
        generation_time_ms = max(0.0, float(metrics.last_token_ts - metrics.first_token_ts) * 1000)
        return GenerationResult(
            raw_text=str(completion.text),
            new_tokens=len(generated_ids),
            input_tokens=len(prompt_ids),
            latency_ms=(time.monotonic() - started) * 1000,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
            logprobs=raw_logprobs,
            sampling_logprobs=list(raw_logprobs),
            request_id=request_id,
            sampling_seed=sampling_seed,
            generation_backend="vllm_async_raw_base",
            adapter_sha256=self._adapter_sha256,
            rollout_adapter_sha256=self._rollout_adapter_sha256,
            adapter_semantic_sha256=self._adapter_semantic_sha256,
            queue_wait_ms=queue_wait_ms,
            first_token_latency_ms=float(metrics.first_token_latency) * 1000,
            generation_time_ms=generation_time_ms,
        )

    def shutdown(self) -> None:
        _require(self._inflight == 0, "cannot shut down raw vLLM with in-flight requests")
        self._engine.shutdown()
        self._phase = "shutdown"


class ThreadsafeVLLMBackend:
    """Expose synchronous ``generate`` to one persistent browser worker."""

    def __init__(
        self,
        *,
        engine: AsyncVLLMGenerationEngine | RawAsyncVLLMGenerationEngine,
        event_loop: asyncio.AbstractEventLoop,
        context: RolloutRequestContext,
        timeout_seconds: float = 900.0,
        event_loop_thread_id: int | None = None,
        initial_turn_index: int = 0,
    ):
        context.validate()
        _require(timeout_seconds > 0, "vLLM bridge timeout must be positive")
        _require(
            isinstance(initial_turn_index, int)
            and not isinstance(initial_turn_index, bool)
            and initial_turn_index >= 0,
            "invalid initial vLLM turn index",
        )
        self._engine = engine
        self._loop = event_loop
        self._context = context
        self._timeout_seconds = float(timeout_seconds)
        self._loop_thread_id = event_loop_thread_id
        self._turn_index = initial_turn_index
        self._lock = threading.Lock()

    def generate(self, messages: list[dict]) -> GenerationResult:
        if self._loop_thread_id is not None and threading.get_ident() == self._loop_thread_id:
            raise RuntimeError("synchronous vLLM bridge cannot block its own event-loop thread")
        with self._lock:
            self._turn_index += 1
            turn_index = self._turn_index
        request_id = (
            f"s{self._context.run_seed}.i{self._context.iteration_index}."
            f"{self._context.group_id}.a{self._context.attempt_index}."
            f"r{self._context.rollout_index}.t{turn_index}"
        )
        sampling_seed = derive_context_sampling_seed(self._context, turn_index=turn_index)
        future = asyncio.run_coroutine_threadsafe(
            self._engine.generate_messages(
                messages,
                request_id=request_id,
                sampling_seed=sampling_seed,
            ),
            self._loop,
        )
        try:
            return future.result(timeout=self._timeout_seconds)
        except Exception as exc:
            future.cancel()
            return GenerationResult(
                error=f"{type(exc).__name__}: {exc}"[:500],
                request_id=request_id,
                sampling_seed=sampling_seed,
                generation_backend=(
                    "vllm_async_raw_base"
                    if isinstance(self._engine, RawAsyncVLLMGenerationEngine)
                    else GENERATION_BACKEND
                ),
                adapter_sha256=self._engine._adapter_sha256,
                rollout_adapter_sha256=(
                    self._engine._rollout_adapter_sha256
                ),
                adapter_semantic_sha256=(
                    self._engine._adapter_semantic_sha256
                ),
            )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "backend": GENERATION_BACKEND,
            "base_model": self._engine.config.base_model,
            "adapter_sha256": self._engine._adapter_sha256,
            "rollout_adapter_sha256": self._engine._rollout_adapter_sha256,
            "adapter_semantic_sha256": self._engine._adapter_semantic_sha256,
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
        }
