from __future__ import annotations

import asyncio
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from miniwebwork.long_horizon_rl.vllm_backend import (
    GENERATION_BACKEND,
    RolloutRequestContext,
    ThreadsafeVLLMBackend,
    VLLMBackendConfig,
    derive_sampling_seed,
    extract_chosen_token_logprobs,
)
from miniwebwork.model_agent.model_backend import GenerationResult


def test_vllm_engine_kwargs_freeze_raw_logprobs_same_gpu_lora_and_batching():
    config = VLLMBackendConfig(
        base_model="/data/share/model/Qwen3.5-4B",
        adapter_path="/tmp/adapter",
        adapter_sha256="a" * 64,
        rollout_adapter_path="/tmp/rollout-adapter",
        rollout_adapter_sha256="b" * 64,
        adapter_semantic_sha256="c" * 64,
        seed=20260801,
    )
    kwargs = config.engine_kwargs()
    assert kwargs["logprobs_mode"] == "raw_logprobs"
    assert kwargs["language_model_only"] is True
    assert kwargs["enable_lora"] is True
    assert kwargs["max_lora_rank"] == 16
    assert kwargs["max_num_seqs"] == 8
    assert kwargs["enable_prefix_caching"] is True
    assert kwargs["enable_chunked_prefill"] is True
    assert kwargs["enforce_eager"] is True
    assert kwargs["enable_sleep_mode"] is True
    assert kwargs["generation_config"] == "vllm"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("max_model_len", 8192, "length"),
        ("max_new_tokens", 256, "token cap"),
        ("gpu_memory_utilization", 0.95, "memory"),
        ("max_num_seqs", 16, "sequence"),
        ("enforce_eager", False, "eager"),
        ("adapter_semantic_sha256", "z" * 64, "semantic hash"),
    ],
)
def test_vllm_config_rejects_resource_or_sampling_contract_drift(field, value, error):
    values = {
        "base_model": "/data/share/model/Qwen3.5-4B",
        "adapter_path": "/tmp/adapter",
        "adapter_sha256": "a" * 64,
        "rollout_adapter_path": "/tmp/rollout-adapter",
        "rollout_adapter_sha256": "b" * 64,
        "adapter_semantic_sha256": "c" * 64,
        "seed": 20260801,
        field: value,
    }
    with pytest.raises(ValueError, match=error):
        VLLMBackendConfig(**values).validate(check_adapter_files=False)


def test_flat_or_legacy_logprobs_extract_exact_chosen_tokens():
    class _Logprob:
        def __init__(self, value):
            self.logprob = value

    values = extract_chosen_token_logprobs(
        [3, 4],
        [{3: _Logprob(-0.2), 7: _Logprob(-1.0)}, {4: _Logprob(-0.3)}],
    )
    assert values == [-0.2, -0.3]
    with pytest.raises(ValueError, match="absent"):
        extract_chosen_token_logprobs([3], [{9: _Logprob(-0.1)}])


def test_sampling_seed_is_deterministic_and_changes_across_rollout_or_turn():
    first = derive_sampling_seed(
        run_seed=20260801,
        iteration_index=0,
        group_id="group-0000",
        attempt_index=0,
        rollout_index=0,
        turn_index=1,
    )
    assert first == derive_sampling_seed(
        run_seed=20260801,
        iteration_index=0,
        group_id="group-0000",
        attempt_index=0,
        rollout_index=0,
        turn_index=1,
    )
    assert first != derive_sampling_seed(
        run_seed=20260801,
        iteration_index=0,
        group_id="group-0000",
        attempt_index=0,
        rollout_index=1,
        turn_index=1,
    )
    assert first != derive_sampling_seed(
        run_seed=20260801,
        iteration_index=0,
        group_id="group-0000",
        attempt_index=0,
        rollout_index=0,
        turn_index=2,
    )


class _FakeAsyncEngine:
    def __init__(self):
        self.config = SimpleNamespace(base_model="base")
        self._adapter_sha256 = "a" * 64
        self._rollout_adapter_sha256 = "b" * 64
        self._adapter_semantic_sha256 = "c" * 64

    async def generate_messages(self, messages, *, request_id, sampling_seed):
        await asyncio.sleep(0.01)
        return GenerationResult(
            raw_text="{}",
            new_tokens=1,
            input_tokens=2,
            prompt_token_ids=[1, 2],
            generated_token_ids=[3],
            logprobs=[-0.1],
            sampling_logprobs=[-0.1],
            request_id=request_id,
            sampling_seed=sampling_seed,
            generation_backend=GENERATION_BACKEND,
            adapter_sha256=self._adapter_sha256,
            rollout_adapter_sha256=self._rollout_adapter_sha256,
            adapter_semantic_sha256=self._adapter_semantic_sha256,
        )


def test_threadsafe_browser_bridges_share_one_async_loop_without_request_collision():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop)
    thread.start()
    ready.wait(timeout=5)
    try:
        engine = _FakeAsyncEngine()
        bridges = [
            ThreadsafeVLLMBackend(
                engine=engine,
                event_loop=loop,
                context=RolloutRequestContext(
                    run_seed=20260801,
                    iteration_index=0,
                    group_id="group-0000",
                    attempt_index=0,
                    trajectory_id=f"trajectory-{index}",
                    rollout_index=index,
                ),
                event_loop_thread_id=thread.ident,
            )
            for index in range(4)
        ]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda bridge: bridge.generate([]), bridges))
        assert len({result.request_id for result in results}) == 4
        assert len({result.sampling_seed for result in results}) == 4
        assert all(result.generation_backend == GENERATION_BACKEND for result in results)
        assert all(result.adapter_sha256 == "a" * 64 for result in results)
        assert all(result.rollout_adapter_sha256 == "b" * 64 for result in results)
        assert all(result.adapter_semantic_sha256 == "c" * 64 for result in results)
        assert all(not result.error for result in results)
        assert all(math.isfinite(result.logprobs[0]) for result in results)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
