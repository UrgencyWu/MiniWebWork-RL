"""Qwen3.5-4B Transformers backend for browser-agent inference and rollout.

The backend exposes two distinct probability concepts:

* ``logprobs``: raw policy log-probabilities from a teacher-forced forward
  pass over ``prompt + generated completion``.  These are the values that a
  later GRPO/PPO update must recompute under the old/current policy.
* ``sampling_logprobs``: optional log-probabilities from Transformers'
  generation scores after sampling processors/warpers.  They are diagnostic
  only and must not be mixed with raw policy log-probabilities.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_path: str = "/data/share/model/Qwen3.5-4B"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    use_cache: bool = True
    local_files_only: bool = True
    enable_thinking: bool = False
    collect_policy_logprobs: bool = True
    collect_sampling_logprobs: bool = False


@dataclass
class GenerationResult:
    raw_text: str = ""
    new_tokens: int = 0
    input_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""
    # Exact tokenized prompt used by the policy.  This is required to
    # reconstruct each browser turn during an RL policy update.
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    # Raw model policy log-probabilities, one per generated token.
    logprobs: list[float] = field(default_factory=list)
    # Optional post-processor/warper sampling log-probabilities.
    sampling_logprobs: list[float] = field(default_factory=list)


def extract_generated_token_logprobs(
    logits: torch.Tensor,
    prompt_length: int,
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """Return raw model log-probabilities assigned to generated tokens.

    ``logits[:, t, :]`` predicts token ``t + 1``.  Therefore the first
    generated token is predicted by the final prompt position
    ``prompt_length - 1``.  The returned tensor has exactly one value per
    generated token.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape [1, seq, vocab], got {tuple(logits.shape)}")
    if generated_ids.ndim != 1:
        raise ValueError(f"Expected generated_ids shape [n], got {tuple(generated_ids.shape)}")
    if prompt_length <= 0:
        raise ValueError(f"prompt_length must be positive, got {prompt_length}")

    num_generated = int(generated_ids.numel())
    if num_generated == 0:
        return logits.new_empty((0,), dtype=torch.float32)

    start = prompt_length - 1
    end = start + num_generated
    if end > logits.shape[1]:
        raise ValueError(
            "Insufficient logits for generated tokens: "
            f"need positions [{start}:{end}], sequence length={logits.shape[1]}"
        )

    token_logits = logits[0, start:end, :]
    if token_logits.shape[0] != num_generated:
        raise RuntimeError(
            f"Logprob alignment mismatch: {token_logits.shape[0]} logits for "
            f"{num_generated} generated tokens"
        )

    ids = generated_ids.to(token_logits.device, dtype=torch.long)
    if ids.numel() and (ids.min().item() < 0 or ids.max().item() >= token_logits.shape[-1]):
        raise ValueError(
            f"Generated token id out of vocabulary range [0, {token_logits.shape[-1]}): "
            f"min={ids.min().item()}, max={ids.max().item()}"
        )

    positions = torch.arange(num_generated, device=token_logits.device)
    return torch.log_softmax(token_logits.float(), dim=-1)[positions, ids]


def _sampling_logprobs(scores: tuple[torch.Tensor, ...], generated_ids: torch.Tensor) -> list[float]:
    """Extract one post-warp sampling log-probability per emitted token."""
    if len(scores) != int(generated_ids.numel()):
        raise RuntimeError(
            f"Generation score mismatch: {len(scores)} score tensors for "
            f"{generated_ids.numel()} generated tokens"
        )

    values: list[float] = []
    for index, score in enumerate(scores):
        token_id = generated_ids[index].to(score.device, dtype=torch.long)
        if token_id.item() < 0 or token_id.item() >= score.shape[-1]:
            raise ValueError(
                f"Generated token id {token_id.item()} outside vocabulary size {score.shape[-1]}"
            )
        value = torch.log_softmax(score[0].float(), dim=-1)[token_id]
        values.append(float(value.detach().cpu()))
    return values


class QwenTransformersBackend:
    """Load Qwen once and generate one JSON action per browser turn."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._peak_memory = 0
        self._load_time = 0.0

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def peak_memory_gb(self) -> float:
        return self._peak_memory / (1024**3) if self._peak_memory else 0.0

    @property
    def load_time_s(self) -> float:
        return self._load_time

    def load(self) -> None:
        """Load model and tokenizer once on the configured logical device."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.time()
        print(f"Loading tokenizer from {self.config.model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            local_files_only=self.config.local_files_only,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = getattr(torch, self.config.dtype)
        print(f"Loading model ({self.config.dtype})...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            device_map={"": self.config.device},
            local_files_only=self.config.local_files_only,
            trust_remote_code=True,
        )
        self._model.eval()

        if torch.cuda.is_available() and str(self.config.device).startswith("cuda"):
            torch.cuda.synchronize()
            self._peak_memory = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            device_index = torch.device(self.config.device).index or 0
            gpu_name = torch.cuda.get_device_properties(device_index).name
            print(f"GPU: {gpu_name}, peak memory: {self.peak_memory_gb:.1f} GB")

        self._load_time = time.time() - started
        self._loaded = True
        print(f"Model loaded in {self._load_time:.1f}s")

    def generate(self, messages: list[dict]) -> GenerationResult:
        """Render chat messages, generate an action, and capture rollout data."""
        if not self._loaded:
            self.load()

        started = time.time()
        result = GenerationResult()

        try:
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,
            )
            inputs = self._tokenizer(rendered, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.config.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.config.device)
            prompt_length = int(input_ids.shape[1])
            result.input_tokens = prompt_length
            result.prompt_token_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]

            want_sampling_scores = self.config.do_sample and self.config.collect_sampling_logprobs
            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.do_sample,
                "use_cache": self.config.use_cache,
                "num_beams": 1,
                "return_dict_in_generate": True,
                "output_scores": want_sampling_scores,
            }
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask
            if self.config.do_sample:
                if self.config.temperature <= 0:
                    raise ValueError("temperature must be > 0 when do_sample=True")
                generation_kwargs["temperature"] = self.config.temperature
                generation_kwargs["top_p"] = self.config.top_p
            if self._tokenizer.pad_token_id is not None:
                generation_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
            if self._tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

            with torch.inference_mode():
                generated = self._model.generate(input_ids, **generation_kwargs)

            sequences = generated.sequences
            new_ids = sequences[0, prompt_length:]
            result.new_tokens = int(new_ids.numel())
            result.generated_token_ids = [int(value) for value in new_ids.detach().cpu().tolist()]
            result.raw_text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

            if self.config.do_sample and self.config.collect_policy_logprobs and result.new_tokens:
                full_ids = sequences[:, : prompt_length + result.new_tokens]
                full_attention_mask = None
                if attention_mask is not None:
                    completion_mask = torch.ones(
                        (attention_mask.shape[0], result.new_tokens),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    full_attention_mask = torch.cat([attention_mask, completion_mask], dim=1)

                with torch.inference_mode():
                    forward_output = self._model(
                        input_ids=full_ids,
                        attention_mask=full_attention_mask,
                        use_cache=False,
                    )
                policy_logprobs = extract_generated_token_logprobs(
                    forward_output.logits,
                    prompt_length,
                    new_ids,
                )
                result.logprobs = [float(value) for value in policy_logprobs.detach().cpu().tolist()]

            if want_sampling_scores and result.new_tokens:
                result.sampling_logprobs = _sampling_logprobs(tuple(generated.scores or ()), new_ids)

            if len(result.prompt_token_ids) != result.input_tokens:
                raise RuntimeError("Prompt token count does not match input_tokens")
            if result.generated_token_ids and len(result.generated_token_ids) != result.new_tokens:
                raise RuntimeError("Generated token count does not match new_tokens")
            if result.logprobs and len(result.logprobs) != result.new_tokens:
                raise RuntimeError("Policy logprob count does not match generated token count")
            if result.sampling_logprobs and len(result.sampling_logprobs) != result.new_tokens:
                raise RuntimeError("Sampling logprob count does not match generated token count")

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Model generation failed")

        result.latency_ms = (time.time() - started) * 1000
        return result

    def get_chat_template_hash(self) -> str:
        template = getattr(self._tokenizer, "chat_template", "") if self._tokenizer else ""
        return hashlib.sha256(template.encode() if template else b"").hexdigest()

    def get_model_info(self) -> dict:
        return {
            "model_path": self.config.model_path,
            "model_type": self._model.config.model_type if self._model else "unknown",
            "dtype": self.config.dtype,
            "device": self.config.device,
            "chat_template_sha256": self.get_chat_template_hash(),
            "enable_thinking": self.config.enable_thinking,
            "generation_config": {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.do_sample,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "use_cache": self.config.use_cache,
                "collect_policy_logprobs": self.config.collect_policy_logprobs,
                "collect_sampling_logprobs": self.config.collect_sampling_logprobs,
            },
            "load_time_s": self._load_time,
            "peak_memory_gb": self.peak_memory_gb,
        }

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
