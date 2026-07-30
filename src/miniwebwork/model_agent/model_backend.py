"""Qwen3.5-4B Transformers backend — single GPU, greedy/stochastic decoding."""

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


@dataclass
class GenerationResult:
    raw_text: str = ""
    new_tokens: int = 0
    input_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""
    logprobs: list[float] = field(default_factory=list)


def extract_generated_token_logprobs(
    logits: torch.Tensor,
    prompt_length: int,
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """Return log-probabilities assigned to generated tokens.

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


class QwenTransformersBackend:
    """Loads Qwen model once and provides generation for chat messages."""

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
        return self._peak_memory / (1024 ** 3) if self._peak_memory else 0

    @property
    def load_time_s(self) -> float:
        return self._load_time

    def load(self):
        """Load model and tokenizer once."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        print(f"Loading tokenizer from {self.config.model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            local_files_only=self.config.local_files_only,
            trust_remote_code=True,
        )

        print(f"Loading model ({self.config.dtype})...")
        dtype = getattr(torch, self.config.dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            device_map={"": self.config.device},
            local_files_only=self.config.local_files_only,
            trust_remote_code=True,
        )
        self._model.eval()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._peak_memory = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            device_index = torch.device(self.config.device).index or 0
            gpu_name = torch.cuda.get_device_properties(device_index).name
            print(f"GPU: {gpu_name}, peak memory: {self.peak_memory_gb:.1f} GB")

        self._load_time = time.time() - t0
        self._loaded = True
        print(f"Model loaded in {self._load_time:.1f}s")

    def generate(self, messages: list) -> GenerationResult:
        """Tokenize messages, generate, decode new tokens and record logprobs."""
        if not self._loaded:
            self.load()

        t0 = time.time()
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
            result.input_tokens = input_ids.shape[1]

            gen_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.do_sample,
                "use_cache": self.config.use_cache,
                "num_beams": 1,
                "return_dict_in_generate": True,
                "output_scores": self.config.do_sample,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            if self.config.do_sample:
                gen_kwargs["temperature"] = self.config.temperature
                gen_kwargs["top_p"] = self.config.top_p
            if self._tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
            if self._tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

            with torch.inference_mode():
                generated = self._model.generate(input_ids, **gen_kwargs)

            sequences = generated.sequences
            new_ids = sequences[0, input_ids.shape[1]:]
            result.new_tokens = int(new_ids.numel())
            result.raw_text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

            # Generation scores are already aligned one-to-one with emitted tokens.
            # Prefer them over a second full forward pass; this avoids cache/state
            # coupling and is the exact on-policy distribution used for sampling.
            if self.config.do_sample and result.new_tokens > 0:
                scores = list(generated.scores or [])
                if len(scores) != result.new_tokens:
                    raise RuntimeError(
                        f"Generation score mismatch: {len(scores)} score tensors for "
                        f"{result.new_tokens} generated tokens"
                    )
                token_logprobs = []
                for idx, score in enumerate(scores):
                    token_id = new_ids[idx].to(score.device, dtype=torch.long)
                    if token_id.item() < 0 or token_id.item() >= score.shape[-1]:
                        raise ValueError(
                            f"Generated token id {token_id.item()} outside vocabulary "
                            f"size {score.shape[-1]}"
                        )
                    lp = torch.log_softmax(score[0].float(), dim=-1)[token_id]
                    token_logprobs.append(float(lp.detach().cpu()))
                result.logprobs = token_logprobs

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Model generation failed")

        result.latency_ms = (time.time() - t0) * 1000
        return result

    def get_chat_template_hash(self) -> str:
        ct = getattr(self._tokenizer, "chat_template", "") if self._tokenizer else ""
        return hashlib.sha256(ct.encode() if ct else b"").hexdigest()

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
            },
            "load_time_s": self._load_time,
            "peak_memory_gb": self.peak_memory_gb,
        }

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
