"""Qwen3.5-4B Transformers backend — single GPU, greedy decoding."""

import hashlib
import time
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_path: str = "/data/share/model/Qwen3.5-4B"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    max_new_tokens: int = 128
    do_sample: bool = False
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


class QwenTransformersBackend:
    """Loads Qwen model once, provides generate() for chat messages."""

    def __init__(self, config: ModelConfig = None):
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

        # Record memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._peak_memory = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            gpu_name = torch.cuda.get_device_properties(0).name
            print(f"GPU: {gpu_name}, peak memory: {self.peak_memory_gb:.1f} GB")

        self._load_time = time.time() - t0
        self._loaded = True
        print(f"Model loaded in {self._load_time:.1f}s")

    def generate(self, messages: list) -> GenerationResult:
        """Tokenize messages, generate, decode only new tokens."""
        if not self._loaded:
            self.load()

        t0 = time.time()
        result = GenerationResult()

        try:
            # Render chat template to text first, then tokenize
            render_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if self.config.enable_thinking is not None:
                render_kwargs["enable_thinking"] = self.config.enable_thinking

            rendered = self._tokenizer.apply_chat_template(messages, **render_kwargs)
            inputs = self._tokenizer(rendered, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.config.device)
            result.input_tokens = input_ids.shape[1]

            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    use_cache=self.config.use_cache,
                    num_beams=1,
                )
                # Add pad/eos if available
                if self._tokenizer.pad_token_id is not None:
                    gen_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
                if self._tokenizer.eos_token_id is not None:
                    gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

                outputs = self._model.generate(input_ids, **gen_kwargs)

            # Decode only new tokens
            new_ids = outputs[0][input_ids.shape[1]:]
            result.new_tokens = len(new_ids)
            result.raw_text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        except Exception as e:
            result.error = str(e)

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
                "use_cache": self.config.use_cache,
            },
            "load_time_s": self._load_time,
            "peak_memory_gb": self.peak_memory_gb,
        }

    def unload(self):
        if self._model:
            del self._model
            self._model = None
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
