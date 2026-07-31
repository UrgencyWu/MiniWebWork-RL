#!/usr/bin/env python3
"""Audit strict on-policy log-probability equivalence on a saved rollout.

The collector records two probability concepts: the model-policy likelihood
recomputed by the optimizer and the behavior likelihood used to sample an
action.  For an unwarped strict run they must agree.  This script makes that
claim falsifiable on an actual prompt from a rollout artifact by comparing
Transformers generation scores, generation raw logits, a full teacher-forced
forward pass, and a prefix-by-prefix no-cache forward pass.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if not SRC_DIR.is_dir():
    raise RuntimeError(f"Invalid source directory: {SRC_DIR}")
sys.path.insert(0, str(SRC_DIR))

from miniwebwork.model_agent.model_backend import extract_generated_token_logprobs


DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _choose_source(artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
    best: tuple[dict[str, Any], dict[str, Any], float] | None = None
    for record in artifact.get("records", []):
        for step in record.get("steps", []):
            raw = step.get("token_logprobs", [])
            sampled = step.get("sampling_logprobs", [])
            prompt = step.get("prompt_token_ids", [])
            completion = step.get("generated_token_ids", [])
            if not (
                isinstance(raw, list)
                and isinstance(sampled, list)
                and isinstance(prompt, list)
                and isinstance(completion, list)
                and prompt
                and completion
                and len(raw) == len(sampled) == len(completion)
            ):
                continue
            difference = max(abs(float(a) - float(b)) for a, b in zip(raw, sampled))
            if best is None or difference > best[2]:
                best = (record, step, difference)
    if best is None:
        raise ValueError("artifact contains no complete prompt/completion logprob evidence")
    return best


def _selected_logprobs(
    scores: tuple[torch.Tensor, ...] | list[torch.Tensor],
    token_ids: torch.Tensor,
) -> list[float]:
    if len(scores) != int(token_ids.numel()):
        raise ValueError(
            f"score/token length mismatch: {len(scores)} scores for {token_ids.numel()} tokens"
        )
    values: list[float] = []
    for score, token_id in zip(scores, token_ids):
        values.append(float(torch.log_softmax(score[0].float(), dim=-1)[token_id].cpu()))
    return values


def _series_difference(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        return {"comparable": False, "reason": "length_mismatch"}
    if not left:
        return {"comparable": False, "reason": "empty"}
    differences = [abs(a - b) for a, b in zip(left, right)]
    index = max(range(len(differences)), key=differences.__getitem__)
    return {
        "comparable": True,
        "count": len(differences),
        "max_abs_difference": differences[index],
        "mean_abs_difference": sum(differences) / len(differences),
        "max_index": index,
        "left_at_max": left[index],
        "right_at_max": right[index],
    }


def _generate_trace(
    model,
    prompt: torch.Tensor,
    *,
    pad_token_id: int | None,
    eos_token_id: int | list[int] | None,
    use_cache: bool,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generation = model.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        use_cache=use_cache,
        num_beams=1,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    completion = generation.sequences[0, prompt.shape[1] :]
    scores = tuple(generation.scores or ())
    raw_logits = tuple(getattr(generation, "logits", None) or ())
    result: dict[str, Any] = {
        "use_cache": use_cache,
        "generated_token_ids": [int(value) for value in completion.cpu().tolist()],
        "sampling_logprobs": _selected_logprobs(scores, completion),
        "score_count": len(scores),
        "raw_generation_logits_available": len(raw_logits) == int(completion.numel()),
    }
    if result["raw_generation_logits_available"]:
        result["raw_generation_logprobs"] = _selected_logprobs(raw_logits, completion)
        result["sampling_vs_generation_raw"] = _series_difference(
            result["sampling_logprobs"], result["raw_generation_logprobs"]
        )
    return result


def _teacher_forced_logprobs(
    model,
    prompt: torch.Tensor,
    completion: torch.Tensor,
) -> list[float]:
    full = torch.cat([prompt, completion.unsqueeze(0)], dim=1)
    with torch.inference_mode():
        output = model(
            input_ids=full,
            attention_mask=torch.ones_like(full),
            use_cache=False,
        )
    values = extract_generated_token_logprobs(
        output.logits,
        prompt_length=prompt.shape[1],
        generated_ids=completion,
    )
    return [float(value) for value in values.cpu().tolist()]


def _prefix_no_cache_logprobs(
    model,
    prompt: torch.Tensor,
    completion: torch.Tensor,
) -> list[float]:
    full = torch.cat([prompt, completion.unsqueeze(0)], dim=1)
    values: list[float] = []
    with torch.inference_mode():
        for index, token_id in enumerate(completion):
            prefix = full[:, : prompt.shape[1] + index]
            output = model(
                input_ids=prefix,
                attention_mask=torch.ones_like(prefix),
                use_cache=False,
            )
            values.append(
                float(torch.log_softmax(output.logits[0, -1].float(), dim=-1)[token_id].cpu())
            )
    return values


def _generation_config_snapshot(model) -> dict[str, Any]:
    config = model.generation_config.to_dict()
    relevant = (
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "top_h",
        "typical_p",
        "epsilon_cutoff",
        "eta_cutoff",
        "repetition_penalty",
        "encoder_repetition_penalty",
        "no_repeat_ngram_size",
        "renormalize_logits",
        "remove_invalid_values",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "suppress_tokens",
        "begin_suppress_tokens",
        "bad_words_ids",
        "sequence_bias",
        "guidance_scale",
        "dola_layers",
        "cache_implementation",
    )
    return {key: config.get(key) for key in relevant}


def _load_model(base_model: str, adapter: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter, torch_dtype=torch.bfloat16)
    model.enable_adapter_layers()
    model = model.to("cuda:0")
    model.eval()
    torch.cuda.synchronize()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    artifact = json.loads(args.artifact.expanduser().resolve().read_text(encoding="utf-8"))
    record, step, source_difference = _choose_source(artifact)
    prompt_ids = step["prompt_token_ids"]
    model, tokenizer = _load_model(args.base_model, args.adapter.expanduser().resolve())
    try:
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device="cuda:0").unsqueeze(0)
        cache_enabled = _generate_trace(
            model,
            prompt,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        cache_disabled = _generate_trace(
            model,
            prompt,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        for trace in (cache_enabled, cache_disabled):
            completion = torch.tensor(trace["generated_token_ids"], dtype=torch.long, device="cuda:0")
            trace["teacher_forced_logprobs"] = _teacher_forced_logprobs(model, prompt, completion)
            trace["prefix_no_cache_logprobs"] = _prefix_no_cache_logprobs(model, prompt, completion)
            trace["sampling_vs_teacher_forced"] = _series_difference(
                trace["sampling_logprobs"], trace["teacher_forced_logprobs"]
            )
            trace["teacher_forced_vs_prefix_no_cache"] = _series_difference(
                trace["teacher_forced_logprobs"], trace["prefix_no_cache_logprobs"]
            )
            if trace.get("raw_generation_logits_available"):
                trace["generation_raw_vs_teacher_forced"] = _series_difference(
                    trace["raw_generation_logprobs"], trace["teacher_forced_logprobs"]
                )

        report = {
            "schema_version": "m3_0b_logprob_audit_v1",
            "artifact": str(args.artifact.expanduser().resolve()),
            "adapter": str(args.adapter.expanduser().resolve()),
            "base_model": args.base_model,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "source": {
                "task_id": record.get("task_id"),
                "rollout_index": record.get("rollout_index"),
                "turn": step.get("turn"),
                "artifact_max_raw_sampling_difference": source_difference,
                "prompt_token_count": len(prompt_ids),
                "artifact_completion_token_count": len(step["generated_token_ids"]),
            },
            "model_generation_config": _generation_config_snapshot(model),
            "traces": {"use_cache_true": cache_enabled, "use_cache_false": cache_disabled},
        }
        _atomic_json_write(args.output.expanduser().resolve(), report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
