#!/usr/bin/env python3
"""Run or aggregate exact Student-token-prefix logits alignment probes."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
import gc
import importlib.metadata
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from miniwebwork.long_horizon_rl.contracts import (  # noqa: E402
    atomic_write_json,
    directory_sha256,
    sha256_json,
)
from miniwebwork.m6_phase10b_opd import (  # noqa: E402
    LOGIT_MODEL_SCHEMA,
    MODEL_SPECS,
    PI0_ADAPTER_PATH,
    build_logit_probe_report,
    validate_logit_model_report,
    validate_model_tokenizer_manifest,
)

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CANONICAL_ACTIONS = (
    '{"command":"search[wireless mouse]"}',
    '{"command":"click[Red]"}',
    '{"command":"click[Buy Now]"}',
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _git_sha() -> str:
    value = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase10-B repository Git SHA drift")
    return value


def _hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("content_sha256") == _self_hash(value),
             f"Phase10-B input self-hash drift: {path}")
    return value


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer.encode(text, add_special_tokens=False)
    _require(isinstance(value, list) and value and all(isinstance(item, int) for item in value),
             "Phase10-B tokenizer returned invalid token IDs")
    return value


def _run_model(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    identity = str(args.identity)
    _require(identity in MODEL_SPECS, "Phase10-B unknown model identity")
    spec = MODEL_SPECS[identity]
    model_path = args.model_path.expanduser().resolve()
    student_tokenizer_path = args.student_tokenizer.expanduser().resolve()
    _require(str(model_path) == spec["path"], "Phase10-B model path drift")
    _require(str(student_tokenizer_path) == MODEL_SPECS["student"]["path"],
             "Phase10-B student tokenizer path drift")
    adapter = args.adapter.expanduser().resolve() if args.adapter else None
    expected_adapter = Path(PI0_ADAPTER_PATH) if identity == "student" else None
    _require(adapter == expected_adapter, "Phase10-B probe adapter drift")
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase10-B model probe output already exists")
    manifest = validate_model_tokenizer_manifest(_hashed(args.model_manifest.expanduser().resolve()))
    manifest_row = next(row for row in manifest["models"] if row["identity"] == identity)
    _require(manifest_row["path"] == str(model_path), "Phase10-B model manifest path drift")

    student_tokenizer = AutoTokenizer.from_pretrained(
        student_tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    native_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    system_prompt = args.prompt.expanduser().resolve().read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Instruction: Find a wireless mouse under $50 in red.\n"
                "Observation: Search results page. Available actions: "
                "search[query], click[item], click[Next]."
            ),
        },
    ]
    rendered = student_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    student_prefix = _token_ids(student_tokenizer, rendered)
    native_prefix = _token_ids(native_tokenizer, rendered)
    prefix_match = native_prefix == student_prefix
    student_action_ids = {action: _token_ids(student_tokenizer, action) for action in CANONICAL_ACTIONS}
    native_action_ids = {action: _token_ids(native_tokenizer, action) for action in CANONICAL_ACTIONS}
    action_match = native_action_ids == student_action_ids
    special_ids = {
        name: getattr(student_tokenizer, name)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    native_special_ids = {
        name: getattr(native_tokenizer, name)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    kernel_trust = None
    kernel_context = nullcontext()
    if identity == "S_finish":
        from transformers.integrations import hub_kernels

        mapping = hub_kernels._HUB_KERNEL_MAPPING.get("finegrained-fp8")  # noqa: SLF001
        _require(
            mapping == {"repo_id": "kernels-community/finegrained-fp8", "version": 4},
            "Phase10-B S_finish FP8 kernel mapping drift",
        )
        kernel_trust = dict(mapping)
        # Scope remote-code trust to one audited Transformers mapping and only
        # around this frozen model load/forward. The shared environment remains
        # unchanged and the global flag is restored on context exit.
        kernel_context = hub_kernels.allow_all_hub_kernels()
    with kernel_context:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map={"": "cuda:0"},
        )
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        input_ids = torch.tensor([student_prefix], dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, -1, :].float()
            probabilities = torch.softmax(logits, dim=-1)
            probability_sum_error = abs(float(probabilities.sum().cpu()) - 1.0)
            top_values, top_indices = torch.topk(probabilities, k=16, dim=-1)
    finite = bool(torch.isfinite(logits).all().item())
    runtime_vocab = int(logits.shape[-1])
    topk = [
        {"token_id": int(index), "probability": float(value)}
        for index, value in zip(top_indices[0].cpu(), top_values[0].cpu())
    ]
    result = {
        "schema_version": LOGIT_MODEL_SCHEMA,
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "producer_git_sha": _git_sha(),
        "identity": identity,
        "role": spec["role"],
        "model_name": spec["model_name"],
        "model_path": str(model_path),
        "input_adapter": str(adapter) if adapter else None,
        "input_adapter_sha256": directory_sha256(adapter) if adapter else None,
        "runtime_dependencies": {
            "kernels": importlib.metadata.version("kernels") if identity == "S_finish" else None,
            "trusted_fp8_kernel": kernel_trust,
        },
        "model_manifest_content_sha256": manifest["content_sha256"],
        "student_tokenizer_path": str(student_tokenizer_path),
        "student_tokenizer_used": True,
        "prefix_token_count": len(student_prefix),
        "prefix_token_ids_sha256": sha256_json(student_prefix),
        "prefix_token_ids_match": prefix_match,
        "canonical_action_token_ids_sha256": sha256_json(student_action_ids),
        "canonical_action_token_ids_match": action_match,
        "student_special_token_ids": special_ids,
        "native_special_token_ids": native_special_ids,
        "special_token_ids_match": special_ids == native_special_ids,
        "vocab_size": runtime_vocab,
        "finite_logits": finite,
        "probability_sum_abs_error": probability_sum_error,
        "top16": topk,
        "top16_sha256": sha256_json(topk),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checks": {
            "same_rendered_student_chat_template": True,
            "model_accepted_student_token_ids": True,
            "model_output_vocab_matches_student": runtime_vocab == 248320,
            "finite_logits": finite,
            "probability_mass_safe": probability_sum_error <= 1e-5,
        },
    }
    result["content_sha256"] = sha256_json(result)
    validate_logit_model_report(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps({
        "output": str(output),
        "identity": identity,
        "vocab_size": runtime_vocab,
        "prefix_match": prefix_match,
        "action_match": action_match,
        "probability_sum_abs_error": probability_sum_error,
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
        "elapsed_seconds": result["elapsed_seconds"],
        "content_sha256": result["content_sha256"],
    }, indent=2, sort_keys=True))
    del outputs, model, logits, probabilities, input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()


async def _run_vllm_model(args: argparse.Namespace) -> None:
    """Use vLLM's native FP8 path when HF requires an unavailable hub kernel."""

    from transformers import AutoConfig, AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    identity = str(args.identity)
    _require(identity == "S_finish", "Phase10-B vLLM alignment fallback is restricted to S_finish")
    model_path = args.model_path.expanduser().resolve()
    student_tokenizer_path = args.student_tokenizer.expanduser().resolve()
    _require(str(model_path) == MODEL_SPECS[identity]["path"], "Phase10-B vLLM model path drift")
    _require(str(student_tokenizer_path) == MODEL_SPECS["student"]["path"],
             "Phase10-B vLLM Student tokenizer path drift")
    _require(args.adapter is None, "Phase10-B vLLM Specialist cannot use Student adapter")
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase10-B vLLM model probe output already exists")
    manifest = validate_model_tokenizer_manifest(_hashed(args.model_manifest.expanduser().resolve()))
    manifest_row = next(row for row in manifest["models"] if row["identity"] == identity)
    _require(manifest_row["path"] == str(model_path), "Phase10-B vLLM manifest path drift")
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    native_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    raw_vocab_size = getattr(config, "vocab_size", None)
    if raw_vocab_size is None:
        raw_vocab_size = getattr(config, "text_config").vocab_size
    vocab_size = int(raw_vocab_size)
    system_prompt = args.prompt.expanduser().resolve().read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "Instruction: Find a wireless mouse under $50 in red.\n"
            "Observation: Search results page. Available actions: search[query], click[item], click[Next]."
        )},
    ]
    rendered = student_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    student_prefix = _token_ids(student_tokenizer, rendered)
    native_prefix = _token_ids(native_tokenizer, rendered)
    student_action_ids = {action: _token_ids(student_tokenizer, action) for action in CANONICAL_ACTIONS}
    native_action_ids = {action: _token_ids(native_tokenizer, action) for action in CANONICAL_ACTIONS}
    special_ids = {name: getattr(student_tokenizer, name) for name in ("bos_token_id", "eos_token_id", "pad_token_id")}
    native_special_ids = {name: getattr(native_tokenizer, name) for name in ("bos_token_id", "eos_token_id", "pad_token_id")}
    started = time.monotonic()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=str(model_path),
        tokenizer=str(student_tokenizer_path),
        dtype="bfloat16",
        seed=20260850,
        max_model_len=8192,
        gpu_memory_utilization=0.5,
        max_num_seqs=1,
        enforce_eager=True,
        max_logprobs=64,
        logprobs_mode="raw_logprobs",
        language_model_only=True,
        enable_lora=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        generation_config="vllm",
        trust_remote_code=True,
    ))
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        seed=20260850,
        max_tokens=1,
        logprobs=64,
        flat_logprobs=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
        detokenize=True,
        skip_special_tokens=True,
    )
    final = None
    try:
        async for item in engine.generate(TokensPrompt(prompt_token_ids=student_prefix), sampling, "p10b-sfinish-vllm"):
            final = item
        _require(final is not None and final.finished and len(final.outputs) == 1,
                 "Phase10-B vLLM alignment did not finish")
        completion = final.outputs[0]
        _require(len(completion.token_ids) == 1 and len(completion.logprobs) == 1,
                 "Phase10-B vLLM alignment completion drift")
        top_items = []
        for token_id, value in completion.logprobs[0].items():
            logprob = float(value.logprob if hasattr(value, "logprob") else value)
            _require(math.isfinite(logprob), "Phase10-B vLLM top-k logprob is non-finite")
            top_items.append({"token_id": int(token_id), "logprob": logprob, "probability": math.exp(logprob)})
        top_items.sort(key=lambda row: (-row["probability"], row["token_id"]))
        top_mass = sum(row["probability"] for row in top_items)
        _require(0.0 < top_mass <= 1.0 + 1e-5, "Phase10-B vLLM top-k probability mass drift")
        rest_mass = max(0.0, 1.0 - top_mass)
    finally:
        engine.shutdown()
    result = {
        "schema_version": LOGIT_MODEL_SCHEMA,
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "producer_git_sha": _git_sha(),
        "identity": identity,
        "role": MODEL_SPECS[identity]["role"],
        "model_name": MODEL_SPECS[identity]["model_name"],
        "model_path": str(model_path),
        "input_adapter": None,
        "input_adapter_sha256": None,
        "runtime_dependencies": {"backend": "vllm_native_fp8", "kernels": None},
        "model_manifest_content_sha256": manifest["content_sha256"],
        "student_tokenizer_path": str(student_tokenizer_path),
        "student_tokenizer_used": True,
        "prefix_token_count": len(student_prefix),
        "prefix_token_ids_sha256": sha256_json(student_prefix),
        "prefix_token_ids_match": native_prefix == student_prefix,
        "canonical_action_token_ids_sha256": sha256_json(student_action_ids),
        "canonical_action_token_ids_match": native_action_ids == student_action_ids,
        "student_special_token_ids": special_ids,
        "native_special_token_ids": native_special_ids,
        "special_token_ids_match": special_ids == native_special_ids,
        "vocab_size": vocab_size,
        "finite_logits": all(math.isfinite(row["logprob"]) for row in top_items),
        "probability_sum_abs_error": abs((top_mass + rest_mass) - 1.0),
        "top64": top_items,
        "top64_sha256": sha256_json(top_items),
        "top64_probability_mass": top_mass,
        "rest_mass": rest_mass,
        "full_vocab_finite_verified": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": None,
        "checks": {
            "same_rendered_student_chat_template": True,
            "model_accepted_student_token_ids": True,
            "model_output_vocab_matches_student": vocab_size == 248320,
            "finite_returned_topk_logits": all(math.isfinite(row["logprob"]) for row in top_items),
            "topk_plus_rest_probability_mass_safe": abs((top_mass + rest_mass) - 1.0) <= 1e-5,
        },
    }
    result["content_sha256"] = sha256_json(result)
    validate_logit_model_report(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps({
        "output": str(output),
        "identity": identity,
        "backend": "vllm_native_fp8",
        "vocab_size": vocab_size,
        "top64_probability_mass": top_mass,
        "rest_mass": rest_mass,
        "content_sha256": result["content_sha256"],
    }, indent=2, sort_keys=True))


def _aggregate(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase10-B aggregate probe output already exists")
    manifest = _hashed(args.model_manifest.expanduser().resolve())
    reports = [_hashed(path.expanduser().resolve()) for path in args.model_report]
    result = build_logit_probe_report(
        model_reports=reports,
        model_manifest=manifest,
        producer_git_sha=_git_sha(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps({
        "output": str(output),
        "all_models_pass": result["all_models_pass"],
        "identities": [row["identity"] for row in result["models"]],
        "content_sha256": result["content_sha256"],
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("model")
    model.add_argument("--identity", required=True, choices=tuple(MODEL_SPECS))
    model.add_argument("--model-path", type=Path, required=True)
    model.add_argument("--student-tokenizer", type=Path, required=True)
    model.add_argument("--adapter", type=Path)
    model.add_argument("--prompt", type=Path, required=True)
    model.add_argument("--model-manifest", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    vllm_model = subparsers.add_parser("vllm-model")
    for action in model._actions[1:]:  # argparse has no public clone helper; keep both CLIs identical.
        if action.dest == "help":
            continue
        kwargs: dict[str, Any] = {"required": action.required}
        if action.type is not None:
            kwargs["type"] = action.type
        if action.choices is not None:
            kwargs["choices"] = action.choices
        vllm_model.add_argument(*action.option_strings, **kwargs)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--model-manifest", type=Path, required=True)
    aggregate.add_argument("--model-report", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "model":
        _run_model(args)
    elif args.command == "vllm-model":
        asyncio.run(_run_vllm_model(args))
    else:
        _aggregate(args)


if __name__ == "__main__":
    main()
