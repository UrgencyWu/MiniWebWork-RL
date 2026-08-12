#!/usr/bin/env python3
"""Run the recoverable development-only M6 mini-SFT stage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol  # noqa: E402
from miniwebwork.m6_pilot import validate_pilot_authorization  # noqa: E402
from miniwebwork.webshop_rl.m6_corpus import validate_conditional_learnability_audit  # noqa: E402
from miniwebwork.webshop_rl.m6_sft_training import (  # noqa: E402
    M6SFTConfig,
    benchmark_microbatches,
    build_update_schedule,
    build_token_audit,
    load_jsonl_examples,
    load_retention_examples,
    load_recovery,
    load_trainable_lora_model,
    train_mini_sft,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--pilot-authorization", type=Path)
    args = parser.parse_args()
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M6 mini SFT requires one Slurm GPU")
    protocol = load_protocol()
    _require(_git_sha() == protocol["git_sha"], "M6 mini SFT Git/protocol drift")
    data = args.data_dir.expanduser().resolve()
    required = {
        name: data / name
        for name in (
            "corpus.json",
            "corpus_audit.json",
            "token_audit.json",
            "train.jsonl",
            "dev.jsonl",
            "retention.json",
        )
    }
    _require(all(path.is_file() for path in required.values()), "M6 mini SFT data files are incomplete")
    corpus_audit = validate_conditional_learnability_audit(json.loads(required["corpus_audit.json"].read_text(encoding="utf-8")))
    pilot_authorization = None
    if corpus_audit["passed"] is not True:
        _require(args.pilot_authorization is not None, "M6 mini SFT corpus did not pass")
        pilot_authorization = validate_pilot_authorization(
            json.loads(args.pilot_authorization.read_text(encoding="utf-8")),
            corpus_audit=corpus_audit,
        )
        required["pilot_authorization.json"] = args.pilot_authorization.expanduser().resolve()
    config = M6SFTConfig.from_protocol(protocol["payload"], seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model.expanduser().resolve()), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        _require(tokenizer.eos_token_id is not None, "M6 tokenizer lacks pad/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train = load_jsonl_examples(required["train.jsonl"], tokenizer, config)
    dev = load_jsonl_examples(required["dev.jsonl"], tokenizer, config)
    retention = load_retention_examples(required["retention.json"], config)
    token_audit_inputs = {
        key: required[key]
        for key in ("corpus.json", "corpus_audit.json", "train.jsonl", "dev.jsonl", "retention.json")
    }
    if pilot_authorization is not None:
        token_audit_inputs["pilot_authorization.json"] = required["pilot_authorization.json"]
    rebuilt = build_token_audit(
        train_examples=train,
        dev_examples=dev,
        retention_examples=retention,
        base_model=args.base_model,
        input_files={
            key.replace(".json", "").replace(".jsonl", ""): value
            for key, value in token_audit_inputs.items()
        },
        config=config,
    )
    _require(rebuilt["passed"] is True, f"M6 runtime token audit failed: {rebuilt['checks']}")
    stored_token_audit = json.loads(required["token_audit.json"].read_text(encoding="utf-8"))
    expected_token_audit = dict(stored_token_audit)
    observed_token_audit_hash = expected_token_audit.pop("content_sha256", None)
    _require(observed_token_audit_hash == sha256_json(expected_token_audit), "M6 stored token audit self-hash drift")
    _require(stored_token_audit.get("passed") is True, "M6 stored token audit did not pass")
    _require(stored_token_audit.get("splits") == rebuilt.get("splits"), "M6 stored/runtime token audit split drift")
    _require(stored_token_audit.get("retention") == rebuilt.get("retention"), "M6 stored/runtime retention audit drift")
    _require(stored_token_audit.get("base_model") == rebuilt.get("base_model"), "M6 stored/runtime base-model drift")
    _require(stored_token_audit.get("config") == rebuilt.get("config"), "M6 stored/runtime SFT config drift")
    _require(
        stored_token_audit.get("input_file_sha256") == rebuilt.get("input_file_sha256"),
        "M6 stored/runtime SFT input-file drift",
    )
    _require(stored_token_audit.get("tokenizer_file_sha256") == rebuilt.get("tokenizer_file_sha256"), "M6 tokenizer binding drift")
    _require(stored_token_audit.get("protocol_sha256") == protocol["sha256"], "M6 token audit protocol drift")
    _require(stored_token_audit.get("git_sha") == protocol["git_sha"], "M6 token audit Git drift")
    _require(
        stored_token_audit.get("pilot_authorization_content_sha256")
        == (pilot_authorization["content_sha256"] if pilot_authorization is not None else None),
        "M6 token audit pilot-authorization drift",
    )
    _require(
        stored_token_audit.get("corpus_audit_sha256") == sha256_file(required["corpus_audit.json"]),
        "M6 token/corpus audit binding drift",
    )
    input_sha = {name: sha256_file(path) for name, path in required.items()}
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation = {
        "schema_version": "m6_mini_sft_invocation_v1",
        "development_only": True,
        "formal_training": False,
        "git_sha": protocol["git_sha"],
        "protocol_sha256": protocol["sha256"],
        "input_sha256": input_sha,
        "config": config.to_payload(),
        "pilot_authorization_content_sha256": (
            pilot_authorization["content_sha256"] if pilot_authorization is not None else None
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    invocation["content_sha256"] = sha256_json(invocation)
    invocation_path = output / "invocation.json"
    if invocation_path.is_file():
        prior_invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        _require(prior_invocation == invocation, "M6 SFT invocation changed across recovery")
    else:
        atomic_write_json(invocation_path, invocation)
    schedule = build_update_schedule(len(train), len(retention), seed=config.seed)
    recovery = load_recovery(output, sha256_json(schedule), input_sha)
    benchmark_path = output / "microbatch_benchmark.json"
    if benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        expected_benchmark = dict(benchmark)
        observed_benchmark_hash = expected_benchmark.pop("content_sha256", None)
        _require(observed_benchmark_hash == sha256_json(expected_benchmark), "M6 SFT benchmark self-hash drift")
        _require(benchmark.get("input_sha256") == input_sha and benchmark.get("passed") is True, "M6 SFT benchmark/input drift")
    else:
        model = load_trainable_lora_model(
            args.base_model,
            config,
            resume_adapter=recovery["adapter"] if recovery else None,
        )
        benchmark = benchmark_microbatches(
            model=model,
            examples=train,
            retention_examples=retention,
            tokenizer=tokenizer,
            config=config,
            output_path=benchmark_path,
            input_sha256=input_sha,
        )
        del model
        torch.cuda.empty_cache()
    report = train_mini_sft(
        base_model=args.base_model,
        tokenizer=tokenizer,
        train_examples=train,
        dev_examples=dev,
        retention_examples=retention,
        output_dir=output,
        input_sha256=input_sha,
        config=config,
        microbatch_size=int(benchmark["selected_microbatch_size"]),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
