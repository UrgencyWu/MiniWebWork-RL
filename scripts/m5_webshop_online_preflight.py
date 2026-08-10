#!/usr/bin/env python3
"""Run the real, recoverable M5 WebShop K4 online GPU preflight."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from transformers import AutoTokenizer

from miniwebwork.long_horizon_rl.adapter_view import build_vllm_adapter_view
from miniwebwork.long_horizon_rl.contracts import (
    atomic_write_json,
    canonical_json_bytes,
    directory_sha256,
    sha256_file,
    sha256_json,
)
from miniwebwork.long_horizon_rl.model_manifest import validate_base_model_manifest
from miniwebwork.long_horizon_rl.vllm_backend import (
    AsyncVLLMGenerationEngine,
    RolloutRequestContext,
    ThreadsafeVLLMBackend,
)
from miniwebwork.m5_webshop_protocol import (
    eligible_goal_indices,
    load_protocol,
    task_id_for_goal_index,
)
from miniwebwork.model_agent.agent_loop import run_model_episode
from miniwebwork.webshop_rl.agent import QwenWebShopAgent
from miniwebwork.webshop_rl.credit import ANCHOR_METHOD, BASELINE_METHOD
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment
from miniwebwork.webshop_rl.online_training import (
    M5VLLMBackendConfig,
    audit_collection,
    build_committed_group,
    train_policy_preflight,
    trajectory_from_episode,
    validate_committed_group,
    validate_learner_report,
)

PREFLIGHT_SEED = 20260810
GROUP_COUNT = 32
CONCURRENT_GROUPS = 8
MAX_ATTEMPTS = 4
SFT_PRODUCER_GIT_SHA = "9cedc2a7cc8cb557f5b583d3c9dd7ae3502486c1"
SFT_PRODUCER_PROTOCOL_SHA256 = "f6a6a2ded1fd6c926873d920d921937d32734aca0812a6aa72d54c1cfe13535f"
SFT_ADAPTER_SHA256 = "c97c9265fe0043a8cda59908713429eef9e13d9619261a144c13cfa5fb4d7334"
SFT_CORPUS_SHA256 = {
    "train.jsonl": "e91aea8214a33f1750f05bc64cb0fc26fc582c7cd1bced50153c91cd1674ace1",
    "dev.jsonl": "641a0f133bbb356c98397e9d132ecd0f4e7b3a7f0b9e73d4f2247155663d00d2",
    "corpus_audit.json": "f923b92d20c3378119124c486a324fc9878fa7184514815be90ab661d95f4c81",
    "token_audit.json": "eb34c78f19557dad72a67fb4f725a42c6059c9734370d0b35b842cf98d4f5099",
}
AGENT_PROMPT_SHA256 = "c33178c5c9a4e3f9cf683e6a1debb1e2f762a3cd9f27338292c8e8fa34f712aa"
BASE_MODEL_MANIFEST_SHA256 = "290ecd9ec4eaa1f5ac6927b10e9cb4c600d22aec78a6d743baa8a01d72c1b7a3"
BASE_MODEL_FUNCTIONAL_SHA256 = "6b2cdb9cf894cec7eb1dcef2a57682a9d73e22f2a81a58ef3b1f32854f031b85"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _assert_clean() -> None:
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    _require(not status, "M5 online preflight requires a clean tracked worktree")


def _hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["content_sha256"] = sha256_json(value)
    return value


class DurableTokenLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, payload: Mapping[str, Any]) -> None:
        line = canonical_json_bytes(dict(payload)) + b"\n"
        with self._lock, self.path.open("ab", buffering=0) as handle:
            handle.write(line)
            os.fsync(handle.fileno())

    def audit(self) -> dict[str, Any]:
        records = []
        if self.path.is_file():
            with self.path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        item = json.loads(line)
                    except Exception as exc:
                        raise ValueError(f"M5 token ledger line {line_number} is corrupt") from exc
                    _require(isinstance(item, dict), "M5 token ledger record is malformed")
                    records.append(item)
        return {
            "record_count": len(records),
            "generated_action_tokens": sum(int(item["generated_action_tokens"]) for item in records),
            "sha256": sha256_file(self.path) if self.path.is_file() else hashlib.sha256(b"").hexdigest(),
        }


class TelemetryRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.process: subprocess.Popen | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("wb")
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
                "--loop=2",
            ],
            stdout=handle,
            stderr=subprocess.DEVNULL,
        )
        self._handle = handle
        return self

    def __exit__(self, *_):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self._handle.close()


def _telemetry_audit(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        if path.is_file():
            rows.extend(row for row in csv.reader(path.open(encoding="utf-8")) if len(row) >= 8)
    _require(rows, "M5 GPU telemetry is empty")
    gpu = [float(row[3].strip()) for row in rows]
    memory = [float(row[5].strip()) for row in rows]
    return {
        "paths": [str(path) for path in paths],
        "file_sha256": {str(path): sha256_file(path) for path in paths},
        "samples": len(rows),
        "mean_gpu_utilization_fraction": statistics.fmean(gpu) / 100.0,
        "median_gpu_utilization_fraction": statistics.median(gpu) / 100.0,
        "p95_gpu_utilization_fraction": sorted(gpu)[min(len(gpu) - 1, math.ceil(0.95 * len(gpu)) - 1)] / 100.0,
        "maximum_memory_used_mib": max(memory),
    }


def _task_order() -> list[str]:
    roster = eligible_goal_indices("train")
    ordered = sorted(roster, key=lambda index: hashlib.sha256(f"{PREFLIGHT_SEED}|{index}".encode()).digest())
    return [task_id_for_goal_index(index) for index in ordered[:GROUP_COUNT]]


def _minimal_sft_compatibility(*, protocol: Mapping[str, Any], base_model: Path, adapter: Path) -> dict[str, Any]:
    """Check only bytes and semantics directly consumed by online inference."""

    _require(adapter.is_dir() and directory_sha256(adapter) == SFT_ADAPTER_SHA256, "audited SFT adapter is missing or incomplete")
    required_adapter_files = {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    _require(required_adapter_files <= {path.name for path in adapter.iterdir() if path.is_file()}, "audited SFT adapter lacks required files")
    corpus_root = PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "preflight" / "sft_corpus"
    corpus = {name: sha256_file(corpus_root / name) for name in SFT_CORPUS_SHA256}
    _require(corpus == SFT_CORPUS_SHA256, "SFT corpus bytes changed before online inference")
    prompt_path = PROJECT_ROOT / "prompts" / "webshop_agent_v1_compact.txt"
    prompt_sha = sha256_file(prompt_path)
    _require(prompt_sha == AGENT_PROMPT_SHA256, "WebShop inference prompt changed after SFT")
    model = validate_base_model_manifest(verify_files=True)
    _require(model["sha256"] == BASE_MODEL_MANIFEST_SHA256, "base model manifest changed")
    _require(model["payload"]["functional_file_set_sha256"] == BASE_MODEL_FUNCTIONAL_SHA256, "base model bytes changed")
    base_tokenizer = AutoTokenizer.from_pretrained(str(base_model), local_files_only=True, trust_remote_code=True)
    adapter_tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True, trust_remote_code=True)
    tokenizer_summary = lambda tokenizer: {
        "vocab_sha256": sha256_json(tokenizer.get_vocab()),
        "added_vocab_sha256": sha256_json(tokenizer.get_added_vocab()),
        "special_tokens_sha256": sha256_json(tokenizer.special_tokens_map),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest(),
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    base_tokenizer_summary = tokenizer_summary(base_tokenizer)
    adapter_tokenizer_summary = tokenizer_summary(adapter_tokenizer)
    _require(base_tokenizer_summary == adapter_tokenizer_summary, "SFT tokenizer semantics changed")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    expected_lora = protocol["sft"]["lora"]
    lora_matches = (
        adapter_config.get("r") == expected_lora["r"]
        and adapter_config.get("lora_alpha") == expected_lora["alpha"]
        and math.isclose(float(adapter_config.get("lora_dropout")), float(expected_lora["dropout"]))
        and set(adapter_config.get("target_modules", [])) == set(expected_lora["target_modules"])
        and adapter_config.get("base_model_name_or_path") == str(base_model)
    )
    _require(lora_matches, "SFT LoRA/base-model configuration changed")
    return {
        "passed": True,
        "producer_git_sha": SFT_PRODUCER_GIT_SHA,
        "producer_protocol_sha256": SFT_PRODUCER_PROTOCOL_SHA256,
        "consumer_git_sha": _git_sha(),
        "adapter_sha256": SFT_ADAPTER_SHA256,
        "corpus_sha256": corpus,
        "agent_prompt_sha256": prompt_sha,
        "base_model_manifest_sha256": model["sha256"],
        "base_model_functional_file_set_sha256": model["payload"]["functional_file_set_sha256"],
        "tokenizer_semantics": base_tokenizer_summary,
        "lora": expected_lora,
    }


def _write_attempt(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, _hashed(payload))


async def _collect_one_group(
    *,
    engine: AsyncVLLMGenerationEngine,
    loop: asyncio.AbstractEventLoop,
    event_loop_thread_id: int,
    task_id: str,
    group_index: int,
    groups_root: Path,
    attempts_root: Path,
    ledger: DurableTokenLedger,
    lineage: Mapping[str, str],
    base_url: str,
) -> dict[str, Any]:
    group_id = f"g{group_index:04d}"
    destination = groups_root / f"{group_id}.json"
    if destination.is_file():
        return validate_committed_group(json.loads(destination.read_text(encoding="utf-8")))

    for attempt_index in range(MAX_ATTEMPTS):
        async def one_rollout(rollout_index: int):
            trajectory_id = f"{group_id}.a{attempt_index}.r{rollout_index}"
            context = RolloutRequestContext(
                run_seed=PREFLIGHT_SEED,
                iteration_index=0,
                group_id=group_id,
                attempt_index=attempt_index,
                trajectory_id=trajectory_id,
                rollout_index=rollout_index,
            )
            backend = ThreadsafeVLLMBackend(
                engine=engine,
                event_loop=loop,
                context=context,
                timeout_seconds=900,
                event_loop_thread_id=event_loop_thread_id,
            )
            env = WebShopHTTPEnvironment(base_url=base_url, split="train", timeout_seconds=120)
            agent = QwenWebShopAgent(backend)

            def charge(turn: dict[str, Any]) -> None:
                generated_ids = turn.get("generated_token_ids", [])
                ledger.append(
                    {
                        "schema_version": "m5_webshop_generated_turn_ledger_v1",
                        "group_id": group_id,
                        "task_id": task_id,
                        "attempt_index": attempt_index,
                        "rollout_index": rollout_index,
                        "turn_index": int(turn.get("model_turn_index", 0)),
                        "request_id": str(turn.get("request_id", "")),
                        "generated_action_tokens": len(generated_ids),
                        "generated_token_sha256": sha256_json(generated_ids),
                        "adapter_sha256": lineage["adapter_sha256"],
                    }
                )

            try:
                result = await asyncio.to_thread(
                    run_model_episode,
                    task_id,
                    env,
                    agent,
                    18,
                    15,
                    charge,
                    None,
                )
                return result
            finally:
                env.close()

        episodes = await asyncio.gather(*(one_rollout(index) for index in range(4)))
        attempt = {
            "schema_version": "m5_webshop_k4_attempt_v1",
            "git_sha": lineage["git_sha"],
            "protocol_sha256": lineage["protocol_sha256"],
            "group_id": group_id,
            "task_id": task_id,
            "attempt_index": attempt_index,
            "rollouts": [
                {
                    "rollout_index": index,
                    "rollout_valid": episode.get("rollout_valid"),
                    "reward": episode.get("reward"),
                    "termination_reason": episode.get("termination_reason"),
                    "error": episode.get("error"),
                    "model_turns": episode.get("model_turns"),
                }
                for index, episode in enumerate(episodes)
            ],
        }
        _write_attempt(attempts_root / f"{group_id}.a{attempt_index}.json", attempt)
        if not all(episode.get("rollout_valid") is True for episode in episodes):
            continue
        trajectories = [
            trajectory_from_episode(
                episode,
                trajectory_id=f"{group_id}.a{attempt_index}.r{index}",
                rollout_index=index,
                adapter_sha256=lineage["adapter_sha256"],
                rollout_adapter_sha256=lineage["rollout_adapter_sha256"],
                adapter_semantic_sha256=lineage["adapter_semantic_sha256"],
            )
            for index, episode in enumerate(episodes)
        ]
        group = build_committed_group(
            task_id=task_id,
            group_id=group_id,
            attempt_index=attempt_index,
            trajectories=trajectories,
            **lineage,
        )
        atomic_write_json(destination, group)
        return group
    raise RuntimeError(f"M5 group exhausted infrastructure retries: {group_id}")


def _attempt_validity(attempts_root: Path) -> dict[str, Any]:
    valid = total = 0
    for path in sorted(attempts_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = dict(payload)
        observed = expected.pop("content_sha256", None)
        _require(observed == sha256_json(expected), "M5 attempt self-hash drift")
        for rollout in payload["rollouts"]:
            total += 1
            valid += rollout["rollout_valid"] is True
    _require(total > 0, "M5 attempt audit is empty")
    return {"trajectory_attempt_count": total, "infrastructure_valid_trajectory_count": valid, "infrastructure_valid_fraction": valid / total}


async def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_bundle = load_protocol()
    protocol = protocol_bundle["payload"]
    _assert_clean()
    _require(_git_sha() == protocol_bundle["git_sha"], "M5 online Git lineage drift")
    _require(protocol["formal_submission_allowed"] is False, "M5 preflight cannot be formal")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M5 online preflight requires one visible GPU")
    initial_adapter = args.initial_adapter.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    compatibility = _minimal_sft_compatibility(
        protocol=protocol,
        base_model=base_model,
        adapter=initial_adapter,
    )
    groups_root = output / "groups"
    attempts_root = output / "attempts"
    groups_root.mkdir(exist_ok=True)
    attempts_root.mkdir(exist_ok=True)
    ledger = DurableTokenLedger(output / "generated_turn_ledger.jsonl")
    view_audit = build_vllm_adapter_view(
        source_adapter=initial_adapter,
        destination=output / "initial_rollout_adapter",
        base_model=base_model,
    )
    lineage = {
        "git_sha": protocol_bundle["git_sha"],
        "protocol_sha256": protocol_bundle["sha256"],
        "adapter_sha256": directory_sha256(initial_adapter),
        "rollout_adapter_sha256": view_audit["view_directory_sha256"],
        "adapter_semantic_sha256": view_audit["semantic_tensor_sha256"],
    }
    invocation_path = output / "invocation.json"
    invocation = _hashed(
        {
            "schema_version": "m5_webshop_online_preflight_invocation_v1",
            "formal_training": False,
            "git_sha": lineage["git_sha"],
            "protocol_sha256": lineage["protocol_sha256"],
            "task_order": _task_order(),
            "task_order_sha256": sha256_json(_task_order()),
            "group_count": GROUP_COUNT,
            "K": 4,
            "parallel_lanes": 32,
            "base_model": str(base_model),
            "initial_adapter": str(initial_adapter),
            "sft_compatibility": compatibility,
            **{key: value for key, value in lineage.items() if key.endswith("sha256")},
        }
    )
    if invocation_path.is_file():
        _require(json.loads(invocation_path.read_text(encoding="utf-8")) == invocation, "M5 online invocation drift across recovery")
    else:
        atomic_write_json(invocation_path, invocation)

    config = M5VLLMBackendConfig(
        base_model=str(base_model),
        adapter_path=str(initial_adapter),
        adapter_sha256=lineage["adapter_sha256"],
        rollout_adapter_path=str(output / "initial_rollout_adapter"),
        rollout_adapter_sha256=lineage["rollout_adapter_sha256"],
        adapter_semantic_sha256=lineage["adapter_semantic_sha256"],
        seed=PREFLIGHT_SEED,
    )
    engine = await AsyncVLLMGenerationEngine.create(config)
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    generation_telemetry = output / "telemetry" / f"generation_{job_id}.csv"
    try:
        resumed_group_count = len(list(groups_root.glob("g*.json")))
        pending = [
            (index, task_id)
            for index, task_id in enumerate(_task_order())
            if not (groups_root / f"g{index:04d}.json").is_file()
        ]
        with TelemetryRecorder(generation_telemetry):
            loop = asyncio.get_running_loop()
            loop_thread = threading.get_ident()
            for start in range(0, len(pending), CONCURRENT_GROUPS):
                batch = pending[start : start + CONCURRENT_GROUPS]
                await asyncio.gather(
                    *(
                        _collect_one_group(
                            engine=engine,
                            loop=loop,
                            event_loop_thread_id=loop_thread,
                            task_id=task_id,
                            group_index=index,
                            groups_root=groups_root,
                            attempts_root=attempts_root,
                            ledger=ledger,
                            lineage=lineage,
                            base_url=args.base_url,
                        )
                        for index, task_id in batch
                    )
                )

        groups = [validate_committed_group(json.loads((groups_root / f"g{index:04d}.json").read_text(encoding="utf-8"))) for index in range(GROUP_COUNT)]
        ledger_audit = ledger.audit()
        collection = audit_collection(groups, all_generated_action_tokens=ledger_audit["generated_action_tokens"])
        attempt = _attempt_validity(attempts_root)
        collection_checks = {
            "infrastructure_valid": attempt["infrastructure_valid_fraction"] >= 0.99,
            "mixed_reward": collection["mixed_reward_group_fraction"] >= 0.20,
            "success_lower": collection["success_rate"] >= 0.03,
            "success_upper": collection["success_rate"] <= 0.70,
            "initial_anchor": collection["initial_shared_anchor_group_fraction"] == 1.0,
            "informative_micro": collection["informative_micro_turn_fraction"] >= 0.02,
            "shared_noninitial": collection["shared_noninitial_group_fraction"] >= 0.05,
        }
        _require(all(collection_checks.values()), f"M5 collection gates failed: {collection_checks}")
        collection_report = _hashed(
            {
                "schema_version": "m5_webshop_online_collection_report_v1",
                "git_sha": lineage["git_sha"],
                "protocol_sha256": lineage["protocol_sha256"],
                "ledger": ledger_audit,
                "attempts": attempt,
                "metrics": collection,
                "checks": collection_checks,
                "recovery": {
                    "same_root_resume_supported": True,
                    "resumed_group_count": resumed_group_count,
                    "successor_job_id": (output / "successor_job_id").read_text(encoding="utf-8").strip() if (output / "successor_job_id").is_file() else None,
                },
                "group_sha256": {group["group_id"]: group["content_sha256"] for group in groups},
                "passed": True,
            }
        )
        atomic_write_json(output / "collection_report.json", collection_report)

        sleep_initial = await engine.switch_to_learner()
        learner_reports = {}
        for method in (BASELINE_METHOD, ANCHOR_METHOD):
            telemetry_path = output / "telemetry" / f"learner_{method}_{job_id}.csv"
            existing_report = output / "learners" / method / "learner_report.json"
            if existing_report.is_file():
                learner_reports[method] = validate_learner_report(existing_report)
            else:
                with TelemetryRecorder(telemetry_path):
                    learner_reports[method] = train_policy_preflight(
                        method=method,
                        groups=groups,
                        all_generated_action_tokens=ledger_audit["generated_action_tokens"],
                        base_model=base_model,
                        initial_adapter=initial_adapter,
                        output_root=output / "learners",
                        protocol=protocol,
                        git_sha=lineage["git_sha"],
                        protocol_sha256=lineage["protocol_sha256"],
                    )
            validate_learner_report(Path(learner_reports[method]["output_adapter"]).parent / "learner_report.json")

        generation_audit = _telemetry_audit(sorted((output / "telemetry").glob("generation_*.csv")))
        learner_audit = _telemetry_audit(sorted((output / "telemetry").glob("learner_*.csv")))
        telemetry_checks = {
            "generation_median": generation_audit["median_gpu_utilization_fraction"] >= 0.60,
            "learner_median": learner_audit["median_gpu_utilization_fraction"] >= 0.80,
        }
        _require(all(telemetry_checks.values()), f"M5 GPU utilization gates failed: {telemetry_checks}")
        report = _hashed(
            {
                "schema_version": "m5_webshop_online_preflight_report_v1",
                "complete": True,
                "passed": True,
                "formal_training": False,
                "git_sha": lineage["git_sha"],
                "protocol_sha256": lineage["protocol_sha256"],
                "sft_compatibility": compatibility,
                "invocation_sha256": sha256_file(invocation_path),
                "collection_report_sha256": sha256_file(output / "collection_report.json"),
                "initial_sleep": sleep_initial,
                "learners": {method: {"report_sha256": sha256_file(Path(value["output_adapter"]).parent / "learner_report.json"), "optimizer_updates": value["optimizer_updates"], "effective_optimizer_action_token_fraction": value["effective_optimizer_action_token_fraction"]} for method, value in learner_reports.items()},
                "generation_telemetry": generation_audit,
                "learner_telemetry": learner_audit,
                "telemetry_checks": telemetry_checks,
            }
        )
        atomic_write_json(output / "preflight_report.json", report)
        return report
    finally:
        try:
            if engine._phase == "learner":
                pass
            elif engine._phase == "generation":
                await engine.switch_to_learner()
        except Exception:
            pass
        engine.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = asyncio.run(run(args))
        print(json.dumps(report, indent=2, sort_keys=True))
    except BaseException as exc:
        output = args.output_dir.expanduser().resolve()
        # Slurm cancellation is the intended first-allocation recovery event;
        # SIGTERM must not be rewritten as an algorithm failure artifact.
        if isinstance(exc, KeyboardInterrupt):
            raise
        failure = _hashed(
            {
                "schema_version": "m5_webshop_online_preflight_failure_v1",
                "complete": True,
                "passed": False,
                "git_sha": _git_sha(),
                "exception_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "traceback": traceback.format_exc()[-16000:],
            }
        )
        atomic_write_json(output / "preflight_failure.json", failure)
        raise


if __name__ == "__main__":
    main()
