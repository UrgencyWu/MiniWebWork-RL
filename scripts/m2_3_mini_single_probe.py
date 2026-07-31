#!/usr/bin/env python3
"""Canonical M2.3-mini rollout collector.

One process evaluates one policy under one explicit sampling distribution. The
artifact preserves exact prompt/completion tokens, raw model-policy
log-probabilities, actual sampling-distribution log-probabilities, environment
evidence, and terminal Verifier results. Infrastructure failures never enter
the reward stream.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if not SRC_DIR.is_dir():
    raise RuntimeError(f"Invalid source directory: {SRC_DIR}")
sys.path.insert(0, str(SRC_DIR))

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.model_agent.model_backend import ModelConfig, QwenTransformersBackend
from miniwebwork.model_agent.output_parser import parse
from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
from miniwebwork.rollout import (
    INFRASTRUCTURE_FAILURE,
    NO_FAILURE,
    POLICY_FAILURE,
    RolloutRecord,
    RolloutStep,
    derive_rollout_seed,
    strict_raw_policy_distribution,
    summarize_group,
)
import miniwebwork.model_agent.prompt_builder as prompt_builder

DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
DEFAULT_TASK_DIR = PROJECT_ROOT / "data" / "tasks" / "rollout_dev_no_solution_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m2_3_mini"
DEFAULT_K = 8
DEFAULT_SEED = 20260731
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 0
PROMPT_CONTRACT = "browser_agent_v2"
MAX_MODEL_TURNS = 25
MAX_OUTPUT_FAILURES = 3
STRICT_LOGPROB_MATCH_TOLERANCE = 5e-2


class Heartbeat:
    """Atomic progress record for Slurm timeout and signal diagnostics."""

    def __init__(
        self,
        path: Path,
        policy: str,
        temperature: float,
        top_p: float,
        top_k: int,
        total_tasks: int,
    ):
        self.path = path
        self.payload = {
            "policy": policy,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tasks_done": 0,
            "total_tasks": total_tasks,
            "current_task": "",
            "current_rollout": 0,
            "last_error": "",
        }
        self._started = time.time()
        self.write()

    def update(self, **changes) -> None:
        self.payload.update(changes)
        self.write()

    def finish_task(self) -> None:
        self.payload["tasks_done"] += 1
        self.payload["current_task"] = ""
        self.payload["current_rollout"] = 0
        self.write()

    def write(self) -> None:
        self.payload["uptime_s"] = round(time.time() - self._started, 1)
        self.payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic_json_write(self.path, self.payload)


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise ValueError(f"Directory contains no files: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(_file_sha256(file).encode("ascii"))
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _float_tag(value: float) -> str:
    return format(value, ".8g").replace(".", "p").replace("-", "m")


def _load_tasks(task_dir: Path, split: str) -> tuple[list[dict], str]:
    public_path = task_dir / f"{split}_public.jsonl"
    oracle_path = task_dir / f"{split}_oracle.jsonl"
    if not public_path.is_file() or not oracle_path.is_file():
        raise FileNotFoundError(
            f"Expected {public_path.name} and {oracle_path.name} in {task_dir}"
        )

    tasks: list[dict] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        public_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        task = json.loads(raw_line)
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Missing task_id in {public_path}:{line_number}")
        if task_id in seen:
            raise ValueError(f"Duplicate task_id {task_id!r} in {public_path}")
        seen.add(task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"No tasks found in {public_path}")

    digest = hashlib.sha256()
    for path in (public_path, oracle_path):
        digest.update(path.name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return tasks, digest.hexdigest()


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _disable_allocator_warmup() -> None:
    """Node-specific compatibility workaround; model loads on CPU first."""
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


def load_policy(
    base_model_path: str,
    adapter_dir: Path,
    temperature: float,
    top_p: float,
    top_k: int,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if prompt_builder.PROMPT_VERSION != PROMPT_CONTRACT:
        raise RuntimeError(
            f"Prompt contract drift: expected {PROMPT_CONTRACT}, "
            f"got {prompt_builder.PROMPT_VERSION}"
        )

    _disable_allocator_warmup()
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        torch_dtype=torch.bfloat16,
    )
    model.enable_adapter_layers()
    if not getattr(model, "active_adapters", None):
        raise RuntimeError("PEFT adapter loaded without an active adapter")
    model = model.to("cuda:0")
    model.eval()
    torch.cuda.synchronize()

    backend = QwenTransformersBackend(
        ModelConfig(
            model_path=base_model_path,
            max_new_tokens=128,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            dtype="bfloat16",
            device="cuda:0",
            enable_thinking=False,
            collect_policy_logprobs=True,
            collect_sampling_logprobs=True,
        )
    )
    backend._model = model
    backend._tokenizer = tokenizer
    backend._loaded = True
    prompt_builder.HISTORY_WINDOW = 5
    return backend, QwenBrowserAgent(backend, prompt_builder, parse)


def _step_from_attempt(turn: int, page_type: str, attempt) -> RolloutStep:
    return RolloutStep(
        turn=turn,
        page_type=page_type,
        prompt_hash=attempt.prompt_hash,
        prompt_token_ids=list(attempt.prompt_token_ids),
        raw_model_output=attempt.raw_output,
        generated_token_ids=list(attempt.generated_token_ids),
        token_logprobs=list(attempt.token_logprobs),
        sampling_logprobs=list(attempt.sampling_logprobs),
        strict_json_success=attempt.strict_json_success,
        fallback_used=attempt.fallback_used,
        schema_valid=attempt.schema_valid,
        schema_errors=list(attempt.errors),
        parsed_action=attempt.action.to_dict() if attempt.action else None,
        skipped=not attempt.schema_valid,
    )


def _infrastructure_record(
    *,
    task: dict,
    rollout_index: int,
    rollout_seed: int,
    policy: str,
    temperature: float,
    top_p: float,
    top_k: int,
    episode_id: str,
    reason: str,
    steps: list[RolloutStep],
    environment_steps: int,
) -> RolloutRecord:
    record = RolloutRecord(
        task_id=task["task_id"],
        task_type=task.get("task_type", ""),
        episode_id=episode_id,
        rollout_index=rollout_index,
        rollout_seed=rollout_seed,
        policy=policy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        success=False,
        reward=None,
        rollout_valid=False,
        failure_origin=INFRASTRUCTURE_FAILURE,
        termination_reason=reason[:500],
        model_turns=len(steps),
        environment_steps=environment_steps,
        schema_valid_count=sum(step.schema_valid for step in steps),
        schema_invalid_count=sum(not step.schema_valid for step in steps),
        steps=steps,
    )
    record.validate()
    return record


def _evidence_complete(attempt) -> bool:
    if attempt.output_tokens == 0:
        return not attempt.prompt_token_ids or len(attempt.prompt_token_ids) == attempt.input_tokens
    return (
        len(attempt.prompt_token_ids) == attempt.input_tokens
        and len(attempt.generated_token_ids) == attempt.output_tokens
        and len(attempt.token_logprobs) == attempt.output_tokens
        and len(attempt.sampling_logprobs) == attempt.output_tokens
    )


def _max_raw_sampling_difference(records: list[RolloutRecord]) -> float | None:
    differences: list[float] = []
    for record in records:
        if not record.rollout_valid:
            continue
        for step in record.steps:
            if not step.generated_token_ids:
                continue
            if len(step.token_logprobs) != len(step.sampling_logprobs):
                return None
            differences.extend(
                abs(raw - sampled)
                for raw, sampled in zip(step.token_logprobs, step.sampling_logprobs)
            )
    return max(differences) if differences else None


def _strict_group_distribution_compatible(
    records: list[RolloutRecord],
    *,
    parameter_compatible: bool,
) -> tuple[bool, float | None]:
    difference = _max_raw_sampling_difference(records)
    compatible = (
        parameter_compatible
        and difference is not None
        and difference <= STRICT_LOGPROB_MATCH_TOLERANCE
    )
    return compatible, difference


def run_rollout(
    task: dict,
    rollout_index: int,
    master_seed: int,
    policy: str,
    temperature: float,
    top_p: float,
    top_k: int,
    task_dir: Path,
    agent: QwenBrowserAgent,
) -> RolloutRecord:
    task_id = task["task_id"]
    rollout_seed = derive_rollout_seed(master_seed, task_id, rollout_index)
    _seed_all(rollout_seed)
    run_id = f"probe_{uuid.uuid4().hex[:10]}"
    episode_id = run_id
    steps: list[RolloutStep] = []
    environment_steps = 0
    output_failure_streak = 0
    success = False
    termination_reason = "max_model_turns"
    verification: dict = {}
    environment = None
    record: RolloutRecord | None = None

    try:
        environment = ProcurementBrowserEnv(
            max_steps=MAX_MODEL_TURNS,
            run_id=run_id,
            headless=True,
            task_dir=task_dir,
        )
        environment.set_agent_name("m2_3_mini_rollout_probe")
        observation = environment.reset(task_id)
        episode_id = observation.episode_id
        agent.reset(observation.task_id, observation.instruction)

        for turn in range(1, MAX_MODEL_TURNS + 1):
            attempt = agent.act(observation)
            step = _step_from_attempt(turn, observation.page_type, attempt)
            steps.append(step)

            backend_error = any(
                error.startswith(("generation_error", "rollout_evidence_error"))
                for error in attempt.errors
            )
            if backend_error or not _evidence_complete(attempt):
                record = _infrastructure_record(
                    task=task,
                    rollout_index=rollout_index,
                    rollout_seed=rollout_seed,
                    policy=policy,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    episode_id=episode_id,
                    reason=(
                        "model_backend_error"
                        if backend_error
                        else "missing_rollout_evidence"
                    ),
                    steps=steps,
                    environment_steps=environment_steps,
                )
                break

            if not attempt.schema_valid:
                output_failure_streak += 1
                if output_failure_streak >= MAX_OUTPUT_FAILURES:
                    termination_reason = "model_output_failure_limit"
                    break
                continue

            output_failure_streak = 0
            try:
                result = environment.step(attempt.action)
            except Exception as exc:
                step.skipped = False
                step.env_error_code = f"{type(exc).__name__}: {exc}"[:500]
                record = _infrastructure_record(
                    task=task,
                    rollout_index=rollout_index,
                    rollout_seed=rollout_seed,
                    policy=policy,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    episode_id=episode_id,
                    reason="environment_step_error",
                    steps=steps,
                    environment_steps=environment_steps,
                )
                break

            environment_steps += 1
            action_result = result.info.get("action_result", {})
            step.skipped = False
            step.env_action_success = action_result.get("success")
            step.env_error_code = action_result.get("error_code", "")
            step.terminated = result.terminated
            step.truncated = result.truncated

            next_page_type = (
                result.observation.page_type
                if result.observation is not None
                else "unknown"
            )
            agent.record_feedback(attempt, result, next_page_type)
            if result.observation is not None:
                observation = result.observation

            if result.terminated or result.truncated:
                success = float(result.reward) > 0.5
                termination_reason = result.info.get("termination_reason", "terminal")
                break

        if record is None:
            trajectory = environment.trajectory
            if trajectory is not None:
                verification = trajectory.verification or {}
                termination_reason = trajectory.termination_reason or termination_reason
                if verification:
                    success = bool(verification.get("success", success))

            record = RolloutRecord(
                task_id=task_id,
                task_type=task.get("task_type", ""),
                episode_id=episode_id,
                rollout_index=rollout_index,
                rollout_seed=rollout_seed,
                policy=policy,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                success=success,
                reward=1.0 if success else 0.0,
                rollout_valid=True,
                failure_origin=NO_FAILURE if success else POLICY_FAILURE,
                termination_reason=termination_reason,
                model_turns=len(steps),
                environment_steps=environment_steps,
                schema_valid_count=sum(step.schema_valid for step in steps),
                schema_invalid_count=sum(not step.schema_valid for step in steps),
                verification=verification,
                steps=steps,
            )
            record.validate()

    except Exception as exc:
        record = _infrastructure_record(
            task=task,
            rollout_index=rollout_index,
            rollout_seed=rollout_seed,
            policy=policy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            episode_id=episode_id,
            reason=f"rollout_exception:{type(exc).__name__}:{exc}",
            steps=steps,
            environment_steps=environment_steps,
        )
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as exc:
                record = _infrastructure_record(
                    task=task,
                    rollout_index=rollout_index,
                    rollout_seed=rollout_seed,
                    policy=policy,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    episode_id=episode_id,
                    reason=f"environment_cleanup_error:{type(exc).__name__}:{exc}",
                    steps=steps,
                    environment_steps=environment_steps,
                )

    if record is None:
        raise RuntimeError("Rollout ended without producing a record")
    record.validate()
    return record


def _metrics(records: list[RolloutRecord], groups: list[dict]) -> dict:
    valid = [record for record in records if record.rollout_valid]
    steps = [step for record in valid for step in record.steps]
    executed = [
        step
        for step in steps
        if not step.skipped and step.env_action_success is not None
    ]
    no_solution = [
        record for record in valid if record.task_type == "no_feasible_product"
    ]
    completion_steps = [step for step in steps if step.generated_token_ids]
    max_difference = _max_raw_sampling_difference(valid)
    return {
        "total_trajectories": len(records),
        "valid_trajectories": len(valid),
        "infrastructure_errors": len(records) - len(valid),
        "total_successes": sum(record.success for record in valid),
        "success_rate": sum(record.success for record in valid) / max(len(valid), 1),
        "total_model_turns": len(steps),
        "strict_json_rate": sum(step.strict_json_success for step in steps)
        / max(len(steps), 1),
        "schema_valid_action_rate": sum(step.schema_valid for step in steps)
        / max(len(steps), 1),
        "environment_action_success_rate": sum(
            bool(step.env_action_success) for step in executed
        )
        / max(len(executed), 1),
        "raw_policy_logprob_coverage": sum(
            len(step.token_logprobs) == len(step.generated_token_ids)
            for step in completion_steps
        )
        / max(len(completion_steps), 1),
        "sampling_logprob_coverage": sum(
            len(step.sampling_logprobs) == len(step.generated_token_ids)
            for step in completion_steps
        )
        / max(len(completion_steps), 1),
        "max_raw_sampling_logprob_abs_diff": max_difference,
        "premature_finish": sum(
            record.termination_reason == "premature_finish" for record in valid
        ),
        "no_solution_trajectories": len(no_solution),
        "no_solution_successes": sum(record.success for record in no_solution),
        "groups_with_reward_variance": sum(
            group["has_reward_variance"] for group in groups
        ),
        "groups_with_learning_signal": sum(
            group["has_learning_signal"] for group in groups
        ),
        "groups_update_distribution_compatible": sum(
            group["update_distribution_compatible"] for group in groups
        ),
        "groups_valid_for_grpo": sum(
            group["valid_for_grpo_update"] for group in groups
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical M2.3-mini rollout collector")
    parser.add_argument("--policy", required=True, choices=["A", "B"])
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.K <= 0:
        raise ValueError("K must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        raise ValueError("top-p must be finite and in (0, 1]")
    if args.top_k < 0:
        raise ValueError("top-k must be non-negative")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("max-tasks must be positive")

    task_dir = args.task_dir.expanduser().resolve()
    adapter_dir = args.adapter.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, task_source_hash = _load_tasks(task_dir, args.split)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    policy = "A_M2.2R" if args.policy == "A" else "B_M2.3-mini"
    adapter_hash = _directory_sha256(adapter_dir)
    prompt_hash = _file_sha256(Path(prompt_builder.__file__).resolve())
    git_sha = _git_sha()
    parameter_distribution_compatible = strict_raw_policy_distribution(
        args.temperature,
        args.top_p,
        args.top_k,
    )
    distribution_tag = (
        f"t{_float_tag(args.temperature)}_p{_float_tag(args.top_p)}_k{args.top_k}"
    )
    heartbeat = Heartbeat(
        output_dir / f"heartbeat_{args.policy}_{distribution_tag}.json",
        policy,
        args.temperature,
        args.top_p,
        args.top_k,
        len(tasks),
    )

    backend, agent = load_policy(
        args.base_model,
        adapter_dir,
        args.temperature,
        args.top_p,
        args.top_k,
    )
    records: list[RolloutRecord] = []
    groups: list[dict] = []
    started = time.time()
    try:
        for task_index, task in enumerate(tasks, start=1):
            task_id = task["task_id"]
            heartbeat.update(current_task=task_id, current_rollout=0)
            task_records: list[RolloutRecord] = []
            print(f"[{task_index}/{len(tasks)}] {task_id}", flush=True)

            for rollout_index in range(args.K):
                heartbeat.update(current_rollout=rollout_index + 1)
                record = run_rollout(
                    task,
                    rollout_index,
                    args.seed,
                    policy,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    task_dir,
                    agent,
                )
                records.append(record)
                task_records.append(record)
                if not record.rollout_valid:
                    heartbeat.update(last_error=record.termination_reason)
                print(
                    f"  k={rollout_index} valid={record.rollout_valid} "
                    f"success={record.success} term={record.termination_reason}",
                    flush=True,
                )

            group_compatible, max_difference = _strict_group_distribution_compatible(
                task_records,
                parameter_compatible=parameter_distribution_compatible,
            )
            group = summarize_group(
                task_records,
                args.K,
                update_distribution_compatible=group_compatible,
            ).to_dict()
            group["max_raw_sampling_logprob_abs_diff"] = max_difference
            group["strict_logprob_match_tolerance"] = STRICT_LOGPROB_MATCH_TOLERANCE
            groups.append(group)
            heartbeat.finish_task()
            _atomic_json_write(
                output_dir / f"incremental_{args.policy}_{distribution_tag}.json",
                {
                    "schema_version": "3.3",
                    "complete": False,
                    "git_sha": git_sha,
                    "policy": policy,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "parameter_distribution_compatible": parameter_distribution_compatible,
                    "strict_logprob_match_tolerance": STRICT_LOGPROB_MATCH_TOLERANCE,
                    "K": args.K,
                    "seed": args.seed,
                    "task_source_sha256": task_source_hash,
                    "adapter_sha256": adapter_hash,
                    "groups": groups,
                    "records": [record.to_dict() for record in records],
                },
            )

        result = {
            "schema_version": "3.3",
            "phase": "m2_3_mini_rollout_collection",
            "complete": True,
            "git_sha": git_sha,
            "policy": policy,
            "base_model": args.base_model,
            "adapter_path": str(adapter_dir),
            "adapter_sha256": adapter_hash,
            "prompt_contract": PROMPT_CONTRACT,
            "prompt_builder_sha256": prompt_hash,
            "chat_template_sha256": backend.get_chat_template_hash(),
            "task_dir": str(task_dir),
            "task_source_sha256": task_source_hash,
            "split": args.split,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "parameter_distribution_compatible": parameter_distribution_compatible,
            "strict_logprob_match_tolerance": STRICT_LOGPROB_MATCH_TOLERANCE,
            "K": args.K,
            "seed": args.seed,
            "max_model_turns": MAX_MODEL_TURNS,
            "max_output_failures": MAX_OUTPUT_FAILURES,
            "logprob_contract": {
                "token_logprobs": "raw model-policy log probabilities",
                "sampling_logprobs": "post generation-processor behavior probabilities",
            },
            "metrics": _metrics(records, groups),
            "groups": groups,
            "records": [record.to_dict() for record in records],
            "elapsed_s": round(time.time() - started, 1),
        }
        output_path = output_dir / (
            f"single_probe_{args.policy}_{distribution_tag}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        _atomic_json_write(output_path, result)
        print(f"Results saved: {output_path}", flush=True)
        print(json.dumps(result["metrics"], indent=2, ensure_ascii=False), flush=True)
    finally:
        backend.unload()
        del agent, backend
        gc.collect()


if __name__ == "__main__":
    main()
