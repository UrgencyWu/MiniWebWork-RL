#!/usr/bin/env python3
"""Canonical single-policy, single-temperature rollout probe.

One Slurm job evaluates one policy/temperature pair.  The script preserves the
exact generated tokens and old-policy log-probabilities required by M3.0 while
keeping infrastructure failures out of the reward stream.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
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
    summarize_group,
)
import miniwebwork.model_agent.prompt_builder as prompt_builder

DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
DEFAULT_TASK_DIR = PROJECT_ROOT / "data" / "tasks" / "rollout_dev_no_solution_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m2_3_mini"
DEFAULT_K = 8
DEFAULT_SEED = 20260731
MAX_OUTPUT_FAILURES = 3
MAX_MODEL_TURNS = 25
PROMPT_CONTRACT = "browser_agent_v2"


class Heartbeat:
    def __init__(self, path: Path, policy: str, temperature: float, total_tasks: int):
        self.path = path
        self.policy = policy
        self.temperature = temperature
        self.total_tasks = total_tasks
        self.tasks_done = 0
        self.current_task = ""
        self.current_rollout = 0
        self.last_error = ""
        self.started = time.time()
        self.write()

    def update(self, *, task: str | None = None, rollout: int | None = None, error: str = ""):
        if task is not None:
            self.current_task = task
        if rollout is not None:
            self.current_rollout = rollout
        if error:
            self.last_error = error[:500]
        self.write()

    def finish_task(self):
        self.tasks_done += 1
        self.current_task = ""
        self.current_rollout = 0
        self.write()

    def write(self):
        payload = {
            "policy": self.policy,
            "temperature": self.temperature,
            "uptime_s": round(time.time() - self.started, 1),
            "tasks_done": self.tasks_done,
            "total_tasks": self.total_tasks,
            "current_task": self.current_task,
            "current_rollout": self.current_rollout,
            "last_error": self.last_error,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _atomic_json_write(self.path, payload)


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise ValueError(f"Adapter directory contains no files: {path}")
    for file in files:
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(_sha256_file(file).encode("ascii"))
    return digest.hexdigest()


def _task_source_hash(task_dir: Path, split: str) -> str:
    public_file = task_dir / f"{split}_public.jsonl"
    oracle_file = task_dir / f"{split}_oracle.jsonl"
    if not public_file.is_file() or not oracle_file.is_file():
        raise FileNotFoundError(
            f"Expected {public_file.name} and {oracle_file.name} in {task_dir}"
        )
    digest = hashlib.sha256()
    for file in (public_file, oracle_file):
        digest.update(file.name.encode("utf-8"))
        digest.update(_sha256_file(file).encode("ascii"))
    return digest.hexdigest()


def _load_split_tasks(task_dir: Path, split: str) -> list[dict]:
    path = task_dir / f"{split}_public.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Task file not found: {path}")
    tasks: list[dict] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        task = json.loads(raw_line)
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Missing task_id in {path}:{line_number}")
        if task_id in seen:
            raise ValueError(f"Duplicate task_id {task_id!r} in {path}")
        seen.add(task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tasks


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _disable_transformers_allocator_warmup() -> None:
    """Compatibility workaround for the audited cluster image.

    The model is loaded on CPU and moved to the single Slurm-visible GPU only
    after PEFT attachment.  Transformers allocator warmup is therefore not
    needed and has caused driver-level failures on this specific node.
    """
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


def load_policy(base_model_path: str, adapter_path: Path, temperature: float):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
    if prompt_builder.PROMPT_VERSION != PROMPT_CONTRACT:
        raise RuntimeError(
            f"Prompt contract drift: expected {PROMPT_CONTRACT}, got {prompt_builder.PROMPT_VERSION}"
        )

    _disable_transformers_allocator_warmup()
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
        adapter_path,
        torch_dtype=torch.bfloat16,
    )
    model.enable_adapter_layers()
    if not getattr(model, "active_adapters", None):
        raise RuntimeError("PEFT adapter loaded without an active adapter")
    model = model.to("cuda:0")
    model.eval()
    torch.cuda.synchronize()

    config = ModelConfig(
        model_path=base_model_path,
        max_new_tokens=128,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        dtype="bfloat16",
        device="cuda:0",
        enable_thinking=False,
        collect_policy_logprobs=True,
        collect_sampling_logprobs=False,
    )
    backend = QwenTransformersBackend(config)
    backend._model = model
    backend._tokenizer = tokenizer
    backend._loaded = True

    prompt_builder.HISTORY_WINDOW = 5
    agent = QwenBrowserAgent(backend, prompt_builder, parse)
    return backend, agent


def _infrastructure_record(
    *,
    task: dict,
    policy: str,
    temperature: float,
    rollout_index: int,
    rollout_seed: int,
    episode_id: str,
    reason: str,
    steps: list[RolloutStep] | None = None,
) -> RolloutRecord:
    steps = steps or []
    valid_count = sum(1 for step in steps if step.schema_valid)
    return RolloutRecord(
        task_id=task["task_id"],
        task_type=task.get("task_type", ""),
        episode_id=episode_id,
        rollout_index=rollout_index,
        rollout_seed=rollout_seed,
        policy=policy,
        temperature=temperature,
        success=False,
        reward=None,
        rollout_valid=False,
        failure_origin=INFRASTRUCTURE_FAILURE,
        termination_reason=reason,
        model_turns=len(steps),
        environment_steps=sum(1 for step in steps if not step.skipped),
        schema_valid_count=valid_count,
        schema_invalid_count=len(steps) - valid_count,
        steps=steps,
    )


def run_rollout(
    task: dict,
    rollout_index: int,
    master_seed: int,
    policy: str,
    temperature: float,
    task_dir: Path,
    agent: QwenBrowserAgent,
) -> RolloutRecord:
    task_id = task["task_id"]
    rollout_seed = derive_rollout_seed(master_seed, task_id, rollout_index)
    _seed_everything(rollout_seed)
    run_id = f"probe_{uuid.uuid4().hex[:10]}"
    steps: list[RolloutStep] = []
    environment_steps = 0
    output_failure_streak = 0
    success = False
    reward = 0.0
    termination_reason = "max_model_turns"
    verification: dict = {}
    env = ProcurementBrowserEnv(
        max_steps=MAX_MODEL_TURNS,
        run_id=run_id,
        headless=True,
        task_dir=task_dir,
    )

    try:
        env.set_agent_name("m2_3_mini_rollout_probe")
        observation = env.reset(task_id)
        agent.reset(observation.task_id, observation.instruction)

        for turn in range(1, MAX_MODEL_TURNS + 1):
            try:
                attempt = agent.act(observation)
            except Exception as exc:
                return _infrastructure_record(
                    task=task,
                    policy=policy,
                    temperature=temperature,
                    rollout_index=rollout_index,
                    rollout_seed=rollout_seed,
                    episode_id=run_id,
                    reason=f"agent_exception:{type(exc).__name__}:{exc}",
                    steps=steps,
                )

            step = RolloutStep(
                turn=turn,
                page_type=observation.page_type,
                prompt_hash=attempt.prompt_hash,
                raw_model_output=attempt.raw_output[:1000],
                generated_token_ids=list(attempt.generated_token_ids),
                token_logprobs=list(attempt.token_logprobs),
                schema_valid=attempt.schema_valid,
                schema_errors=list(attempt.errors),
                parsed_action=attempt.action.to_dict() if attempt.action else None,
                skipped=not attempt.schema_valid,
            )

            if any(error.startswith(("generation_error", "rollout_evidence_error")) for error in attempt.errors):
                steps.append(step)
                return _infrastructure_record(
                    task=task,
                    policy=policy,
                    temperature=temperature,
                    rollout_index=rollout_index,
                    rollout_seed=rollout_seed,
                    episode_id=run_id,
                    reason="model_backend_error",
                    steps=steps,
                )

            if not attempt.schema_valid:
                steps.append(step)
                output_failure_streak += 1
                if output_failure_streak >= MAX_OUTPUT_FAILURES:
                    termination_reason = "model_output_failure_limit"
                    break
                continue

            output_failure_streak = 0
            try:
                result = env.step(attempt.action)
            except Exception as exc:
                step.skipped = False
                step.env_error_code = f"{type(exc).__name__}:{exc}"[:500]
                steps.append(step)
                return _infrastructure_record(
                    task=task,
                    policy=policy,
                    temperature=temperature,
                    rollout_index=rollout_index,
                    rollout_seed=rollout_seed,
                    episode_id=run_id,
                    reason="environment_step_error",
                    steps=steps,
                )

            environment_steps += 1
            action_result = result.info.get("action_result", {})
            step.skipped = False
            step.env_action_success = action_result.get("success")
            step.env_error_code = action_result.get("error_code", "")
            step.terminated = result.terminated
            step.truncated = result.truncated
            steps.append(step)

            if result.observation is not None:
                agent.record_feedback(attempt, result, result.observation.page_type)
                observation = result.observation
            else:
                agent.record_feedback(attempt, result, "unknown")

            if result.terminated or result.truncated:
                reward = float(result.reward)
                success = reward > 0.5
                termination_reason = result.info.get("termination_reason", "terminal")
                break

        trajectory = env.trajectory
        if trajectory is not None:
            verification = trajectory.verification or {}
            if trajectory.termination_reason:
                termination_reason = trajectory.termination_reason
            if verification:
                success = bool(verification.get("success", success))
                reward = 1.0 if success else 0.0

        record = RolloutRecord(
            task_id=task_id,
            task_type=task.get("task_type", ""),
            episode_id=getattr(env, "_episode_id", "") or run_id,
            rollout_index=rollout_index,
            rollout_seed=rollout_seed,
            policy=policy,
            temperature=temperature,
            success=success,
            reward=1.0 if success else 0.0,
            rollout_valid=True,
            failure_origin=NO_FAILURE if success else POLICY_FAILURE,
            termination_reason=termination_reason,
            model_turns=len(steps),
            environment_steps=environment_steps,
            schema_valid_count=sum(1 for step in steps if step.schema_valid),
            schema_invalid_count=sum(1 for step in steps if not step.schema_valid),
            verification=verification,
            steps=steps,
        )
        record.validate()
        return record
    finally:
        try:
            env.close()
        except Exception:
            # A shutdown failure happens after the rollout record has already
            # been determined.  It is surfaced to the job log and the next
            # environment creation will fail fast if resources leaked.
            print(f"[WARN] Environment cleanup failed for {task_id}/{rollout_index}", flush=True)


def _run_metrics(records: list[RolloutRecord], groups: list[dict]) -> dict:
    valid = [record for record in records if record.rollout_valid]
    valid_turns = sum(record.model_turns for record in valid)
    schema_valid = sum(record.schema_valid_count for record in valid)
    no_solution = [record for record in valid if record.task_type == "no_feasible_product"]
    return {
        "total_trajectories": len(records),
        "valid_trajectories": len(valid),
        "infrastructure_errors": len(records) - len(valid),
        "total_successes": sum(1 for record in valid if record.success),
        "success_rate": sum(1 for record in valid if record.success) / max(len(valid), 1),
        "total_model_turns": valid_turns,
        "schema_valid_action_rate": schema_valid / max(valid_turns, 1),
        "premature_finish": sum(
            1 for record in valid if record.termination_reason == "premature_finish"
        ),
        "no_solution_trajectories": len(no_solution),
        "no_solution_successes": sum(1 for record in no_solution if record.success),
        "groups_with_reward_variance": sum(1 for group in groups if group["has_reward_variance"]),
        "groups_valid_for_grpo": sum(1 for group in groups if group["valid_for_grpo_update"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical M2.3-mini rollout probe")
    parser.add_argument("--policy", required=True, choices=["A", "B"])
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.K <= 0:
        raise ValueError("K must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive for stochastic rollout")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("max-tasks must be positive")

    task_dir = args.task_dir.expanduser().resolve()
    adapter_dir = args.adapter.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _load_split_tasks(task_dir, args.split)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    policy = "A_M2.2R" if args.policy == "A" else "B_M2.3-mini"
    task_hash = _task_source_hash(task_dir, args.split)
    adapter_hash = _sha256_directory(adapter_dir)
    prompt_hash = _sha256_file(Path(prompt_builder.__file__).resolve())
    heartbeat = Heartbeat(
        output_dir / f"heartbeat_{args.policy}_t{args.temperature}.json",
        policy,
        args.temperature,
        len(tasks),
    )

    print(
        f"Policy={policy} temperature={args.temperature} K={args.K} "
        f"tasks={len(tasks)} seed={args.seed}",
        flush=True,
    )
    backend, agent = load_policy(args.base_model, adapter_dir, args.temperature)
    records: list[RolloutRecord] = []
    groups: list[dict] = []
    started = time.time()

    try:
        for task_index, task in enumerate(tasks, start=1):
            task_id = task["task_id"]
            heartbeat.update(task=task_id, rollout=0)
            task_records: list[RolloutRecord] = []
            print(f"[{task_index}/{len(tasks)}] {task_id}", flush=True)
            for rollout_index in range(args.K):
                heartbeat.update(rollout=rollout_index + 1)
                record = run_rollout(
                    task,
                    rollout_index,
                    args.seed,
                    policy,
                    args.temperature,
                    task_dir,
                    agent,
                )
                task_records.append(record)
                records.append(record)
                print(
                    f"  k={rollout_index} valid={record.rollout_valid} "
                    f"success={record.success} term={record.termination_reason}",
                    flush=True,
                )

            summary = summarize_group(task_records, requested_k=args.K).to_dict()
            groups.append(summary)
            heartbeat.finish_task()

            incremental = {
                "schema_version": "3.0",
                "phase": "m2_3_mini_rollout_probe",
                "complete": False,
                "policy": policy,
                "temperature": args.temperature,
                "K": args.K,
                "seed": args.seed,
                "task_source_sha256": task_hash,
                "adapter_sha256": adapter_hash,
                "groups": groups,
                "records": [record.to_dict() for record in records],
            }
            _atomic_json_write(
                output_dir / f"incremental_{args.policy}_t{args.temperature}.json",
                incremental,
            )

        output = {
            "schema_version": "3.0",
            "phase": "m2_3_mini_rollout_probe",
            "complete": True,
            "policy": policy,
            "base_model": args.base_model,
            "adapter_path": str(adapter_dir),
            "adapter_sha256": adapter_hash,
            "prompt_contract": PROMPT_CONTRACT,
            "prompt_builder_sha256": prompt_hash,
            "task_dir": str(task_dir),
            "task_source_sha256": task_hash,
            "split": args.split,
            "temperature": args.temperature,
            "top_p": 0.9,
            "K": args.K,
            "seed": args.seed,
            "max_model_turns": MAX_MODEL_TURNS,
            "max_output_failures": MAX_OUTPUT_FAILURES,
            "metrics": _run_metrics(records, groups),
            "groups": groups,
            "records": [record.to_dict() for record in records],
            "elapsed_s": round(time.time() - started, 1),
        }
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = output_dir / f"single_probe_{args.policy}_t{args.temperature}_{timestamp}.json"
        _atomic_json_write(result_path, output)
        print(f"Results saved: {result_path}", flush=True)
        print(json.dumps(output["metrics"], indent=2, ensure_ascii=False), flush=True)
    finally:
        backend.unload()
        del agent, backend
        gc.collect()


if __name__ == "__main__":
    main()
