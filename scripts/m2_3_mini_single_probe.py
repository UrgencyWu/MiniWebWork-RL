#!/usr/bin/env python3
"""M2.3-mini single-temperature rollout probe.

One invocation evaluates one policy/temperature pair.  The script is strict
about experimental contracts: infrastructure failures never become policy
rewards, prompt/task/adaptor identities are recorded, and every model turn is
retained for diagnosis.
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
import miniwebwork.model_agent.prompt_builder as prompt_builder

DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
DEFAULT_TASK_DIR = PROJECT_ROOT / "data" / "tasks" / "rollout_dev_no_solution_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m2_3_mini"
DEFAULT_K = 8
DEFAULT_SEED = 20260731
MAX_MODEL_TURNS = 25
MAX_OUTPUT_FAILURES = 3


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(file_path.relative_to(path)).encode("utf-8"))
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def stable_rollout_seed(master_seed: int, task_id: str, rollout_index: int) -> int:
    digest = hashlib.sha256(f"{master_seed}:{task_id}:{rollout_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class ProbeHeartbeat:
    def __init__(self, output_dir: Path, policy: str, temperature: float):
        self.path = output_dir / f"heartbeat_{policy}_t{temperature}.json"
        self.policy = policy
        self.temperature = temperature
        self.started_at = time.time()
        self.tasks_done = 0
        self.total_tasks = 0
        self.current_task = ""
        self.current_k = 0
        self.last_error = ""
        self.write()

    def write(self) -> None:
        atomic_write_json(self.path, {
            "policy": self.policy,
            "temperature": self.temperature,
            "uptime_s": round(time.time() - self.started_at, 1),
            "tasks_done": self.tasks_done,
            "total_tasks": self.total_tasks,
            "current_task": self.current_task,
            "current_k": self.current_k,
            "last_error": self.last_error,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })


def load_policy(base_model_path: str, adapter_path: str, temperature: float):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompt_builder.PROMPT_VERSION != "browser_agent_v2":
        raise RuntimeError(
            f"Prompt contract drift: {prompt_builder.PROMPT_VERSION!r}; "
            "expected 'browser_agent_v2'"
        )
    prompt_builder.HISTORY_WINDOW = 5

    adapter_dir = Path(adapter_path).resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model, adapter_dir, torch_dtype=torch.bfloat16,
    )
    model.enable_adapter_layers()
    model.eval()
    if not getattr(model, "active_adapters", []):
        raise RuntimeError("Adapter loaded but no active adapter is registered")

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
    )
    backend = QwenTransformersBackend(config)
    backend._model = model
    backend._tokenizer = tokenizer
    backend._loaded = True
    agent = QwenBrowserAgent(backend, prompt_builder, parse)
    return backend, agent, tokenizer, adapter_dir


def run_rollout(
    task: dict[str, Any],
    rollout_index: int,
    agent: QwenBrowserAgent,
    task_dir: Path,
    master_seed: int,
) -> dict[str, Any]:
    task_id = task["task_id"]
    rollout_seed = stable_rollout_seed(master_seed, task_id, rollout_index)
    seed_everything(rollout_seed)
    run_id = f"probe_{uuid.uuid4().hex[:10]}"

    model_turns = 0
    environment_steps = 0
    schema_valid_count = 0
    schema_invalid_count = 0
    output_failure_streak = 0
    step_events: list[dict[str, Any]] = []
    termination_reason = "max_model_turns"
    rollout_valid = True
    failure_origin = "policy"
    success = False
    verification: dict[str, Any] = {}

    try:
        with ProcurementBrowserEnv(
            max_steps=MAX_MODEL_TURNS,
            run_id=run_id,
            headless=True,
            task_dir=task_dir,
        ) as env:
            env.set_agent_name("m2_3_mini_rollout_probe")
            obs = env.reset(task_id)
            agent.reset(obs.task_id, obs.instruction)

            for _ in range(MAX_MODEL_TURNS):
                page_type_before = obs.page_type
                attempt = agent.act(obs)
                model_turns += 1

                event = {
                    "turn": model_turns,
                    "page_type": page_type_before,
                    "raw_model_output": (attempt.raw_output or "")[:2000],
                    "strict_json_success": attempt.strict_json_success,
                    "fallback_used": attempt.fallback_used,
                    "schema_valid": attempt.schema_valid,
                    "schema_errors": list(attempt.errors),
                    "prompt_hash": attempt.prompt_hash,
                    "input_tokens": attempt.input_tokens,
                    "output_tokens": attempt.output_tokens,
                    "parsed_action": attempt.action.to_dict() if attempt.action else None,
                    "generation_logprobs": getattr(agent._backend.generate, "logprobs", None),
                }

                if not attempt.schema_valid or attempt.action is None:
                    schema_invalid_count += 1
                    output_failure_streak += 1
                    event["skipped_env_step"] = True
                    step_events.append(event)
                    if output_failure_streak >= MAX_OUTPUT_FAILURES:
                        termination_reason = "model_output_failure_limit"
                        break
                    continue

                schema_valid_count += 1
                output_failure_streak = 0
                result = env.step(attempt.action)
                environment_steps += 1
                event.update({
                    "skipped_env_step": False,
                    "env_action_success": result.info.get("action_result", {}).get("success"),
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "termination_reason": result.info.get("termination_reason", ""),
                })
                step_events.append(event)

                if result.observation is not None:
                    agent.record_feedback(attempt, result, result.observation.page_type)
                    obs = result.observation
                else:
                    agent.record_feedback(attempt, result, "unknown")

                if result.terminated or result.truncated:
                    termination_reason = result.info.get("termination_reason", "terminal")
                    success = bool(result.reward > 0.5)
                    break

            trajectory = env.trajectory
            if trajectory is None:
                raise RuntimeError("Environment did not create a trajectory")
            verification = trajectory.verification or {}
            success = bool(verification.get("success", success))
            termination_reason = trajectory.termination_reason or termination_reason

    except Exception as exc:
        rollout_valid = False
        failure_origin = "infrastructure"
        termination_reason = "infrastructure_error"
        verification = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:1000],
        }

    if schema_valid_count + schema_invalid_count != model_turns:
        rollout_valid = False
        failure_origin = "infrastructure"
        termination_reason = "metrics_invariant_failure"

    reward = (1.0 if success else 0.0) if rollout_valid else None
    return {
        "task_id": task_id,
        "task_type": task.get("task_type", ""),
        "episode_id": run_id,
        "k": rollout_index,
        "rollout_seed": rollout_seed,
        "rollout_valid": rollout_valid,
        "failure_origin": failure_origin,
        "success": success,
        "reward": reward,
        "termination_reason": termination_reason,
        "model_turns": model_turns,
        "environment_steps": environment_steps,
        "schema_valid_count": schema_valid_count,
        "schema_invalid_count": schema_invalid_count,
        "verification": verification,
        "step_events": step_events,
    }


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    all_trajs = [traj for group in groups for traj in group["trajectories"]]
    valid = [t for t in all_trajs if t["rollout_valid"]]
    total_turns = sum(t["model_turns"] for t in valid)
    valid_actions = sum(t["schema_valid_count"] for t in valid)
    invalid_actions = sum(t["schema_invalid_count"] for t in valid)
    no_solution_ids = {g["task_id"] for g in groups if g["task_type"] == "no_feasible_product"}
    no_solution_trajs = [t for t in valid if t["task_id"] in no_solution_ids]

    return {
        "total_trajectories": len(all_trajs),
        "valid_trajectories": len(valid),
        "infrastructure_errors": len(all_trajs) - len(valid),
        "total_successes": sum(1 for t in valid if t["success"]),
        "success_rate": sum(1 for t in valid if t["success"]) / max(len(valid), 1),
        "total_model_turns": total_turns,
        "total_schema_valid_actions": valid_actions,
        "total_schema_invalid_actions": invalid_actions,
        "schema_valid_action_rate": valid_actions / max(total_turns, 1),
        "premature_finish": sum(1 for t in valid if t["termination_reason"] == "premature_finish"),
        "no_solution_tasks": len(no_solution_ids),
        "no_solution_trajectories": len(no_solution_trajs),
        "no_solution_successes": sum(1 for t in no_solution_trajs if t["success"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M2.3-mini single-temperature rollout probe")
    parser.add_argument("--policy", required=True, choices=["A", "B"])
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.K <= 0:
        raise ValueError("K must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive for stochastic rollout")

    args.task_dir = args.task_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MINIWEBWORK_TASK_DIR"] = str(args.task_dir)

    public_path = args.task_dir / f"{args.split}_public.jsonl"
    if not public_path.is_file():
        raise FileNotFoundError(public_path)
    tasks = [json.loads(line) for line in public_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tasks:
        raise ValueError(f"No tasks found in {public_path}")

    policy_label = "A_M2.2R" if args.policy == "A" else "B_M2.3-mini"
    backend, agent, tokenizer, adapter_dir = load_policy(
        args.base_model, args.adapter, args.temperature,
    )

    heartbeat = ProbeHeartbeat(args.output_dir, args.policy, args.temperature)
    heartbeat.total_tasks = len(tasks)
    heartbeat.write()

    groups: list[dict[str, Any]] = []
    started_at = time.time()
    incremental_path = args.output_dir / f"incremental_{args.policy}_t{args.temperature}.json"

    for task_index, task in enumerate(tasks):
        heartbeat.current_task = task["task_id"]
        heartbeat.current_k = 0
        heartbeat.write()
        task_started = time.time()
        trajectories = []
        for k in range(args.K):
            heartbeat.current_k = k
            heartbeat.write()
            trajectories.append(run_rollout(task, k, agent, args.task_dir, args.seed))

        valid_rewards = [t["reward"] for t in trajectories if t["rollout_valid"]]
        mean_reward = sum(valid_rewards) / max(len(valid_rewards), 1)
        variance = (
            sum((r - mean_reward) ** 2 for r in valid_rewards) / len(valid_rewards)
            if valid_rewards else 0.0
        )
        group = {
            "task_id": task["task_id"],
            "task_type": task.get("task_type", ""),
            "policy": policy_label,
            "temperature": args.temperature,
            "K": args.K,
            "num_valid": len(valid_rewards),
            "num_infrastructure_errors": len(trajectories) - len(valid_rewards),
            "success_count": sum(1 for r in valid_rewards if r == 1.0),
            "group_reward_mean": mean_reward,
            "group_reward_std": variance ** 0.5,
            "has_variance": variance > 0.0,
            "valid_for_update": variance > 0.0 and 0 < sum(valid_rewards) < len(valid_rewards),
            "reward_sequence": valid_rewards,
            "elapsed_s": round(time.time() - task_started, 2),
            "trajectories": trajectories,
        }
        groups.append(group)
        heartbeat.tasks_done = task_index + 1
        heartbeat.current_task = ""
        heartbeat.current_k = 0
        heartbeat.write()

        atomic_write_json(incremental_path, {
            "schema_version": "3.0",
            "complete": False,
            "policy": policy_label,
            "temperature": args.temperature,
            "K": args.K,
            "seed": args.seed,
            "groups": groups,
            "metrics": summarize(groups),
        })

    output = {
        "schema_version": "3.0",
        "complete": True,
        "phase": "m2_3_mini_single_probe",
        "policy": policy_label,
        "base_model": args.base_model,
        "adapter_path": str(adapter_dir),
        "adapter_sha256": directory_sha256(adapter_dir),
        "task_file": str(public_path),
        "task_file_sha256": file_sha256(public_path),
        "prompt_contract": prompt_builder.PROMPT_VERSION,
        "prompt_builder_sha256": file_sha256(Path(prompt_builder.__file__)),
        "temperature": args.temperature,
        "top_p": 0.9,
        "K": args.K,
        "seed": args.seed,
        "num_tasks": len(tasks),
        "elapsed_s": round(time.time() - started_at, 2),
        "metrics": summarize(groups),
        "groups": groups,
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"single_probe_{args.policy}_t{args.temperature}_{timestamp}.json"
    atomic_write_json(out_path, output)
    print(json.dumps(output["metrics"], indent=2, ensure_ascii=False), flush=True)
    print(f"Results saved to {out_path}", flush=True)

    del backend, agent, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
