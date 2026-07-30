#!/usr/bin/env python3
"""M2.3-mini: Single-temperature rollout probe.

Designed to be run as one Slurm job per (policy, temperature) pair.
Each invocation loads the model once, runs one temperature, saves results.
"""
import argparse
import gc
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

# Derive project root from script location (single source of truth).
# Script lives at <repo>/scripts/m2_3_mini_single_probe.py
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]  # repo root
SRC_DIR = PROJECT_ROOT / "src"
if not SRC_DIR.is_dir():
    raise RuntimeError(f"Invalid source directory: {SRC_DIR}")
sys.path.insert(0, str(SRC_DIR))

try:
    from miniwebwork.agent_env.environment import ProcurementBrowserEnv
    from miniwebwork.agent_env.schemas import AgentAction
    from miniwebwork.tasks import get_public_task
    from miniwebwork.verifier import verify_episode
except ImportError as _e:
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
DEFAULT_TASK_DIR = PROJECT_ROOT / "data" / "tasks" / "rollout_dev_no_solution_v1"
DEFAULT_K = 8
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m2_3_mini"
DEFAULT_SEED = 20260731

HEARTBEAT_INTERVAL_S = 30  # write heartbeat file every 30s
MAX_OUTPUT_FAILURES = 3    # max consecutive schema-invalid turns before abort


class ProbeHeartbeat:
    """Write a heartbeat JSON file so external observers can tell if the
    probe is still alive and where it is in the task list."""

    def __init__(self, output_dir: Path, policy: str, temperature: float):
        self._path = output_dir / f"heartbeat_{policy}_t{temperature}.json"
        self._policy = policy
        self._temperature = temperature
        self._last_write = 0
        self._tasks_done = 0
        self._total_tasks = 0
        self._current_task = ""
        self._current_k_done = 0
        self._last_error = ""
        self._start_time = time.time()
        self.write(force=True)

    def set_total_tasks(self, n: int):
        self._total_tasks = n
        self.write(force=True)

    def start_task(self, task_id: str):
        self._current_task = task_id
        self._current_k_done = 0
        self.write(force=True)

    def advance_k(self, k_done: int):
        self._current_k_done = k_done
        self.write()

    def finish_task(self, task_id: str):
        self._tasks_done += 1
        self._current_task = ""
        self._current_k_done = 0
        self.write(force=True)

    def record_error(self, error: str):
        self._last_error = str(error)[:500]
        self.write(force=True)

    def write(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_write) < HEARTBEAT_INTERVAL_S:
            return
        self._last_write = now
        data = {
            "policy": self._policy,
            "temperature": self._temperature,
            "uptime_s": round(now - self._start_time, 1),
            "tasks_done": self._tasks_done,
            "total_tasks": self._total_tasks,
            "current_task": self._current_task,
            "current_k_done": self._current_k_done,
            "last_error": self._last_error,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            self._path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


def _log_rollout_detail(task_id, k, run_id, gen_error=None, parse_errors=None,
                        termination_reason=None, model_turns=0, success=False):
    """Log per-rollout diagnostic info with enough detail to debug hangs."""
    parts = [f"    [{task_id} k={k}] run_id={run_id}"]
    if gen_error:
        parts.append(f"gen_error={gen_error[:120]}")
    if parse_errors:
        parts.append(f"parse_errors={parse_errors[:120]}")
    if termination_reason:
        parts.append(f"term={termination_reason}")
    parts.append(f"turns={model_turns} success={success}")
    print(" | ".join(parts), flush=True)


def _bypass_caching_allocator_warmup():
    """Monkeypatch transformers' caching_allocator_warmup to no-op.

    The warmup function calls cudaMemGetInfo on all visible GPUs during
    model loading, which triggers IndexKernel on this system (likely a
    driver/PyTorch interaction bug when multiple GPUs are visible and
    some are near-full). Since we load on CPU and move to CUDA afterwards,
    the warmup is unnecessary.
    """
    import transformers.modeling_utils as mu
    mu.caching_allocator_warmup = lambda model, device_map, quantizer: None


def load_policy(base_model_path, adapter_path, temperature):
    """Load base model + adapter for a single temperature run.

    Fail-fast if adapter is missing to avoid silently producing base-model
    results that are mislabeled as a fine-tuned policy.
    """
    import torch
    from peft import PeftModel

    _bypass_caching_allocator_warmup()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading base model on CPU (bypass cudaMemGetInfo)...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    print(f"  Base model on {base_model.device}", flush=True)

    # --- Adapter: fail-fast if missing ---
    adapter_dir = Path(adapter_path).resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}\n"
            f"  Check --adapter path or run training first."
        )

    print(f"  Loading adapter from {adapter_dir} ...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_dir, torch_dtype=torch.bfloat16)
    model.eval()
    model.enable_adapter_layers()

    # Verify adapter is actually loaded
    active = getattr(model, "active_adapters", [])
    if not active:
        raise RuntimeError(
            f"Adapter loaded but no active adapters found. "
            f"Check adapter config at {adapter_dir}"
        )
    print(f"  Active adapters: {active}", flush=True)

    # Move to CUDA after all loading is complete
    print(f"  Moving model to CUDA...", flush=True)
    model = model.to("cuda:0")
    model.eval()
    torch.cuda.synchronize()
    print(f"  Model on {next(model.parameters()).device}, "
          f"mem={torch.cuda.memory_allocated()/1024**3:.1f}GB", flush=True)

    print(f"  Warm-up forward...", flush=True)
    with torch.inference_mode():
        _dummy = tokenizer("warmup", return_tensors="pt").to(model.device)
        _ = model(**_dummy)
    torch.cuda.synchronize()
    print(f"  Warm-up OK", flush=True)

    from miniwebwork.model_agent.model_backend import ModelConfig, QwenTransformersBackend
    from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
    from miniwebwork.model_agent.output_parser import parse
    import miniwebwork.model_agent.prompt_builder as pb

    # --- Freeze prompt contract ---
    assert pb.PROMPT_VERSION == "browser_agent_v2", (
        f"Prompt contract drift detected: {pb.PROMPT_VERSION}. "
        f"Expected 'browser_agent_v2'."
    )
    pb.HISTORY_WINDOW = 5

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

    agent = QwenBrowserAgent(backend, pb, parse)
    return backend, agent, tokenizer, adapter_dir


def run_single_rollout(task_id, backend, agent, tokenizer, K, task_dir,
                       base_seed, heartbeat=None):
    """Run K rollouts for a single task.

    Parameters
    ----------
    task_id : str
    backend, agent, tokenizer : loaded policy
    K : int  – number of rollouts per task
    task_dir : Path  – explicit task directory (passed to env, not env var)
    base_seed : int  – master seed; per-rollout seed = base_seed + idx
    heartbeat : ProbeHeartbeat or None

    Returns
    -------
    list[dict]  – trajectory records
    """
    trajectories = []
    for k in range(K):
        rollout_seed = base_seed + k  # deterministic per (task, k)
        run_id = f"probe_{uuid.uuid4().hex[:8]}"

        try:
            # Pass task_dir explicitly — no global env var override
            with ProcurementBrowserEnv(
                max_steps=25, run_id=run_id, headless=True, task_dir=task_dir
            ) as env:
                env.set_agent_name("rollout_probe")
                obs = env.reset(task_id)
                agent.reset(obs.task_id, obs.instruction)

                # Set per-rollout seed for reproducibility
                random.seed(rollout_seed)
                import numpy as np
                np.random.seed(rollout_seed)
                import torch
                torch.manual_seed(rollout_seed)
                torch.cuda.manual_seed_all(rollout_seed)

                model_turns = 0
                env_steps = 0
                schema_valid_count = 0
                schema_invalid_count = 0
                output_failure_streak = 0
                gen_errors = []
                parse_errors_acc = []
                step_events = []

                termination_reason = "max_steps"
                success = False
                rollout_valid = True
                failure_origin = "policy"

                for step in range(25):
                    # --- Model turn ---
                    try:
                        attempt = agent.act(obs)
                    except Exception as act_exc:
                        _log_rollout_detail(task_id, k, run_id,
                                            gen_error=str(act_exc),
                                            model_turns=model_turns)
                        if heartbeat:
                            heartbeat.record_error(f"agent.act exception: {act_exc}")
                        termination_reason = "agent_exception"
                        rollout_valid = False
                        failure_origin = "infrastructure"
                        break

                    raw_output = attempt.raw_output or ""
                    is_schema_valid = attempt.schema_valid
                    turn_action_dict = None

                    # --- P0-1: Schema-invalid → skip, don't fabricate finish ---
                    if not is_schema_valid:
                        schema_invalid_count += 1
                        output_failure_streak += 1
                        if attempt.errors:
                            parse_errors_acc.extend(attempt.errors)
                        if not raw_output.strip():
                            gen_errors.append("empty_generation")
                        for err in attempt.errors:
                            if "generation_error" in err:
                                gen_errors.append(err)

                        if output_failure_streak >= MAX_OUTPUT_FAILURES:
                            _log_rollout_detail(task_id, k, run_id,
                                                parse_errors="; ".join(parse_errors_acc),
                                                termination_reason="model_output_failure_limit",
                                                model_turns=model_turns)
                            if heartbeat:
                                heartbeat.record_error("model_output_failure_limit")
                            termination_reason = "model_output_failure_limit"
                            break

                        # Skip this turn: no env.step, no history record
                        # But still count as a model turn for statistics
                        model_turns += 1
                        step_events.append({
                            "turn": model_turns,
                            "page_type": obs.page_type,
                            "raw_model_output": raw_output[:500],
                            "schema_valid": False,
                            "schema_errors": attempt.errors,
                            "parsed_action": None,
                            "action_was_fallback": False,
                            "skipped": True,
                        })
                        continue

                    # Schema valid
                    schema_valid_count += 1
                    output_failure_streak = 0
                    action = attempt.action
                    turn_action_dict = action.to_dict() if action else None

                    # Record generation errors even on valid turns (shouldn't happen)
                    for err in attempt.errors:
                        if "generation_error" in err:
                            gen_errors.append(err)
                    if gen_errors:
                        _log_rollout_detail(task_id, k, run_id,
                                            gen_error="; ".join(gen_errors[:3]),
                                            parse_errors="; ".join(parse_errors_acc),
                                            model_turns=model_turns)
                        if heartbeat:
                            heartbeat.record_error(f"gen_errors: {'; '.join(gen_errors[:3])}")
                        termination_reason = "generation_error"
                        break

                    # --- Environment step ---
                    try:
                        result = env.step(action)
                        env_steps += 1
                        terminated = result.terminated
                        truncated = result.truncated
                        reward = result.reward

                        step_events.append({
                            "turn": model_turns + 1,
                            "page_type": obs.page_type,
                            "raw_model_output": raw_output[:500],
                            "schema_valid": True,
                            "schema_errors": [],
                            "parsed_action": turn_action_dict,
                            "action_was_fallback": getattr(attempt, 'fallback_used', False),
                            "env_action_success": result.info.get("action_result", {}).get("success"),
                            "terminated": terminated,
                            "truncated": truncated,
                            "skipped": False,
                        })

                        if result.observation:
                            agent.record_feedback(attempt, result, result.observation.page_type)
                            obs = result.observation
                        else:
                            agent.record_feedback(attempt, result, "unknown")

                        model_turns += 1

                        if terminated or truncated:
                            termination_reason = result.info.get("termination_reason", "terminal")
                            success = result.reward > 0.5
                            break

                    except Exception as _e:
                        # --- P0-3: Infrastructure error, not policy failure ---
                        print(f"    [WARN] env.step() error at turn {model_turns}: {_e}", flush=True)
                        if heartbeat:
                            heartbeat.record_error(f"env.step: {_e}")
                        model_turns += 1
                        termination_reason = "environment_step_error"
                        rollout_valid = False
                        failure_origin = "infrastructure"
                        step_events.append({
                            "turn": model_turns,
                            "page_type": obs.page_type,
                            "raw_model_output": raw_output[:500],
                            "schema_valid": True,
                            "schema_errors": [],
                            "parsed_action": turn_action_dict,
                            "env_error": str(_e)[:300],
                            "skipped": False,
                        })
                        break
                # --- End of step loop ---

                # Read trajectory (after `with` block exits, env is finalized)
                traj = env.trajectory
                if traj is not None:
                    verification = traj.verification
                    if rollout_valid:
                        success = verification.get("success", success)
                    termination_reason = traj.termination_reason or termination_reason
                else:
                    verification = {}
                    if rollout_valid:
                        rollout_valid = False
                        failure_origin = "infrastructure"
                        termination_reason = "trajectory_not_created"

                _log_rollout_detail(task_id, k, run_id,
                                    termination_reason=termination_reason,
                                    model_turns=model_turns, success=success)

                trajectories.append({
                    "task_id": task_id,
                    "episode_id": run_id,
                    "k": k,
                    "rollout_seed": rollout_seed,
                    "success": success,
                    "reward": 1.0 if success else 0.0,
                    "termination_reason": termination_reason,
                    "model_turns": model_turns,
                    "environment_steps": env_steps,
                    "schema_valid_count": schema_valid_count,
                    "schema_invalid_count": schema_invalid_count,
                    "verification": verification,
                    "gen_errors": gen_errors,
                    "parse_errors": parse_errors_acc,
                    "step_events": step_events,
                    # P0-3: Infrastructure vs policy classification
                    "rollout_valid": rollout_valid,
                    "failure_origin": failure_origin,
                })

            if heartbeat:
                heartbeat.advance_k(k + 1)

        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            if heartbeat:
                heartbeat.record_error(str(e))
            trajectories.append({
                "task_id": task_id,
                "episode_id": f"error_{k}",
                "k": k,
                "rollout_seed": base_seed + k,
                "success": False,
                "reward": None,  # P0-3: None, not 0.0
                "termination_reason": f"error: {str(e)[:200]}",
                "model_turns": 0,
                "environment_steps": 0,
                "schema_valid_count": 0,
                "schema_invalid_count": 0,
                "verification": {},
                "gen_errors": [str(e)[:200]],
                "parse_errors": [],
                "step_events": [],
                "rollout_valid": False,
                "failure_origin": "infrastructure",
            })

    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Single-temp rollout probe")
    parser.add_argument("--policy", type=str, required=True, choices=["A", "B"])
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Master seed for per-rollout deterministic sampling")
    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        print(f"ERROR: {_IMPORT_ERROR}")
        sys.exit(1)

    if args.K <= 0:
        raise ValueError(f"K must be positive, got {args.K}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set task dir globally so imports that read it at module level work
    os.environ["MINIWEBWORK_TASK_DIR"] = str(args.task_dir)

    policy_label = "A_M2.2R" if args.policy == "A" else "B_M2.3-mini"
    print(f"\nPOLICY: {policy_label} | Temp: {args.temperature} | Adapter: {args.adapter}", flush=True)
    print(f"Seed: {args.seed} | Task dir: {args.task_dir}", flush=True)

    # Load tasks (used for task list; actual task data loaded by env via task_dir)
    tasks = []
    public_path = args.task_dir / f"{args.split}_public.jsonl"
    if not public_path.exists():
        raise FileNotFoundError(f"Task file not found: {public_path}")
    for line in public_path.read_text().strip().split("\n"):
        if line.strip():
            tasks.append(json.loads(line))

    if not tasks:
        raise ValueError(f"No tasks found in {public_path}")

    # Build task_id → task_type lookup (P1-2)
    task_type_map = {}
    for t in tasks:
        task_type_map[t["task_id"]] = t.get("task_type", "")

    print(f"  Loaded {len(tasks)} tasks from {args.split} split", flush=True)

    # Load policy (P1-1: fail-fast on missing adapter, P1-4: freeze contract)
    backend, agent, tokenizer, adapter_dir = load_policy(
        args.base_model, args.adapter, args.temperature,
    )

    import torch
    heartbeat = ProbeHeartbeat(args.output_dir, args.policy, args.temperature)
    heartbeat.set_total_tasks(len(tasks))

    all_trajs = []
    task_groups = []

    job_start = time.time()
    for task_idx, task in enumerate(tasks):
        tid = task["task_id"]
        print(f"  [{task_idx+1}/{len(tasks)}] Task: {tid} (K={args.K})", flush=True)
        t0 = time.time()

        heartbeat.start_task(tid)

        trajs = run_single_rollout(
            tid, backend, agent, tokenizer, args.K,
            task_dir=args.task_dir,
            base_seed=args.seed,
            heartbeat=heartbeat,
        )
        all_trajs.extend(trajs)

        # Log CUDA memory after each task block
        try:
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"    [diag] CUDA mem after {tid}: {mem_gb:.2f} GB", flush=True)
        except Exception:
            pass

        # --- Group-level metrics (only valid policy rollouts) ---
        valid_rewards = [
            t["reward"] for t in trajs
            if t.get("rollout_valid") and t.get("failure_origin") == "policy"
        ]
        successes = int(sum(1 for r in valid_rewards if r == 1.0))
        n_valid = len(valid_rewards)
        mean_r = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
        std_r = (
            (sum((r - mean_r) ** 2 for r in valid_rewards) / len(valid_rewards)) ** 0.5
            if n_valid > 0 else 0.0
        )

        gen_err_count = sum(1 for t in trajs if t.get("gen_errors"))
        parse_err_count = sum(1 for t in trajs if t.get("parse_errors"))
        infra_count = sum(1 for t in trajs if t.get("failure_origin") == "infrastructure")

        group = {
            "task_id": tid,
            "task_type": task_type_map.get(tid, ""),
            "policy": policy_label,
            "temperature": args.temperature,
            "K": args.K,
            "rollout_seed_base": args.seed,
            "num_trajectories": len(trajs),
            "num_valid": n_valid,
            "num_infrastructure_errors": infra_count,
            "group_reward_mean": mean_r,
            "group_reward_std": std_r,
            "success_count": successes,
            "has_variance": std_r > 0,
            "valid_for_update": std_r > 0 and successes > 0 and n_valid > 0,
            "reward_sequence": valid_rewards,
            "schema_valid_count": sum(t.get("schema_valid_count", 0) for t in trajs),
            "schema_invalid_count": sum(t.get("schema_invalid_count", 0) for t in trajs),
            "gen_error_count": gen_err_count,
            "parse_error_count": parse_err_count,
            "elapsed_s": time.time() - t0,
            "trajectories": trajs,
        }
        task_groups.append(group)

        print(f"    Result: success={successes}/{n_valid} (valid), "
              f"mean={mean_r:.2f}, std={std_r:.3f}, "
              f"infra_errs={infra_count}, gen_errors={gen_err_count}, "
              f"parse_errors={parse_err_count}, elapsed={time.time()-t0:.1f}s",
              flush=True)

        heartbeat.finish_task(tid)

        # --- Incremental save after each task (survives mid-job failure) ---
        _save_incremental(args.output_dir, policy_label, args, task_groups, all_trajs,
                          adapter_dir, job_start)

    # --- Final metrics (P0-4: action-level schema_valid_rate) ---
    total = len(all_trajs)
    # Only count valid policy rollouts for formal metrics
    valid_trajs = [
        t for t in all_trajs
        if t.get("rollout_valid") and t.get("failure_origin") == "policy"
    ]
    n_valid = len(valid_trajs)
    total_successes = int(sum(1 for t in valid_trajs if t["success"]))
    total_model_turns = sum(t.get("model_turns", 0) for t in valid_trajs)
    total_schema_valid_actions = sum(t.get("schema_valid_count", 0) for t in valid_trajs)
    total_schema_invalid_actions = sum(t.get("schema_invalid_count", 0) for t in valid_trajs)

    # Action-level: what fraction of all model actions had valid schema?
    schema_valid_action_rate = (
        total_schema_valid_actions / max(total_model_turns, 1)
    )

    # Trajectory-level: how many trajectories were entirely schema-valid?
    traj_all_valid = sum(
        1 for t in valid_trajs
        if t.get("model_turns", 0) > 0
        and t.get("schema_valid_count", 0) == t.get("model_turns", 0)
    )
    traj_any_valid = sum(
        1 for t in valid_trajs if t.get("schema_valid_count", 0) > 0
    )

    premature_finish = sum(
        1 for t in valid_trajs
        if t.get("termination_reason") == "premature_finish"
    )
    infra_errors = sum(
        1 for t in all_trajs
        if t.get("failure_origin") == "infrastructure"
    )

    # P1-2: no-solution by unique task_id, using task_type metadata
    no_sol_task_ids = sorted({
        tid for tid, ttype in task_type_map.items()
        if ttype == "no_feasible_product"
    })
    no_sol_trajs = [t for t in valid_trajs if t["task_id"] in no_sol_task_ids]
    no_sol_success = int(sum(1 for t in no_sol_trajs if t["success"]))

    metrics = {
        "total_trajectories": total,
        "valid_trajectories": n_valid,
        "infrastructure_errors": infra_errors,
        "total_successes": total_successes,
        "success_rate": total_successes / max(n_valid, 1),
        # P0-4: action-level schema valid rate
        "schema_valid_action_rate": schema_valid_action_rate,
        "total_model_turns": total_model_turns,
        "total_schema_valid_actions": total_schema_valid_actions,
        "total_schema_invalid_actions": total_schema_invalid_actions,
        "trajectory_all_schema_valid_rate": traj_all_valid / max(n_valid, 1),
        "trajectory_any_schema_valid_rate": traj_any_valid / max(n_valid, 1),
        "premature_finish": premature_finish,
        "env_error_rate": infra_errors / max(total, 1),
        "no_solution_successes": no_sol_success,
        "no_solution_tasks": len(no_sol_task_ids),
        "no_solution_trajectories": len(no_sol_trajs),
        "total_gen_errors": sum(1 for t in all_trajs if t.get("gen_errors")),
        "total_parse_errors": sum(1 for t in all_trajs if t.get("parse_errors")),
        "job_elapsed_s": round(time.time() - job_start, 1),
    }

    output = {
        "schema_version": "2.0",
        "phase": "m2_3_mini_single_probe",
        "policy": policy_label,
        "adapter_path": str(adapter_dir),
        "adapter_hash": _dir_hash(adapter_dir),
        "base_model": args.base_model,
        "temperature": args.temperature,
        "K": args.K,
        "seed": args.seed,
        "num_tasks": len(tasks),
        "prompt_contract": "browser_agent_v2",
        "prompt_builder_hash": _prompt_builder_hash(),
        "max_output_failures": MAX_OUTPUT_FAILURES,
        "metrics": metrics,
        "groups": task_groups,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"single_probe_{args.policy}_t{args.temperature}_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved: {out_path}", flush=True)

    print(f"\nSummary: success={total_successes}/{n_valid} (valid), "
          f"schema_valid_action={schema_valid_action_rate:.1%} "
          f"({total_schema_valid_actions}/{total_model_turns} turns), "
          f"premature_finish={premature_finish}, infra_errors={infra_errors}, "
          f"gen_errors={metrics['total_gen_errors']}, "
          f"parse_errors={metrics['total_parse_errors']}, "
          f"job_time={metrics['job_elapsed_s']}s", flush=True)

    # --- Cleanup ---
    del backend, agent, tokenizer
    import torch
    try:
        torch.cuda.empty_cache()
    except Exception as _e:
        print(f"[WARN] empty_cache failed: {_e}", flush=True)
    gc.collect()


def _save_incremental(output_dir, policy_label, args, task_groups, all_trajs,
                      adapter_dir, job_start):
    """Save intermediate results after each task block."""
    try:
        valid_trajs = [
            t for t in all_trajs
            if t.get("rollout_valid") and t.get("failure_origin") == "policy"
        ]
        n_valid = len(valid_trajs)
        total_successes = int(sum(1 for t in valid_trajs if t["success"]))
        incremental = {
            "schema_version": "2.0",
            "phase": "m2_3_mini_single_probe_incremental",
            "policy": policy_label,
            "adapter_path": str(adapter_dir),
            "temperature": args.temperature,
            "K": args.K,
            "seed": args.seed,
            "tasks_completed": len(task_groups),
            "total_trajectories": len(all_trajs),
            "valid_trajectories": n_valid,
            "current_successes": total_successes,
            "job_elapsed_s": round(time.time() - job_start, 1),
            "groups": task_groups,
        }
        path = output_dir / f"incremental_{args.policy}_t{args.temperature}.json"
        path.write_text(json.dumps(incremental, indent=2, ensure_ascii=False))
    except Exception:
        pass  # non-critical


def _dir_hash(path: Path) -> str:
    """SHA256 of adapter directory contents (for reproducibility tracking)."""
    import hashlib
    h = hashlib.sha256()
    try:
        for f in sorted(path.rglob("*")):
            if f.is_file():
                h.update(f.name.encode())
                h.update(str(f.stat().st_size).encode())
    except Exception:
        pass
    return h.hexdigest()[:16]


def _prompt_builder_hash() -> str:
    """Hash of the prompt builder module for contract verification."""
    import hashlib
    import miniwebwork.model_agent.prompt_builder as pb
    src = pb.PROMPT_VERSION + getattr(pb, "__file__", "")
    return hashlib.sha256(src.encode()).hexdigest()[:16]
