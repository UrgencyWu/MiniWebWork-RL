#!/usr/bin/env python3
"""M2.3-mini: Single-temperature rollout probe.

Designed to be run as one Slurm job per (policy, temperature) pair.
Each invocation loads the model once, runs one temperature, saves results.
"""
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

try:
    from miniwebwork.agent_env.environment import ProcurementBrowserEnv
    from miniwebwork.agent_env.schemas import AgentAction
    from miniwebwork.tasks import get_public_task
    from miniwebwork.verifier import verify_episode
except ImportError as _e:
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")
DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
DEFAULT_TASK_DIR = PROJECT_ROOT / "data" / "tasks" / "rollout_dev_no_solution_v1"
DEFAULT_K = 8
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m2_3_mini"


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
    """Load base model + adapter for a single temperature run."""
    import torch
    _bypass_caching_allocator_warmup()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from miniwebwork.model_agent.model_backend import ModelConfig, QwenTransformersBackend
    from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
    from miniwebwork.model_agent.output_parser import parse
    import miniwebwork.model_agent.prompt_builder as pb

    print(f"  Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading base model on CPU (bypass cudaMemGetInfo)...", flush=True)
    # Load on CPU first to avoid caching_allocator_warmup ->
    # cudaMemGetInfo -> IndexKernel on this GPU.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    print(f"  Base model on {base_model.device}", flush=True)

    if adapter_path and Path(adapter_path).exists():
        print(f"  Loading adapter on CPU...", flush=True)
        model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.bfloat16)
    else:
        print(f"  WARNING: adapter not found", flush=True)
        model = base_model

    model.eval()
    model.enable_adapter_layers()
    print(f"  Active adapters: {model.active_adapters}", flush=True)

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

    pb.HISTORY_WINDOW = 5
    agent = QwenBrowserAgent(backend, pb, parse)
    return backend, agent, tokenizer


def run_single_rollout(task_id, backend, agent, tokenizer, K):
    """Run K rollouts for a single task."""
    trajectories = []
    for k in range(K):
        run_id = f"probe_{uuid.uuid4().hex[:8]}"
        os.environ["MINIWEBWORK_TASK_DIR"] = str(DEFAULT_TASK_DIR)

        try:
            with ProcurementBrowserEnv(max_steps=25, run_id=run_id, headless=True) as env:
                env.set_agent_name("rollout_probe")
                obs = env.reset(task_id)
                agent.reset(obs.task_id, obs.instruction)
                model_turns = 0
                env_steps = 0
                schema_valid_count = 0

                for step in range(25):
                    attempt = agent.act(obs)
                    schema_valid = attempt.schema_valid

                    if schema_valid:
                        schema_valid_count += 1
                        action = attempt.action
                    else:
                        action = AgentAction(action="finish")

                    try:
                        result = env.step(action)
                        env_steps += 1
                        terminated = result.terminated
                        truncated = result.truncated
                        reward = result.reward

                        if result.observation:
                            agent.record_feedback(attempt, result, result.observation.page_type)
                            obs = result.observation
                        else:
                            agent.record_feedback(attempt, result, "unknown")

                        model_turns += 1
                        if terminated or truncated:
                            break
                    except Exception as _e:
                        print(f"    [WARN] env.step() error at turn {model_turns}: {_e}", flush=True)
                        break

                traj = env.trajectory
                if traj is not None:
                    verification = traj.verification
                    success = verification.get("success", False)
                    termination_reason = traj.termination_reason
                else:
                    verification = {}
                    success = False
                    termination_reason = "trajectory_not_created"

                trajectories.append({
                    "task_id": task_id,
                    "episode_id": run_id,
                    "k": k,
                    "success": success,
                    "reward": 1.0 if success else 0.0,
                    "termination_reason": termination_reason,
                    "model_turns": model_turns,
                    "environment_steps": env_steps,
                    "schema_valid_count": schema_valid_count,
                    "verification": verification,
                })

        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            trajectories.append({
                "task_id": task_id,
                "episode_id": f"error_{k}",
                "k": k,
                "success": False,
                "reward": 0.0,
                "termination_reason": f"error: {str(e)[:200]}",
                "model_turns": 0,
                "environment_steps": 0,
                "schema_valid_count": 0,
                "verification": {},
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
    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        print(f"ERROR: {_IMPORT_ERROR}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MINIWEBWORK_TASK_DIR"] = str(args.task_dir)

    policy_label = "A_M2.2R" if args.policy == "A" else "B_M2.3-mini"
    print(f"\nPOLICY: {policy_label} | Temp: {args.temperature} | Adapter: {args.adapter}", flush=True)

    tasks = []
    public_path = args.task_dir / f"{args.split}_public.jsonl"
    for line in public_path.read_text().strip().split("\n"):
        if line.strip():
            tasks.append(json.loads(line))

    backend, agent, tokenizer = load_policy(args.base_model, args.adapter, args.temperature)

    all_trajs = []
    task_groups = []

    for task in tasks:
        tid = task["task_id"]
        print(f"  Task: {tid} (K={args.K})", flush=True)
        t0 = time.time()

        trajs = run_single_rollout(tid, backend, agent, tokenizer, args.K)
        all_trajs.extend(trajs)

        rewards = [t["reward"] for t in trajs]
        successes = sum(rewards)
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5

        group = {
            "task_id": tid,
            "policy": policy_label,
            "temperature": args.temperature,
            "K": args.K,
            "num_trajectories": len(trajs),
            "group_reward_mean": mean_r,
            "group_reward_std": std_r,
            "success_count": successes,
            "has_variance": std_r > 0,
            "valid_for_update": std_r > 0 and successes > 0,
            "reward_sequence": rewards,
            "schema_valid_count": sum(1 for t in trajs if t.get("schema_valid_count", 0) > 0),
            "elapsed_s": time.time() - t0,
            "trajectories": trajs,
        }
        task_groups.append(group)

        print(f"    Success: {successes}/{args.K}, mean={mean_r:.2f}, std={std_r:.3f}", flush=True)

    # Compute metrics
    total = len(all_trajs)
    successes = sum(1 for t in all_trajs if t["success"])
    schema_valid = sum(1 for t in all_trajs if t.get("schema_valid_count", 0) > 0)
    premature_finish = sum(1 for t in all_trajs if t.get("termination_reason") == "premature_finish")
    env_errors = sum(1 for t in all_trajs if "error:" in str(t.get("termination_reason", "")))
    no_solution = [t for t in all_trajs if t.get("task_id", "").startswith("M2_3_V")]
    no_sol_success = sum(1 for t in no_solution if t["success"])

    metrics = {
        "total_trajectories": total,
        "total_successes": successes,
        "success_rate": successes / max(total, 1),
        "schema_valid_rate": schema_valid / max(total, 1),
        "premature_finish": premature_finish,
        "env_error_rate": env_errors / max(total, 1),
        "no_solution_successes": no_sol_success,
        "no_solution_tasks": len(no_solution),
    }

    output = {
        "schema_version": "1.0",
        "phase": "m2_3_mini_single_probe",
        "policy": policy_label,
        "adapter_path": args.adapter,
        "base_model": args.base_model,
        "temperature": args.temperature,
        "K": args.K,
        "num_tasks": len(tasks),
        "metrics": metrics,
        "groups": task_groups,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"single_probe_{args.policy}_t{args.temperature}_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved: {out_path}", flush=True)

    print(f"\nSummary: success={successes}/{total}, schema_valid={schema_valid}/{total}, "
          f"premature_finish={premature_finish}, env_errors={env_errors}", flush=True)

    # Cleanup
    del backend, agent, tokenizer
    import torch
    try:
        torch.cuda.empty_cache()
    except Exception as _e:
        print(f"[WARN] empty_cache failed: {_e}", flush=True)


if __name__ == "__main__":
    main()
