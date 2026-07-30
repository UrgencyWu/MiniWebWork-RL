#!/usr/bin/env python3
"""M2.3-mini: Temperature-sweep rollout probe.

Tests the trained adapter at multiple temperatures to verify reward variance.
Reports: success rate, reward variance, schema validity, trajectory diversity.

Usage:
    python scripts/m2_3_mini_rollout_probe.py --adapter outputs/m2_3_mini/seed_1234/final_adapter
    python scripts/m2_3_mini_rollout_probe.py --temperatures 0.2 0.4 0.7 --K 8

Acceptance criteria:
  - At least one task group has mixed rewards (e.g., [0,0,1,0,1,0,0,0])
  - At least half of task groups have reward variance > 0
"""
import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Pre-import environment modules at module level so missing dependencies
# fail fast with a clear error instead of being silently swallowed inside
# the rollout loop.
try:
    from miniwebwork.agent_env.environment import ProcurementBrowserEnv  # noqa: F401
    from miniwebwork.tasks import get_oracle  # noqa: F401
    from miniwebwork.verifier import verify_episode  # noqa: F401
except ImportError as _e:
    # Defer the error so the script can still print --help etc.
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")
DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
DEFAULT_TASK_DIR = PROJECT_ROOT / "data" / "tasks" / "rollout_dev_no_solution_v1"
DEFAULT_TEMPERATURES = [0.2, 0.4, 0.7]
DEFAULT_K = 8
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m2_3_mini"


def load_policy(base_model_path, adapter_path, temperature=0.7):
    """Load base model + LoRA adapter and create backend + agent."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from miniwebwork.model_agent.model_backend import ModelConfig, QwenTransformersBackend
    from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
    from miniwebwork.model_agent.output_parser import parse
    import miniwebwork.model_agent.prompt_builder as pb

    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model...")
    # Pin model to cuda:0. Using device_map="auto" with all visible GPUs
    # triggers IndexKernel crashes when any GPU is near-full (e.g. from
    # other Slurm jobs), because caching_allocator_warmup calls
    # cudaMemGetInfo on every device in the map including saturated ones.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
    )
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model, adapter_path, torch_dtype=torch.bfloat16
        )
    else:
        print(f"WARNING: adapter not found at {adapter_path}, using base model")
        model = base_model

    model.eval()

    # Enable adapter layers (adapter was saved with inference_mode=True)
    model.enable_adapter_layers()
    print(f"  [DEBUG] Model loaded, active_adapters={model.active_adapters}")

    # Model is already on cuda:0 via device_map

    # Warm-up forward pass to initialize CUDA kernels
    print(f"  [DEBUG] Running warm-up forward pass...")
    with torch.inference_mode():
        _dummy = tokenizer("warmup", return_tensors="pt").to(model.device)
        _ = model(**_dummy)
    print(f"  [DEBUG] Warm-up done")

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
    print(f"  [DEBUG] Creating backend...")
    backend = QwenTransformersBackend(config)
    backend._model = model
    backend._tokenizer = tokenizer
    backend._loaded = True
    print(f"  [DEBUG] Backend created")

    pb.HISTORY_WINDOW = 5
    from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
    print(f"  [DEBUG] Creating agent...")
    agent = QwenBrowserAgent(backend, pb, parse)
    print(f"  [DEBUG] Returning from load_policy")
    return backend, agent, tokenizer


def load_tasks(task_dir: Path, split: str = "valid") -> list:
    """Load tasks for rollout probe."""
    public_path = task_dir / f"{split}_public.jsonl"
    if not public_path.exists():
        raise FileNotFoundError(f"Task file not found: {public_path}")

    tasks = []
    for line in public_path.read_text().strip().split("\n"):
        if line.strip():
            tasks.append(json.loads(line))
    return tasks


def run_single_rollout(task_id: str, adapter_path: str, temperature: float,
                        K: int, base_model_path: str,
                        backend=None, agent=None, tokenizer=None) -> list:
    """Run K rollouts for a single task at a given temperature.

    If backend/agent/tokenizer are provided, reuse them (avoids reloading model per task).
    """
    from miniwebwork.agent_env.schemas import AgentAction

    if backend is None:
        backend, agent, tokenizer = load_policy(base_model_path, adapter_path, temperature)

    trajectories = []
    for k in range(K):
        run_id = f"probe_{uuid.uuid4().hex[:8]}"
        os.environ["MINIWEBWORK_TASK_DIR"] = str(DEFAULT_TASK_DIR)

        try:
            with ProcurementBrowserEnv(max_steps=25, run_id=run_id, headless=True) as env:
                env.set_agent_name("rollout_probe")

                obs = env.reset(task_id)
                print(f"  [DEBUG] Env reset: page={obs.page_type}")
                agent.reset(obs.task_id, obs.instruction)
                print(f"  [DEBUG] Agent reset")
                done = False
                model_turns = 0
                env_steps = 0
                schema_valid_count = 0
                total_turns = 0

                for step in range(25):
                    print(f"  [DEBUG] Step {step}, calling agent.act()...")
                    attempt = agent.act(obs)
                    print(f"  [DEBUG] agent.act() done: schema_valid={attempt.schema_valid}")
                    schema_valid = attempt.schema_valid
                    raw_output = attempt.raw_output or ""

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
                        total_turns += 1

                        if terminated or truncated:
                            done = True
                            break
                    except Exception as e:
                        terminated = True
                        reward = 0.0
                        done = True
                        break

                # Verify
                traj = env.trajectory
                verification = traj.verification if traj else {}
                success = verification.get("success", False)

                trajectories.append({
                    "task_id": task_id,
                    "episode_id": run_id,
                    "k": k,
                    "temperature": temperature,
                    "success": success,
                    "reward": 1.0 if success else 0.0,
                    "termination_reason": (traj.termination_reason if traj else "error"),
                    "model_turns": model_turns,
                    "environment_steps": env_steps,
                    "schema_valid_count": schema_valid_count,
                    "total_turns": total_turns,
                    "verification": verification,
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            trajectories.append({
                "task_id": task_id,
                "episode_id": f"error_{k}",
                "k": k,
                "temperature": temperature,
                "success": False,
                "reward": 0.0,
                "termination_reason": f"error: {str(e)[:100]}",
                "model_turns": 0,
                "environment_steps": 0,
                "schema_valid_count": 0,
                "total_turns": 0,
            })

    return trajectories


def cleanup_policy(backend, agent, tokenizer):
    """Free GPU memory after all tasks for a temperature are done."""
    try:
        del backend, agent, tokenizer
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def classify_trajectory(traj: dict) -> str:
    """Classify the failure mode of a trajectory."""
    term_reason = traj.get("termination_reason", "")
    if term_reason == "model_output_failure_limit":
        return "output_format_failure"
    elif term_reason == "max_model_turns":
        return "max_turns_reached"
    elif traj.get("success"):
        return "success"
    elif traj.get("model_turns", 0) == 0:
        return "immediate_failure"
    else:
        return "other_failure"


def main():
    parser = argparse.ArgumentParser(description="M2.3-mini temperature-sweep rollout probe")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (default: outputs/m2_3_mini/seed_1234/final_adapter)")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--K", type=int, default=DEFAULT_K, help="Rollouts per task per temperature")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks (for testing)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # Fail fast on missing dependencies
    if _IMPORT_ERROR is not None:
        print(f"ERROR: Failed to import environment modules: {_IMPORT_ERROR}")
        print("  Ensure the 'miniwebwork' conda environment is activated with playwright installed.")
        sys.exit(1)

    # Find adapter
    if args.adapter:
        adapter_path = Path(args.adapter)
    else:
        adapter_path = PROJECT_ROOT / "outputs" / "m2_3_mini" / "seed_1234" / "final_adapter"
        if not adapter_path.exists():
            adapter_path = PROJECT_ROOT / "outputs" / "m2_2r" / "seed_1234" / "final_adapter"

    if not adapter_path.exists():
        print(f"ERROR: No adapter found at {adapter_path}")
        print("  Specify --adapter <path> or train first")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MINIWEBWORK_TASK_DIR"] = str(args.task_dir)

    print(f"=== M2.3-mini Temperature-Sweep Rollout Probe ===")
    print(f"Adapter: {adapter_path}")
    print(f"Temperatures: {args.temperatures}")
    print(f"K per task: {args.K}")
    print(f"Task dir: {args.task_dir}")

    # Load tasks
    tasks = load_tasks(args.task_dir, args.split)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    print(f"Tasks: {len(tasks)} ({args.split})")

    all_results = []
    task_summaries = []

    for temp in args.temperatures:
        print(f"\n{'='*60}")
        print(f"Temperature: {temp}")
        print(f"{'='*60}")

        # Load model ONCE per temperature (not per task)
        print(f"  Loading policy (temp={temp})...")
        backend, agent, tokenizer = load_policy(
            args.base_model, str(adapter_path), temp
        )

        temp_results = []
        temp_groups = []

        for task in tasks:
            tid = task["task_id"]
            print(f"\n  Task: {tid} (K={args.K})")
            t0 = time.time()

            # Reuse loaded backend/agent/tokenizer
            trajs = run_single_rollout(
                tid, str(adapter_path), temp, args.K, args.base_model,
                backend=backend, agent=agent, tokenizer=tokenizer,
            )
            temp_results.extend(trajs)

            rewards = [t["reward"] for t in trajs]
            successes = sum(rewards)
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            schema_valid = sum(1 for t in trajs if t.get("schema_valid_count", 0) > 0)
            has_variance = std_r > 0
            valid_for_update = has_variance and successes > 0

            group = {
                "task_id": tid,
                "policy_version": "m2_3_mini_seed_1234",
                "temperature": temp,
                "K": args.K,
                "num_trajectories": len(trajs),
                "group_reward_mean": mean_r,
                "group_reward_std": std_r,
                "success_count": successes,
                "has_variance": has_variance,
                "valid_for_update": valid_for_update,
                "reward_sequence": rewards,
                "schema_valid_count": schema_valid,
                "elapsed_s": time.time() - t0,
                "trajectories": trajs,
            }
            temp_groups.append(group)

            print(f"    Success: {successes}/{args.K}, mean={mean_r:.2f}, std={std_r:.3f}, "
                  f"variance={has_variance}, valid={valid_for_update}")

        all_results.extend(temp_results)

        # Temperature-level summary
        temp_successes = sum(g["success_count"] for g in temp_groups)
        temp_total = sum(g["num_trajectories"] for g in temp_groups)
        temp_groups_with_var = sum(1 for g in temp_groups if g["has_variance"])
        temp_valid_groups = sum(1 for g in temp_groups if g["valid_for_update"])

        print(f"\n  Temperature {temp} Summary:")
        print(f"    Success: {temp_successes}/{temp_total} ({temp_successes/temp_total:.1%})")
        print(f"    Groups with variance: {temp_groups_with_var}/{len(temp_groups)}")
        print(f"    Valid for update: {temp_valid_groups}/{len(temp_groups)}")

        # Failure distribution
        failure_counts = defaultdict(int)
        for g in temp_groups:
            for t in g["trajectories"]:
                failure_counts[classify_trajectory(t)] += 1
        print(f"    Failure distribution: {dict(failure_counts)}")

        # Free GPU memory before next temperature
        cleanup_policy(backend, agent, tokenizer)

        task_summaries.extend(temp_groups)

    # Overall summary
    total_trajs = len(all_results)
    total_successes = sum(1 for t in all_results if t["success"])
    total_groups_with_var = sum(1 for g in task_summaries if g["has_variance"])
    total_valid_groups = sum(1 for g in task_summaries if g["valid_for_update"])
    groups_with_mixed = sum(1 for g in task_summaries
                            if 0 in g["reward_sequence"] and 1 in g["reward_sequence"])

    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total trajectories: {total_trajs}")
    print(f"  Total successes: {total_successes}/{total_trajs} ({total_successes/total_trajs:.1%})")
    print(f"  Groups with variance: {total_groups_with_var}/{len(task_summaries)}")
    print(f"  Groups with mixed rewards: {groups_with_mixed}/{len(task_summaries)}")
    print(f"  Valid for GRPO: {total_valid_groups}/{len(task_summaries)}")

    # Acceptance check
    print(f"\n=== ACCEPTANCE CHECK ===")
    has_mixed = groups_with_mixed > 0
    half_have_var = total_groups_with_var >= len(task_summaries) / 2
    print(f"  Mixed rewards in at least one group: {'PASS' if has_mixed else 'FAIL'}")
    print(f"  At least half groups have variance: {'PASS' if half_have_var else 'FAIL'}")

    if has_mixed:
        print(f"\n  >>> ROUTE A ACHIEVED: Ready for M3.0B GRPO <<<")
    elif total_groups_with_var > 0:
        print(f"\n  >>> PARTIAL: Some variance but no mixed rewards <<<")
    else:
        print(f"\n  >>> STILL ROUTE B: No reward variance, needs more data <<<")

    # Save results
    output = {
        "schema_version": "1.0",
        "phase": "m2_3_mini_rollout_probe",
        "adapter_path": str(adapter_path),
        "base_model": args.base_model,
        "temperatures": args.temperatures,
        "K": args.K,
        "num_tasks": len(tasks),
        "total_trajectories": total_trajs,
        "total_successes": total_successes,
        "success_rate": total_successes / max(total_trajs, 1),
        "groups_with_variance": total_groups_with_var,
        "groups_with_mixed_rewards": groups_with_mixed,
        "valid_for_update_groups": total_valid_groups,
        "route_a_achieved": has_mixed,
        "groups": task_summaries,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"rollout_probe_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved: {out_path}")

    # Also save as latest
    latest_path = args.output_dir / "rollout_probe_latest.json"
    latest_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n=== Complete ===")
    return has_mixed


if __name__ == "__main__":
    main()
