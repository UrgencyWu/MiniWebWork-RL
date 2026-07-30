#!/usr/bin/env python3
"""M2.3-mini: Controlled comparison rollout probe.

Compares Policy A (M2.2R seed_1234) vs Policy B (M2.3-mini seed_1234)
under identical conditions.
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
DEFAULT_TEMPERATURES = [0.2, 0.4, 0.7]
DEFAULT_K = 8
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m2_3_mini"


def load_policy(base_model_path, adapter_path, temperature=0.7):
    import torch
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

    print(f"  Loading base model (pinned to cuda:0)...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"  Loading adapter: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(
            base_model, adapter_path, torch_dtype=torch.bfloat16,
        )
    else:
        print(f"  WARNING: adapter not found, using base model", flush=True)
        model = base_model

    model.eval()
    model.enable_adapter_layers()
    print(f"  Active adapters: {model.active_adapters}", flush=True)

    print(f"  Warm-up forward...", flush=True)
    with torch.inference_mode():
        _dummy = tokenizer("warmup", return_tensors="pt").to(model.device)
        _ = model(**_dummy)
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

    pb.HISTORY_WINDOW = 5
    agent = QwenBrowserAgent(backend, pb, parse)
    return backend, agent, tokenizer


def run_single_rollout(task_id, adapter_path, temperature, K, base_model_path,
                        backend=None, agent=None, tokenizer=None):
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
                agent.reset(obs.task_id, obs.instruction)
                done = False
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
                            done = True
                            break
                    except Exception:
                        terminated = True
                        reward = 0.0
                        done = True
                        break

                traj = env.trajectory
                verification = traj.verification if traj else {}
                success = verification.get("success", False)
                termination_reason = traj.termination_reason if traj else "error"

                trajectories.append({
                    "task_id": task_id,
                    "episode_id": run_id,
                    "k": k,
                    "temperature": temperature,
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
                "temperature": temperature,
                "success": False,
                "reward": 0.0,
                "termination_reason": f"error: {str(e)[:200]}",
                "model_turns": 0,
                "environment_steps": 0,
                "schema_valid_count": 0,
                "verification": {},
            })

    return trajectories


def compute_summary_metrics(trajectories):
    total = len(trajectories)
    if total == 0:
        return {}
    successes = sum(1 for t in trajectories if t["success"])
    schema_valid = sum(1 for t in trajectories if t.get("schema_valid_count", 0) > 0)
    immediate_failures = sum(1 for t in trajectories if t.get("model_turns", 0) == 0)
    premature_finish = sum(1 for t in trajectories if t.get("termination_reason") == "premature_finish")
    env_errors = sum(1 for t in trajectories if "error:" in str(t.get("termination_reason", "")))
    no_solution_tasks = [t for t in trajectories if t.get("task_id", "").startswith("M2_3_V")]
    no_solution_successes = sum(1 for t in no_solution_tasks if t["success"])
    return {
        "total_trajectories": total,
        "total_successes": successes,
        "success_rate": successes / total,
        "schema_valid_count": schema_valid,
        "schema_valid_rate": schema_valid / total,
        "immediate_failures": immediate_failures,
        "premature_finish": premature_finish,
        "env_errors": env_errors,
        "env_error_rate": env_errors / total,
        "no_solution_tasks": len(no_solution_tasks),
        "no_solution_successes": no_solution_successes,
        "no_solution_success_rate": no_solution_successes / max(len(no_solution_tasks), 1),
    }


def run_policy_probe(policy_label, adapter_path, base_model_path, tasks, temperatures, K):
    print(f"\n{'='*60}")
    print(f"POLICY: {policy_label}")
    print(f"Adapter: {adapter_path}")
    print(f"{'='*60}", flush=True)

    all_results = []
    task_summaries = []

    for temp in temperatures:
        print(f"\n  Temperature: {temp}", flush=True)
        backend, agent, tokenizer = load_policy(base_model_path, adapter_path, temp)

        temp_results = []
        temp_groups = []

        for task in tasks:
            tid = task["task_id"]
            print(f"    Task: {tid} (K={K})", flush=True)
            t0 = time.time()

            trajs = run_single_rollout(
                tid, adapter_path, temp, K, base_model_path,
                backend=backend, agent=agent, tokenizer=tokenizer,
            )
            temp_results.extend(trajs)

            rewards = [t["reward"] for t in trajs]
            successes = sum(rewards)
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            has_variance = std_r > 0
            valid_for_update = has_variance and successes > 0

            group = {
                "task_id": tid,
                "policy": policy_label,
                "temperature": temp,
                "K": K,
                "num_trajectories": len(trajs),
                "group_reward_mean": mean_r,
                "group_reward_std": std_r,
                "success_count": successes,
                "has_variance": has_variance,
                "valid_for_update": valid_for_update,
                "reward_sequence": rewards,
                "schema_valid_count": sum(1 for t in trajs if t.get("schema_valid_count", 0) > 0),
                "elapsed_s": time.time() - t0,
                "trajectories": trajs,
            }
            temp_groups.append(group)

            print(f"      Success: {successes}/{K}, mean={mean_r:.2f}, std={std_r:.3f}, "
                  f"var={has_variance}, valid={valid_for_update}", flush=True)

        all_results.extend(temp_results)
        task_summaries.extend(temp_groups)

        temp_successes = sum(g["success_count"] for g in temp_groups)
        temp_total = sum(g["num_trajectories"] for g in temp_groups)
        temp_groups_with_var = sum(1 for g in temp_groups if g["has_variance"])
        print(f"\n  Temp {temp}: {temp_successes}/{temp_total} success, "
              f"{temp_groups_with_var}/{len(temp_groups)} groups with variance", flush=True)

        del backend, agent, tokenizer
        import torch
        torch.cuda.empty_cache()

    return all_results, task_summaries


def main():
    parser = argparse.ArgumentParser(description="M2.3-mini controlled comparison probe")
    parser.add_argument("--policy-a", type=str, required=True)
    parser.add_argument("--policy-b", type=str, required=True)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        print(f"ERROR: {_IMPORT_ERROR}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MINIWEBWORK_TASK_DIR"] = str(args.task_dir)

    tasks = []
    public_path = args.task_dir / f"{args.split}_public.jsonl"
    for line in public_path.read_text().strip().split("\n"):
        if line.strip():
            tasks.append(json.loads(line))
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"\nM2.3-mini Controlled Comparison Probe")
    print(f"Policy A: {args.policy_a}")
    print(f"Policy B: {args.policy_b}")
    print(f"Tasks: {len(tasks)}, Temps: {args.temperatures}, K={args.K}", flush=True)

    trajs_a, groups_a = run_policy_probe(
        "A_M2.2R", args.policy_a, args.base_model, tasks, args.temperatures, args.K,
    )
    trajs_b, groups_b = run_policy_probe(
        "B_M2.3-mini", args.policy_b, args.base_model, tasks, args.temperatures, args.K,
    )

    metrics_a = compute_summary_metrics(trajs_a)
    metrics_b = compute_summary_metrics(trajs_b)

    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")

    for label, m in [("Policy A (M2.2R)", metrics_a), ("Policy B (M2.3-mini)", metrics_b)]:
        print(f"\n{label}:")
        print(f"  Success rate: {m.get('success_rate', 0):.1%} ({m.get('total_successes', 0)}/{m.get('total_trajectories', 0)})")
        print(f"  Schema valid: {m.get('schema_valid_rate', 0):.1%}")
        print(f"  Premature finish: {m.get('premature_finish', 0)}")
        print(f"  Env error rate: {m.get('env_error_rate', 0):.1%}")
        print(f"  No-solution successes: {m.get('no_solution_successes', 0)}/{m.get('no_solution_tasks', 0)}")

    # Variance analysis
    print(f"\n{'='*60}")
    print(f"REWARD VARIANCE")
    print(f"{'='*60}")
    all_mixed_a = 0
    all_mixed_b = 0
    all_var_a = 0
    all_var_b = 0

    for temp in args.temperatures:
        ga = [g for g in groups_a if g["temperature"] == temp]
        gb = [g for g in groups_b if g["temperature"] == temp]
        var_a = sum(1 for g in ga if g["has_variance"])
        var_b = sum(1 for g in gb if g["has_variance"])
        mixed_a = sum(1 for g in ga if 0 in g["reward_sequence"] and 1 in g["reward_sequence"])
        mixed_b = sum(1 for g in gb if 0 in g["reward_sequence"] and 1 in g["reward_sequence"])
        all_mixed_a += mixed_a
        all_mixed_b += mixed_b
        all_var_a += var_a
        all_var_b += var_b
        print(f"  T={temp}: A var={var_a}/{len(ga)} mixed={mixed_a} | B var={var_b}/{len(gb)} mixed={mixed_b}")

    # GRPO readiness
    checks = {
        "No-solution task success": metrics_b.get("no_solution_successes", 0) > 0,
        "Mixed rewards": all_mixed_b > 0,
        "At least half groups have variance": all_var_b >= len(groups_b) / 2,
        "Env error rate < 5%": metrics_b.get("env_error_rate", 1) < 0.05,
        "Schema valid not degraded": metrics_b.get("schema_valid_rate", 0) >= metrics_a.get("schema_valid_rate", 0) * 0.9,
    }

    grpo_ready = all(checks.values())
    print(f"\n{'='*60}")
    print(f"GRPO READINESS")
    print(f"{'='*60}")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")
    print(f"\n  >>> {'ROUTE A: Ready for M3.0B GRPO' if grpo_ready else 'NOT READY'} <<<")

    output = {
        "schema_version": "1.0",
        "phase": "m2_3_mini_comparison_probe",
        "base_model": args.base_model,
        "temperatures": args.temperatures,
        "K": args.K,
        "num_tasks": len(tasks),
        "policy_a": {
            "label": "A_M2.2R_seed_1234",
            "adapter_path": args.policy_a,
            "metrics": metrics_a,
            "groups": groups_a,
            "total_trajectories": len(trajs_a),
        },
        "policy_b": {
            "label": "B_M2.3-mini_seed_1234",
            "adapter_path": args.policy_b,
            "metrics": metrics_b,
            "groups": groups_b,
            "total_trajectories": len(trajs_b),
        },
        "grpo_readiness": {
            "all_checks_passed": grpo_ready,
            "checks": {k: bool(v) for k, v in checks.items()},
        },
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"comparison_probe_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")
    return grpo_ready


if __name__ == "__main__":
    main()
