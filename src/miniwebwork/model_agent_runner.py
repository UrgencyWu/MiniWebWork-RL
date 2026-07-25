"""M2.0 Model Agent Baseline Runner CLI."""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from .agent_env.environment import ProcurementBrowserEnv
from .agent_env.trajectory import compute_metrics as env_compute_metrics
from .model_agent.model_backend import QwenTransformersBackend, ModelConfig
from .model_agent.prompt_builder import prompt_sha256, load_system_prompt
from .model_agent.output_parser import parse
from .model_agent.qwen_agent import QwenBrowserAgent
from .model_agent.agent_loop import run_model_episode, save_model_trajectory
from .model_agent.metrics import compute_metrics
from .model_agent.failure_analysis import classify_failures
from .tasks import load_public_tasks, get_oracle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "m2_0"


def main():
    parser = argparse.ArgumentParser(description="M2.0 Qwen Agent Baseline Runner")
    parser.add_argument("--model-path", default="/data/share/model/Qwen3.5-4B")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--max-model-turns", type=int, default=20)
    parser.add_argument("--max-env-steps", type=int, default=15)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prompt-version", default="browser_agent_v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_id = f"m2_0_{uuid.uuid4().hex[:8]}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tasks
    if args.tasks == "all":
        public_tasks = load_public_tasks()
        task_ids = [t["task_id"] for t in public_tasks]
    else:
        task_ids = [t.strip() for t in args.tasks.split(",")]
    if args.limit > 0:
        task_ids = task_ids[:args.limit]

    print(f"Run ID: {run_id}")
    print(f"Tasks: {len(task_ids)} ({task_ids[0]}..)")
    print(f"Model: {args.model_path}")
    print(f"Max model turns: {args.max_model_turns}")
    print(f"Output: {output_dir}")

    if args.dry_run:
        print("DRY RUN — no model loaded")
        return 0

    # Load model once
    config = ModelConfig(
        model_path=args.model_path,
        max_new_tokens=128,
        enable_thinking=False,
    )
    backend = QwenTransformersBackend(config)
    backend.load()

    model_info = backend.get_model_info()
    prompt_info = {
        "prompt_version": args.prompt_version,
        "prompt_sha256": prompt_sha256(args.prompt_version),
        "chat_template_sha256": backend.get_chat_template_hash(),
        "history_window": args.history_window,
    }

    # Import here to avoid circular
    from .model_agent import prompt_builder as pb
    pb.HISTORY_WINDOW = args.history_window

    agent = QwenBrowserAgent(backend, pb, parse)
    task_results = []
    trajectories = []

    with ProcurementBrowserEnv(max_steps=args.max_env_steps, run_id=run_id) as env:
        for i, task_id in enumerate(task_ids):
            oracle = get_oracle(task_id)
            task_type = oracle.get("task_type", "unknown") if oracle else "unknown"
            print(f"\n[{i+1}/{len(task_ids)}] {task_id} ({task_type})...", end=" ", flush=True)
            start = time.time()

            try:
                result = run_model_episode(task_id, env, agent, args.max_model_turns, args.max_env_steps)
            except Exception as e:
                result = {"task_id": task_id, "success": False, "termination_reason": "model_error",
                          "error": str(e)[:500], "model_turns": 0, "environment_steps": 0, "turns": []}

            result["run_id"] = run_id
            result["task_type"] = task_type
            elapsed = time.time() - start
            status = "PASS" if result["success"] else "FAIL"
            print(f"{status} mt={result['model_turns']} es={result['environment_steps']} ({elapsed:.1f}s) {result['termination_reason']}")

            task_results.append(result)

            # Save per-task
            per_task_dir = output_dir / "per_task"
            per_task_dir.mkdir(parents=True, exist_ok=True)
            with open(per_task_dir / f"{task_id}.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            # Save trajectory
            traj_dir = output_dir / "trajectories"
            save_model_trajectory(result, traj_dir, model_info, prompt_info)

            # Save raw outputs
            raw_dir = output_dir / "raw_outputs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{task_id}.jsonl"
            with open(raw_path, "w") as f:
                for t in result.get("turns", []):
                    f.write(json.dumps({"turn": t.get("model_turn_index", 0),
                                        "raw_output": t.get("raw_output", "")},
                                       ensure_ascii=False) + "\n")

            if env.trajectory:
                trajectories.append(env.trajectory)

    # Compute metrics
    metrics = compute_metrics(task_results)
    metrics["run_id"] = run_id
    failure = classify_failures(task_results)

    # Save
    (output_dir / "m2_0_base_agent_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (output_dir / "m2_0_failure_analysis.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False))
    (output_dir / "m2_0_prompt_manifest.json").write_text(json.dumps(prompt_info, indent=2, ensure_ascii=False))

    # Run manifest
    manifest = {"run_id": run_id, "model_info": model_info, "prompt_info": prompt_info,
                "args": vars(args), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (output_dir / "m2_0_run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Rule comparison
    try:
        rule_metrics_path = PROJECT_ROOT / "artifacts" / "m1_2" / "m1_2_baseline_metrics.json"
        if rule_metrics_path.exists():
            rule_metrics = json.loads(rule_metrics_path.read_text())
        else:
            rule_metrics = {}
    except Exception:
        rule_metrics = {}

    comparison = {
        "rule_agent": {"success": rule_metrics.get("successful_tasks", 2), "total": rule_metrics.get("total_tasks", 15),
                       "rate": rule_metrics.get("success_rate", 0.133), "avg_steps": rule_metrics.get("average_steps", 5.1)},
        "qwen_base": {"success": metrics["successful_tasks"], "total": metrics["total_tasks"],
                      "rate": metrics["success_rate"], "avg_steps": metrics["average_environment_steps"]},
    }
    (output_dir / "m2_0_rule_vs_qwen_comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

    # Summary
    print(f"\n=== Results ===")
    print(f"Success: {metrics['successful_tasks']}/{metrics['total_tasks']} ({metrics['success_rate']:.1%})")
    print(f"Avg model turns: {metrics['average_model_turns']:.1f}, Avg env steps: {metrics['average_environment_steps']:.1f}")
    print(f"Strict JSON rate: {metrics['strict_json_success_rate']:.1%}")
    print(f"Schema valid rate: {metrics['action_schema_valid_rate']:.1%}")
    print(f"Failure primary: {json.dumps(failure.get('summary', {}).get('primary_failure_counts', {}))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
