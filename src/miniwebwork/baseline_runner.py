"""
Baseline runner for evaluating agents on all procurement tasks.

Usage:
    python -m miniwebwork.baseline_runner --agent rule --tasks all --max-steps 20
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from .agent_env.environment import ProcurementBrowserEnv
from .agent_env.trajectory import compute_metrics, save_trajectories_jsonl
from .agents.rule_based import RuleBasedProcurementAgent
from .tasks import load_public_tasks

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "m1_2"


def run_task(env: ProcurementBrowserEnv, agent: RuleBasedProcurementAgent,
             task_id: str, max_steps: int, output_dir: Path) -> dict:
    """Run a single task episode and return trajectory metrics."""
    instruction = ""
    try:
        # Reset environment
        obs = env.reset(task_id)
        instruction = obs.instruction
        agent.reset()

        # Set agent name on env
        env.set_agent_name("rule_based")

        # Run episode
        result = None
        for step_idx in range(max_steps):
            action = agent.act(obs)
            result = env.step(action)
            obs = result.observation if result.observation else obs
            if result.terminated or result.truncated:
                break

        traj = env.trajectory
        if traj:
            per_task_dir = output_dir / "per_task"
            traj.save(per_task_dir)
            return {
                "task_id": task_id,
                "episode_id": traj.episode_id,
                "success": traj.success,
                "reward": traj.final_reward,
                "steps": traj.total_steps,
                "invalid_actions": traj.invalid_action_count,
                "termination_reason": traj.termination_reason,
                "error": None,
            }

    except Exception as e:
        return {
            "task_id": task_id,
            "episode_id": "",
            "success": False,
            "reward": 0.0,
            "steps": 0,
            "invalid_actions": 0,
            "termination_reason": "error",
            "error": str(e)[:200],
        }

    return {
        "task_id": task_id,
        "episode_id": "",
        "success": False,
        "reward": 0.0,
        "steps": 0,
        "invalid_actions": 0,
        "termination_reason": "unknown",
        "error": "no trajectory recorded",
    }


def main():
    parser = argparse.ArgumentParser(description="MiniWebWork-RL Baseline Runner")
    parser.add_argument("--agent", default="rule", choices=["rule"], help="Agent type")
    parser.add_argument("--tasks", default="all", help="Task IDs comma-separated or 'all'")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    args = parser.parse_args()

    run_id = args.run_id or uuid.uuid4().hex[:12].upper()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine tasks
    if args.tasks == "all":
        public_tasks = load_public_tasks()
        task_ids = [t["task_id"] for t in public_tasks]
    else:
        task_ids = [t.strip() for t in args.tasks.split(",")]

    print(f"Run ID: {run_id}")
    print(f"Tasks: {len(task_ids)}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {output_dir}")

    # Setup
    agent = RuleBasedProcurementAgent(max_steps=args.max_steps)
    trajectories = []
    results = []

    with ProcurementBrowserEnv(max_steps=args.max_steps, run_id=run_id) as env:
        for i, task_id in enumerate(task_ids):
            print(f"\n[{i+1}/{len(task_ids)}] {task_id}...", end=" ", flush=True)
            start = time.time()
            result = run_task(env, agent, task_id, args.max_steps, output_dir)
            elapsed = time.time() - start
            status = "PASS" if result["success"] else "FAIL"
            print(f"{status} ({result['steps']} steps, {elapsed:.1f}s)")
            if result.get("error"):
                print(f"  Error: {result['error']}")
            results.append(result)

            if env.trajectory:
                trajectories.append(env.trajectory)

    # Compute metrics
    metrics = compute_metrics(trajectories)

    # Save trajectories JSONL
    jsonl_path = output_dir / "m1_2_baseline_trajectories.jsonl"
    save_trajectories_jsonl(trajectories, jsonl_path)

    # Save metrics
    metrics_path = output_dir / "m1_2_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save environment contract
    contract = {
        "observation_schema_version": "1.0",
        "action_schema_version": "1.0",
        "trajectory_schema_version": "1.0",
        "supported_actions": ["click", "fill", "select", "check", "back", "submit", "finish"],
        "max_steps": args.max_steps,
        "reward_definition": "terminal binary: success=1.0, failure=0.0",
        "termination_definitions": {
            "terminated": "submission created and verified, or finish with submission",
            "truncated": "max_steps reached or browser error",
        },
        "observation_limits": {"max_visible_text": 8000},
        "security_restrictions": [
            "No CSS selectors in actions", "No XPath", "No JavaScript",
            "No Oracle access", "No database access", "No verifier access",
            "element_id only from current observation",
        ],
    }
    contract_path = output_dir / "m1_2_environment_contract.json"
    with open(contract_path, "w") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n=== Results ===")
    print(f"Total: {metrics['total_tasks']}")
    print(f"Success: {metrics['successful_tasks']}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    print(f"Average Steps: {metrics['average_steps']:.1f}")
    print(f"Median Steps: {metrics['median_steps']}")
    print(f"Max Steps: {metrics['max_steps']}")
    print(f"Truncated: {metrics['truncated_tasks']}")
    print(f"Invalid Actions: {metrics['total_invalid_actions']}")
    print(f"Invalid Rate: {metrics['invalid_action_rate']:.4f}")
    print(f"\nTrajectories: {jsonl_path}")
    print(f"Metrics: {metrics_path}")

    # Return exit code based on existence of results
    success = metrics["successful_tasks"] > 0
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
