"""Rule-based Agent baseline runner for historical continuity."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from .agent_env.environment import ProcurementBrowserEnv
from .agent_env.schemas import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION
from .agent_env.trajectory import compute_metrics, save_trajectories_jsonl
from .agents.rule_based import RuleBasedProcurementAgent
from .tasks import load_public_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "m1_2"


def run_task(
    environment: ProcurementBrowserEnv,
    agent: RuleBasedProcurementAgent,
    task_id: str,
    max_steps: int,
    output_dir: Path,
) -> dict:
    """Run one deterministic rule-policy episode."""
    try:
        observation = environment.reset(task_id)
        agent.reset()
        environment.set_agent_name("rule_based")

        for _ in range(max_steps):
            action = agent.act(observation)
            step_result = environment.step(action)
            if step_result.observation is not None:
                observation = step_result.observation
            if step_result.terminated or step_result.truncated:
                break

        trajectory = environment.trajectory
        if trajectory is None:
            return {
                "task_id": task_id,
                "episode_id": "",
                "success": False,
                "reward": None,
                "rollout_valid": False,
                "failure_origin": "infrastructure",
                "steps": 0,
                "invalid_actions": 0,
                "termination_reason": "trajectory_not_created",
                "error": "no trajectory recorded",
            }

        trajectory.save(output_dir / "per_task")
        return {
            "task_id": task_id,
            "episode_id": trajectory.episode_id,
            "success": trajectory.success,
            "reward": trajectory.final_reward,
            "rollout_valid": True,
            "failure_origin": "none" if trajectory.success else "policy",
            "steps": trajectory.total_steps,
            "invalid_actions": trajectory.invalid_action_count,
            "termination_reason": trajectory.termination_reason,
            "error": None,
        }
    except Exception as exc:
        return {
            "task_id": task_id,
            "episode_id": "",
            "success": False,
            "reward": None,
            "rollout_valid": False,
            "failure_origin": "infrastructure",
            "steps": 0,
            "invalid_actions": 0,
            "termination_reason": "environment_or_runner_error",
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rule-based browser Agent baseline")
    parser.add_argument("--agent", default="rule", choices=["rule"])
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--task-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    task_dir = args.task_dir.expanduser().resolve() if args.task_dir else None
    public_tasks = load_public_tasks(task_dir)
    known_ids = {task["task_id"] for task in public_tasks}
    if args.tasks == "all":
        task_ids = [task["task_id"] for task in public_tasks]
    else:
        task_ids = [value.strip() for value in args.tasks.split(",") if value.strip()]
        unknown = [task_id for task_id in task_ids if task_id not in known_ids]
        if unknown:
            raise ValueError(f"Unknown tasks in selected source: {unknown}")
    if not task_ids:
        raise ValueError("No tasks selected")

    run_id = args.run_id or uuid.uuid4().hex[:12].upper()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = RuleBasedProcurementAgent(max_steps=args.max_steps)
    trajectories = []
    results = []

    print(f"Run ID: {run_id}")
    print(f"Tasks: {len(task_ids)}")
    print(f"Task source: {task_dir or 'default'}")
    print(f"Max steps: {args.max_steps}")

    with ProcurementBrowserEnv(
        max_steps=args.max_steps,
        run_id=run_id,
        task_dir=task_dir,
    ) as environment:
        for index, task_id in enumerate(task_ids, start=1):
            started = time.time()
            result = run_task(environment, agent, task_id, args.max_steps, output_dir)
            results.append(result)
            if environment.trajectory is not None and result["rollout_valid"]:
                trajectories.append(environment.trajectory)
            status = "PASS" if result["success"] else (
                "INFRA" if not result["rollout_valid"] else "FAIL"
            )
            print(
                f"[{index}/{len(task_ids)}] {task_id}: {status} "
                f"({result['steps']} steps, {time.time() - started:.1f}s)"
            )

    metrics = compute_metrics(trajectories)
    metrics.update(
        requested_tasks=len(results),
        infrastructure_errors=sum(1 for result in results if not result["rollout_valid"]),
    )
    save_trajectories_jsonl(
        trajectories,
        output_dir / "m1_2_baseline_trajectories.jsonl",
    )
    (output_dir / "m1_2_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "m1_2_environment_contract.json").write_text(
        json.dumps(
            {
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "action_schema_version": ACTION_SCHEMA_VERSION,
                "trajectory_schema_version": "1.0",
                "supported_actions": [
                    "click", "fill", "select", "check", "back", "submit", "finish"
                ],
                "max_steps": args.max_steps,
                "reward_definition": {
                    "success": 1.0,
                    "policy_failure": 0.0,
                    "infrastructure_failure": None,
                },
                "task_source": str(task_dir) if task_dir else "default",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Results ===")
    print(f"Valid: {metrics['total_tasks']}/{metrics['requested_tasks']}")
    print(f"Success: {metrics['successful_tasks']}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    print(f"Infrastructure: {metrics['infrastructure_errors']}")
    return 0 if metrics["infrastructure_errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
