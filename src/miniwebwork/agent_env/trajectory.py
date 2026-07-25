"""Trajectory recorder for environment episodes."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class TrajectoryRecorder:
    """Records step-by-step trajectories for a single episode."""

    def __init__(self, run_id: str, task_id: str, episode_id: str,
                 instruction: str, agent_name: str, max_steps: int):
        self.run_id = run_id
        self.task_id = task_id
        self.episode_id = episode_id
        self.instruction = instruction
        self.agent_name = agent_name
        self.max_steps = max_steps
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at = ""
        self.steps = []
        self.final_reward = 0.0
        self.success = False
        self.termination_reason = ""
        self.total_steps = 0
        self.invalid_action_count = 0
        self.verification = {}

    def record_step(self, step_index: int, observation, action: Optional[dict],
                    action_result: Optional[dict], reward: float,
                    terminated: bool, truncated: bool, elapsed_ms: int):
        self.steps.append({
            "step_index": step_index,
            "observation": observation.to_dict() if observation else None,
            "action": action,
            "action_result": action_result,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "elapsed_ms": elapsed_ms,
        })
        if action_result and not action_result.get("success", True):
            self.invalid_action_count += 1
        self.total_steps = step_index + 1

    def finalize(self, final_reward: float, success: bool, termination_reason: str,
                 verification: dict = None):
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.final_reward = final_reward
        self.success = success
        self.termination_reason = termination_reason
        if verification:
            self.verification = verification

    def to_dict(self) -> dict:
        return {
            "trajectory_schema_version": "1.0",
            "run_id": self.run_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "max_steps": self.max_steps,
            "steps": self.steps,
            "final_reward": self.final_reward,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "total_steps": self.total_steps,
            "invalid_action_count": self.invalid_action_count,
            "verification": self.verification,
        }

    def save(self, output_dir: Path):
        """Save trajectory as JSON to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.episode_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path


def save_trajectories_jsonl(trajectories: list, output_path: Path):
    """Save all trajectories as a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")


def compute_metrics(trajectories: list) -> dict:
    """Compute aggregate metrics from a list of trajectories."""
    total = len(trajectories)
    if total == 0:
        return {"total_tasks": 0}

    successful = [t for t in trajectories if t.success]
    failed = [t for t in trajectories if not t.success and t.termination_reason != "truncated"]
    truncated = [t for t in trajectories if t.termination_reason == "truncated"]
    premature = [t for t in trajectories if t.termination_reason == "premature_finish"]

    steps = [t.total_steps for t in trajectories]
    sorted_steps = sorted(steps)
    median_steps = sorted_steps[len(sorted_steps) // 2] if sorted_steps else 0

    total_invalid = sum(t.invalid_action_count for t in trajectories)
    total_actions = sum(t.total_steps for t in trajectories)
    invalid_rate = total_invalid / max(total_actions, 1)

    # Task type breakdown
    task_type_results = {}
    for t in trajectories:
        oracle_type = t.verification.get("task_type", "unknown") if t.verification else "unknown"
        if oracle_type not in task_type_results:
            task_type_results[oracle_type] = {"total": 0, "success": 0}
        task_type_results[oracle_type]["total"] += 1
        if t.success:
            task_type_results[oracle_type]["success"] += 1

    termination_reasons = {}
    for t in trajectories:
        reason = t.termination_reason
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

    per_task = []
    for t in trajectories:
        per_task.append({
            "task_id": t.task_id,
            "episode_id": t.episode_id,
            "success": t.success,
            "reward": t.final_reward,
            "steps": t.total_steps,
            "invalid_actions": t.invalid_action_count,
            "termination_reason": t.termination_reason,
        })

    return {
        "total_tasks": total,
        "successful_tasks": len(successful),
        "success_rate": len(successful) / total,
        "failed_tasks": len(failed),
        "truncated_tasks": len(truncated),
        "premature_finish_count": len(premature),
        "average_steps": sum(steps) / max(total, 1),
        "median_steps": median_steps,
        "max_steps": max(steps) if steps else 0,
        "total_invalid_actions": total_invalid,
        "invalid_action_rate": round(invalid_rate, 4),
        "task_type_breakdown": task_type_results,
        "termination_reason_breakdown": termination_reasons,
        "per_task": per_task,
    }
