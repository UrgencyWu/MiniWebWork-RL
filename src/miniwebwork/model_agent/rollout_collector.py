"""Rollout collection for M3.0A audit and future GRPO training.

Provides:
- ``RolloutGroup``: per-task collection of K trajectories + group statistics
- ``RolloutCollector``: orchestrates multi-trajectory episode collection
- ``compute_group_stats``: reward mean/std, success/no-solution rates
- ``is_valid_for_update``: gate for GRPO eligibility
"""

from __future__ import annotations

import json
import time
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class RolloutGroup:
    """All trajectories for one task under a single policy."""

    task_id: str
    policy_version: str              # e.g. "seed_1234"
    prompt_contract_version: str     # e.g. "browser_agent_v2"
    trajectories: list[dict] = field(default_factory=list)

    # Computed by compute_group_stats()
    group_reward_mean: float = 0.0
    group_reward_std: float = 0.0
    success_count: int = 0
    no_solution_selected_count: int = 0
    submission_count: int = 0
    total_model_turns: int = 0
    valid_for_update: bool = False

    # Per-trajectory reward sequence (for variance analysis)
    reward_sequence: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "policy_version": self.policy_version,
            "prompt_contract_version": self.prompt_contract_version,
            "num_trajectories": len(self.trajectories),
            "group_reward_mean": self.group_reward_mean,
            "group_reward_std": self.group_reward_std,
            "success_count": self.success_count,
            "no_solution_selected_count": self.no_solution_selected_count,
            "submission_count": self.submission_count,
            "total_model_turns": self.total_model_turns,
            "valid_for_update": self.valid_for_update,
            "reward_sequence": self.reward_sequence,
            "trajectories": self.trajectories,
        }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


def _count_no_solution_selections(trajectory: dict) -> int:
    """Count how many turns submitted a no_solution decision type."""
    count = 0
    for turn in trajectory.get("turns", []):
        action = turn.get("action") or {}
        # Check for no_solution in the action payload
        payload = turn.get("observation", {}).get("last_action_result", {})
        # Also check if action itself indicates no_solution
        if action.get("action") == "submit":
            # Look for no_solution submission in the observation context
            pass
        # Check the trajectory for no_solution indicators
        raw = turn.get("raw_output", "")
        if "no_solution" in raw or '"no_solution"' in raw:
            count += 1
    return count


def _has_submission(trajectory: dict) -> bool:
    """Check if trajectory produced a valid submission."""
    verification = trajectory.get("verification", {})
    return bool(verification.get("submission_id", ""))


def compute_group_stats(group: RolloutGroup) -> RolloutGroup:
    """Compute group-level statistics from collected trajectories."""
    rewards = [t.get("reward", 0.0) for t in group.trajectories]
    group.reward_sequence = rewards
    group.group_reward_mean = _safe_mean(rewards)
    group.group_reward_std = _safe_std(rewards)
    group.success_count = sum(1 for r in rewards if r > 0.0)
    group.submission_count = sum(1 for t in group.trajectories if _has_submission(t))
    group.no_solution_selected_count = sum(
        1 for t in group.trajectories
        if t.get("verification", {}).get("decision_type", "") == "no_solution"
    )
    group.total_model_turns = sum(
        t.get("model_turns", 0) for t in group.trajectories
    )
    return group


def is_valid_for_update(group: RolloutGroup) -> bool:
    """Check if a rollout group is valid for GRPO-style policy update.

    Requirements:
    1. At least one trajectory succeeded (reward > 0)
    2. Reward variance > 0 (group contains both successes and failures,
       or at least non-identical outcomes)
    """
    has_success = group.success_count > 0
    has_variance = group.group_reward_std > 0.0
    return has_success and has_variance


class RolloutCollector:
    """Collects K trajectories per task and computes rollout statistics."""

    def __init__(
        self,
        backend,
        agent,
        output_dir: Path,
        prompt_contract_version: str = "browser_agent_v2",
    ):
        self.backend = backend
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.prompt_contract_version = prompt_contract_version

        # Sub-directories
        self.traj_dir = self.output_dir / "trajectories"
        self.traj_dir.mkdir(parents=True, exist_ok=True)

    def collect_group(
        self,
        task_id: str,
        K: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        policy_version: str = "",
        max_model_turns: int = 20,
        max_env_steps: int = 15,
        verbose: bool = True,
    ) -> RolloutGroup:
        """Collect K sequential trajectories for one task.

        Each trajectory uses a fresh episode (env.reset) with stochastic
        sampling.  The environment is shared across all K episodes —
        ``env.reset()`` handles cleanup of the previous episode.
        """
        from ..agent_env.environment import ProcurementBrowserEnv

        if not policy_version:
            policy_version = getattr(self.backend, "_policy_version", "unknown")

        group = RolloutGroup(
            task_id=task_id,
            policy_version=policy_version,
            prompt_contract_version=self.prompt_contract_version,
        )

        # Configure backend for sampling
        orig_do_sample = self.backend.config.do_sample
        orig_temperature = self.backend.config.temperature
        orig_top_p = self.backend.config.top_p

        self.backend.config.do_sample = True
        self.backend.config.temperature = temperature
        self.backend.config.top_p = top_p

        try:
            with ProcurementBrowserEnv(max_steps=max_env_steps) as env:
                env.set_agent_name(f"rollout_{policy_version}")

                for k in range(K):
                    if verbose:
                        print(
                            f"  [{k+1}/{K}] {task_id} ... ",
                            end="",
                            flush=True,
                        )

                    t0 = time.time()
                    try:
                        result = self._run_episode(
                            task_id, env, max_model_turns, max_env_steps
                        )
                    except Exception as e:
                        result = self._error_result(task_id, str(e)[:200])

                    elapsed = time.time() - t0
                    result["collect_elapsed_s"] = elapsed

                    group.trajectories.append(result)

                    if verbose:
                        status = "PASS" if result.get("success") else "FAIL"
                        turns = result.get("model_turns", 0)
                        reward = result.get("reward", 0.0)
                        print(
                            f"{status} turns={turns} reward={reward} "
                            f"({elapsed:.1f}s)"
                        )

                    # Clear CUDA cache between episodes (non-critical, ignore errors)
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass  # CUDA errors may be reported here; ignore

        finally:
            # Restore original backend config
            self.backend.config.do_sample = orig_do_sample
            self.backend.config.temperature = orig_temperature
            self.backend.config.top_p = orig_top_p

        compute_group_stats(group)
        group.valid_for_update = is_valid_for_update(group)

        if verbose:
            self._print_group_summary(group)

        return group

    def _run_episode(
        self, task_id: str, env, max_model_turns: int, max_env_steps: int
    ) -> dict:
        """Run a single episode with current agent + environment."""
        from ..model_agent.agent_loop import run_model_episode

        result = run_model_episode(
            task_id, env, self.agent, max_model_turns, max_env_steps
        )

        # Attach logprobs from each turn
        for i, turn in enumerate(result.get("turns", [])):
            turn["logprobs"] = []

        # Build compact trajectory for storage
        compact_trajectory = self._compact_trajectory(result)
        result["_compact_trajectory"] = compact_trajectory

        return result

    def _compact_trajectory(self, result: dict) -> dict:
        """Build a compact version of the trajectory for storage."""
        turns_compact = []
        for turn in result.get("turns", []):
            turns_compact.append({
                "model_turn_index": turn.get("model_turn_index", 0),
                "environment_step_index": turn.get("environment_step_index", 0),
                "raw_output": turn.get("raw_output", ""),
                "input_tokens": turn.get("input_tokens", 0),
                "output_tokens": turn.get("output_tokens", 0),
                "latency_ms": turn.get("latency_ms", 0.0),
                "strict_json_success": turn.get("strict_json_success", False),
                "schema_valid": turn.get("schema_valid", False),
                "action": turn.get("action"),
                "action_result": turn.get("action_result"),
                "logprobs": turn.get("logprobs", []),
                "errors": turn.get("errors", []),
            })

        return {
            "turns": turns_compact,
            "model_turns": result.get("model_turns", 0),
            "environment_steps": result.get("environment_steps", 0),
            "elapsed_s": result.get("elapsed_s", 0.0),
        }

    def _error_result(self, task_id: str, error: str) -> dict:
        """Build an error result dict."""
        return {
            "task_id": task_id,
            "episode_id": "",
            "success": False,
            "reward": 0.0,
            "termination_reason": "model_error",
            "failure_reasons": [],
            "model_turns": 0,
            "environment_steps": 0,
            "turns": [],
            "error": error,
        }

    def _print_group_summary(self, group: RolloutGroup):
        """Print group statistics."""
        print(
            f"  Group stats: "
            f"success={group.success_count}/{len(group.trajectories)}, "
            f"mean_reward={group.group_reward_mean:.3f}, "
            f"std={group.group_reward_std:.3f}, "
            f"submissions={group.submission_count}, "
            f"valid_for_update={group.valid_for_update}"
        )

    def save_group(self, group: RolloutGroup) -> Path:
        """Save RolloutGroup to JSON file."""
        task_dir = self.traj_dir / group.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        path = task_dir / f"rollout_{group.policy_version}_K{len(group.trajectories)}.json"
        path.write_text(
            json.dumps(group.to_dict(), indent=2, ensure_ascii=False, default=str)
        )
        return path

    def save_groups_summary(self, groups: list[RolloutGroup]) -> Path:
        """Save summary of multiple rollout groups."""
        summary = {
            "prompt_contract_version": self.prompt_contract_version,
            "num_tasks": len(groups),
            "groups": [g.to_dict() for g in groups],
            "aggregate": {
                "total_trajectories": sum(len(g.trajectories) for g in groups),
                "total_successes": sum(g.success_count for g in groups),
                "tasks_with_success": sum(1 for g in groups if g.success_count > 0),
                "tasks_with_variance": sum(1 for g in groups if g.group_reward_std > 0),
                "tasks_valid_for_update": sum(1 for g in groups if g.valid_for_update),
            },
        }

        path = self.output_dir / "rollout_summary.json"
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        )
        return path


def analyze_no_solution_behavior(trajectory: dict) -> dict:
    """Analyze a single trajectory's behavior on no-solution tasks.

    Returns a dict with flags indicating which pipeline stages were
    reached:
    - empty_result_reached: reached a page showing 0 results
    - no_solution_action_emitted: model output contains no_solution action
    - no_solution_clicked: clicked the declare-no-solution button
    - form_reached: reached procurement_form page
    - submitted: created a submission in the DB
    - verifier_success: verifier confirmed correctness

    Also extracts:
    - termination_reason
    - verifier_success
    - failure_reasons
    - decision_type
    - model_turns
    - environment_steps
    """
    obs = trajectory.get("turns", [])
    verification = trajectory.get("verification", {})

    result = {
        "empty_result_reached": False,
        "no_solution_action_emitted": False,
        "no_solution_clicked": False,
        "form_reached": False,
        "submitted": bool(verification.get("submission_id")),
        "verifier_success": verification.get("success", False),
        "termination_reason": trajectory.get("termination_reason", ""),
        "failure_reasons": verification.get("failure_reasons", []),
        "decision_type": verification.get("decision_type", ""),
        "model_turns": trajectory.get("model_turns", 0),
        "environment_steps": trajectory.get("environment_steps", 0),
        "submission_id": verification.get("submission_id", ""),
        "error": trajectory.get("error", ""),
    }

    # Scan turns for behavior indicators
    for i, turn in enumerate(trajectory.get("turns", [])):
        page_type = turn.get("observation", {}).get("page_type", "")
        raw_output = turn.get("raw_output", "")
        action = turn.get("action") or {}
        action_result = turn.get("action_result") or {}

        # Check if page shows empty results
        if page_type == "products":
            visible_text = turn.get("observation", {}).get("visible_text", "")
            if "共 0 条" in visible_text or "0 条结果" in visible_text:
                result["empty_result_reached"] = True

        # Check for no_solution action in output
        if "no_solution" in raw_output.lower():
            result["no_solution_action_emitted"] = True

        # Check for declare-no-solution click
        if (action.get("action") == "click" and
                "declare-no-solution" in (action.get("target", "") or "")):
            result["no_solution_clicked"] = True

        # Check for procurement_form page
        if page_type == "procurement_form":
            result["form_reached"] = True

    return result


def compute_no_solution_metrics(trajectories: list[dict]) -> dict:
    """Compute aggregate no-solution behavior metrics from trajectories."""
    total = len(trajectories)
    if total == 0:
        return {"total": 0}

    analyses = [analyze_no_solution_behavior(t) for t in trajectories]

    empty_result = sum(1 for a in analyses if a["empty_result_reached"])
    no_sol_emitted = sum(1 for a in analyses if a["no_solution_action_emitted"])
    no_sol_clicked = sum(1 for a in analyses if a["no_solution_clicked"])
    form_reached = sum(1 for a in analyses if a["form_reached"])
    submitted = sum(1 for a in analyses if a["submitted"])
    verifier_ok = sum(1 for a in analyses if a["verifier_success"])

    # Failure classification
    missing_submission = sum(
        1 for a in analyses
        if "missing_submission" in a["failure_reasons"]
    )
    false_no_solution = sum(
        1 for a in analyses
        if "false_no_solution" in str(a["failure_reasons"])
    )

    return {
        "total": total,
        "empty_result_reached_rate": empty_result / total,
        "no_solution_selected_rate": no_sol_emitted / total,
        "no_solution_clicked_rate": no_sol_clicked / total,
        "form_reached_rate": form_reached / total,
        "submission_persisted_rate": submitted / total,
        "verifier_success_rate": verifier_ok / total,
        "false_no_solution_count": false_no_solution,
        "missing_submission_count": missing_submission,
        "pipeline_stages": {
            "empty_result": empty_result,
            "no_sol_emitted": no_sol_emitted,
            "no_sol_clicked": no_sol_clicked,
            "form_reached": form_reached,
            "submitted": submitted,
            "verifier_ok": verifier_ok,
        },
    }
