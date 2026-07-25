"""Agent loop: runs environment + model agent for one task episode."""

import json
import time
import traceback
from pathlib import Path
from typing import Optional

from ..agent_env.environment import ProcurementBrowserEnv
from ..agent_env.schemas import AgentAction


def run_model_episode(
    task_id: str,
    env: ProcurementBrowserEnv,
    agent,
    max_model_turns: int = 20,
    max_env_steps: int = 15,
) -> dict:
    """Run one episode with model agent. Returns model trajectory dict."""
    t0 = time.time()
    turns = []
    result = {
        "task_id": task_id, "episode_id": "", "success": False,
        "reward": 0.0, "termination_reason": "unknown",
        "model_turns": 0, "environment_steps": 0,
        "turns": turns,
    }

    try:
        obs = env.reset(task_id)
        agent.reset(task_id, obs.instruction)
        env.set_agent_name("qwen_base")
        result["episode_id"] = env._episode_id

        consecutive_failures = 0
        done = False

        while not done and agent.model_turn < max_model_turns:
            # Model turn
            attempt = agent.act(obs)
            turns.append({
                "model_turn_index": attempt.model_turn_index,
                "environment_step_index": obs.step_index,
                "observation": obs.to_dict(),
                "rendered_prompt_sha256": attempt.prompt_hash,
                "input_tokens": attempt.input_tokens,
                "raw_output": attempt.raw_output,
                "output_tokens": attempt.output_tokens,
                "latency_ms": attempt.latency_ms,
                "strict_json_success": attempt.strict_json_success,
                "fallback_used": attempt.fallback_used,
                "schema_valid": attempt.schema_valid,
                "action": attempt.action.to_dict() if attempt.action else None,
                "errors": attempt.errors,
                "action_result": None,
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
            })

            # Check for parse/schema failure
            if not attempt.schema_valid or not attempt.action:
                consecutive_failures += 1
                agent.record_feedback(attempt, None, obs.page_type)
                if consecutive_failures >= 3:
                    result["termination_reason"] = "model_output_failure_limit"
                    done = True
                continue

            # Valid action — execute in environment
            consecutive_failures = 0
            step_result = env.step(attempt.action)
            action_result_dict = step_result.info.get("action_result", {})
            agent.record_feedback(attempt, None, obs.page_type)

            # Update last turn
            turns[-1]["action_result"] = action_result_dict
            turns[-1]["reward"] = step_result.reward
            turns[-1]["terminated"] = step_result.terminated
            turns[-1]["truncated"] = step_result.truncated

            if step_result.observation:
                obs = step_result.observation

            if step_result.terminated:
                result["success"] = step_result.reward > 0
                result["reward"] = step_result.reward
                result["termination_reason"] = step_result.info.get("termination_reason", "terminated")
                done = True
            elif step_result.truncated:
                result["termination_reason"] = step_result.info.get("termination_reason", "truncated")
                done = True

        if not done:
            if agent.model_turn >= max_model_turns:
                result["termination_reason"] = "max_model_turns"
            else:
                result["termination_reason"] = "max_environment_steps"

    except Exception as e:
        result["termination_reason"] = "model_error"
        result["error"] = str(e)[:500]
        traceback.print_exc()

    result["model_turns"] = agent.model_turn
    result["environment_steps"] = obs.step_index if 'obs' in dir() else 0
    result["elapsed_s"] = time.time() - t0

    # Get verifier result from trajectory
    if env.trajectory:
        v = env.trajectory.verification
        result["failure_reasons"] = v.get("failure_reasons", [])
        result["verifier_success"] = v.get("success", False)

    result["turns"] = turns
    return result


def save_model_trajectory(task_result: dict, output_dir: Path, model_info: dict, prompt_info: dict):
    """Save model trajectory as versioned JSON."""
    traj = {
        "model_trajectory_schema_version": "1.0",
        "run_id": task_result.get("run_id", ""),
        "task_id": task_result["task_id"],
        "episode_id": task_result["episode_id"],
        "instruction": task_result.get("instruction", ""),
        "model": model_info,
        "prompt": prompt_info,
        "turns": task_result.get("turns", []),
        "final": {
            "success": task_result["success"],
            "reward": task_result["reward"],
            "termination_reason": task_result["termination_reason"],
            "failure_reasons": task_result.get("failure_reasons", []),
            "model_turns": task_result["model_turns"],
            "environment_steps": task_result["environment_steps"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task_result['task_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2, ensure_ascii=False)
    return path
