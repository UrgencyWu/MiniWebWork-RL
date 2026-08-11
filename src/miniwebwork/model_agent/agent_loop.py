"""Canonical closed-loop execution for a model browser Agent."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from pathlib import Path

from ..agent_env.environment import ProcurementBrowserEnv


INFRASTRUCTURE_ERROR_PREFIXES = ("generation_error", "rollout_evidence_error")


def run_model_episode(
    task_id: str,
    env: ProcurementBrowserEnv,
    agent,
    max_model_turns: int = 20,
    max_env_steps: int = 15,
    turn_generated_callback: Callable[[dict], None] | None = None,
    turn_completed_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Run one episode and return a structured evaluation trajectory.

    Policy failures are valid outcomes with reward 0. Model-backend,
    environment, browser, service, and database exceptions invalidate the
    rollout and carry reward ``None``.
    """
    if max_model_turns <= 0 or max_env_steps <= 0:
        raise ValueError("model/environment turn limits must be positive")

    started = time.time()
    turns: list[dict] = []
    result = {
        "task_id": task_id,
        "episode_id": "",
        "success": False,
        "reward": 0.0,
        "task_score": 0.0,
        "rollout_valid": True,
        "failure_origin": "policy",
        "termination_reason": "unknown",
        "model_turns": 0,
        "environment_steps": 0,
        "turns": turns,
    }
    observation = None
    environment_steps = 0

    try:
        observation = env.reset(task_id)
        agent.reset(task_id, observation.instruction)
        env.set_agent_name("qwen_browser_agent")
        result["episode_id"] = observation.episode_id
        consecutive_output_failures = 0

        while agent.model_turn < max_model_turns:
            attempt = agent.act(observation)
            turn = {
                "model_turn_index": attempt.model_turn_index,
                "environment_step_index": observation.step_index,
                "observation": observation.to_dict(),
                "rendered_prompt_sha256": attempt.prompt_hash,
                "prompt_token_ids": attempt.prompt_token_ids,
                "input_tokens": attempt.input_tokens,
                "raw_output": attempt.raw_output,
                "generated_token_ids": attempt.generated_token_ids,
                "token_logprobs": attempt.token_logprobs,
                "sampling_logprobs": attempt.sampling_logprobs,
                "request_id": getattr(attempt, "request_id", ""),
                "sampling_seed": getattr(attempt, "sampling_seed", 0),
                "generation_backend": getattr(attempt, "generation_backend", "unknown"),
                "adapter_sha256": getattr(attempt, "adapter_sha256", ""),
                "rollout_adapter_sha256": getattr(
                    attempt, "rollout_adapter_sha256", ""
                ),
                "adapter_semantic_sha256": getattr(
                    attempt, "adapter_semantic_sha256", ""
                ),
                "queue_wait_ms": getattr(attempt, "queue_wait_ms", 0.0),
                "first_token_latency_ms": getattr(
                    attempt, "first_token_latency_ms", 0.0
                ),
                "generation_time_ms": getattr(attempt, "generation_time_ms", 0.0),
                "output_tokens": attempt.output_tokens,
                "latency_ms": attempt.latency_ms,
                "strict_json_success": attempt.strict_json_success,
                "fallback_used": attempt.fallback_used,
                "schema_valid": attempt.schema_valid,
                "action": attempt.action.to_dict() if attempt.action else None,
                "errors": list(attempt.errors),
                "action_result": None,
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
            }
            turns.append(turn)
            if turn_generated_callback is not None:
                # Charge and fsync generated tokens before browser execution.
                # A callback failure invalidates the rollout because the
                # action can no longer satisfy the durable evidence contract.
                turn_generated_callback(copy.deepcopy(turn))

            if any(
                error.startswith(INFRASTRUCTURE_ERROR_PREFIXES)
                for error in attempt.errors
            ):
                result.update(
                    reward=None,
                    task_score=None,
                    rollout_valid=False,
                    failure_origin="infrastructure",
                    termination_reason="model_backend_error",
                )
                if turn_completed_callback is not None:
                    turn_completed_callback(copy.deepcopy(turn))
                break

            if not attempt.schema_valid or attempt.action is None:
                consecutive_output_failures += 1
                # Invalid model output is a model turn, but it is not an
                # executed action and must not enter action-result history.
                if consecutive_output_failures >= 3:
                    result["termination_reason"] = "model_output_failure_limit"
                    if turn_completed_callback is not None:
                        turn_completed_callback(copy.deepcopy(turn))
                    break
                if turn_completed_callback is not None:
                    turn_completed_callback(copy.deepcopy(turn))
                continue

            consecutive_output_failures = 0
            if environment_steps >= max_env_steps:
                result["termination_reason"] = "max_environment_steps"
                if turn_completed_callback is not None:
                    turn_completed_callback(copy.deepcopy(turn))
                break

            step_result = env.step(attempt.action)
            environment_steps += 1
            action_result = step_result.info.get("action_result", {})
            next_page_type = (
                step_result.observation.page_type
                if step_result.observation is not None
                else "unknown"
            )
            agent.record_feedback(attempt, step_result, next_page_type)

            turn["action_result"] = action_result
            turn["reward"] = step_result.reward
            turn["terminated"] = step_result.terminated
            turn["truncated"] = step_result.truncated
            if turn_completed_callback is not None:
                turn_completed_callback(copy.deepcopy(turn))

            if step_result.observation is not None:
                observation = step_result.observation

            if step_result.terminated or step_result.truncated:
                result["success"] = step_result.reward > 0.5
                result["reward"] = float(step_result.reward)
                result["task_score"] = float(step_result.info.get("task_score", step_result.reward))
                result["failure_origin"] = "none" if result["success"] else "policy"
                result["termination_reason"] = step_result.info.get(
                    "termination_reason",
                    "terminal",
                )
                break
        else:
            result["termination_reason"] = "max_model_turns"

        if result["termination_reason"] == "unknown":
            result["termination_reason"] = "max_model_turns"

    except Exception as exc:
        result.update(
            success=False,
            reward=None,
            task_score=None,
            rollout_valid=False,
            failure_origin="infrastructure",
            termination_reason="environment_or_runner_error",
            error=f"{type(exc).__name__}: {exc}"[:500],
        )

    result["model_turns"] = agent.model_turn
    result["environment_steps"] = environment_steps
    result["elapsed_s"] = time.time() - started

    if env.trajectory is not None:
        verification = env.trajectory.verification or {}
        result["failure_reasons"] = verification.get("failure_reasons", [])
        result["verifier_success"] = verification.get("success", False)
    else:
        result["failure_reasons"] = []
        result["verifier_success"] = False

    return result


def save_model_trajectory(
    task_result: dict,
    output_dir: Path,
    model_info: dict,
    prompt_info: dict,
):
    """Save a versioned model trajectory without changing failure semantics."""
    trajectory = {
        "model_trajectory_schema_version": "2.0",
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
            "rollout_valid": task_result.get("rollout_valid", True),
            "failure_origin": task_result.get("failure_origin", "policy"),
            "termination_reason": task_result["termination_reason"],
            "failure_reasons": task_result.get("failure_reasons", []),
            "model_turns": task_result["model_turns"],
            "environment_steps": task_result["environment_steps"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task_result['task_id']}.json"
    path.write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
