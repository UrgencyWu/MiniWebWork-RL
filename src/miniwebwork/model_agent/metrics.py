"""Aggregate closed-loop Agent metrics with explicit failure denominators."""

from __future__ import annotations


def _rate(numerator: int | float, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def compute_metrics(task_results: list, trajectories: list | None = None) -> dict:
    """Compute metrics from valid policy rollouts only.

    Infrastructure failures remain visible in separate counts but do not enter
    success, reward, action-format, or task-type denominators.
    """
    requested = len(task_results)
    if requested == 0:
        return {
            "requested_tasks": 0,
            "valid_tasks": 0,
            "infrastructure_errors": 0,
            "total_tasks": 0,
        }

    valid_results = [result for result in task_results if result.get("rollout_valid", True)]
    infrastructure_results = [
        result for result in task_results if not result.get("rollout_valid", True)
    ]
    successful = [result for result in valid_results if result.get("success")]
    failed = [result for result in valid_results if not result.get("success")]
    truncated = [
        result
        for result in valid_results
        if result.get("termination_reason") in {"truncated", "max_environment_steps"}
    ]

    all_turns = [
        turn
        for result in valid_results
        for turn in result.get("turns", [])
    ]
    environment_turns = [
        turn
        for turn in all_turns
        if isinstance(turn.get("action_result"), dict)
        and "success" in turn["action_result"]
    ]

    nonempty = [turn for turn in all_turns if turn.get("raw_output", "").strip()]
    strict_json = [turn for turn in all_turns if turn.get("strict_json_success")]
    fallback = [turn for turn in all_turns if turn.get("fallback_used")]
    effective_json = [
        turn
        for turn in all_turns
        if turn.get("strict_json_success")
        or (turn.get("fallback_used") and turn.get("action") is not None)
    ]
    schema_valid = [turn for turn in all_turns if turn.get("schema_valid")]
    has_action = [turn for turn in all_turns if turn.get("action") is not None]

    action_distribution: dict[str, int] = {}
    for turn in has_action:
        action_type = turn["action"].get("action", "unknown")
        action_distribution[action_type] = action_distribution.get(action_type, 0) + 1

    task_type_breakdown: dict[str, dict] = {}
    for result in valid_results:
        task_type = result.get("task_type", "unknown")
        bucket = task_type_breakdown.setdefault(task_type, {"total": 0, "success": 0})
        bucket["total"] += 1
        bucket["success"] += int(bool(result.get("success")))
    for bucket in task_type_breakdown.values():
        bucket["success_rate"] = _rate(bucket["success"], bucket["total"])

    termination_breakdown: dict[str, int] = {}
    for result in valid_results:
        reason = result.get("termination_reason", "unknown")
        termination_breakdown[reason] = termination_breakdown.get(reason, 0) + 1

    model_turns = [result.get("model_turns", 0) for result in valid_results]
    environment_steps = [result.get("environment_steps", 0) for result in valid_results]
    input_tokens = [turn.get("input_tokens", 0) for turn in all_turns]
    output_tokens = [turn.get("output_tokens", 0) for turn in all_turns]
    latencies = [turn.get("latency_ms", 0.0) for turn in all_turns]
    sorted_environment_steps = sorted(environment_steps)

    metrics = {
        "requested_tasks": requested,
        "valid_tasks": len(valid_results),
        "infrastructure_errors": len(infrastructure_results),
        # Compatibility alias: total_tasks is the valid denominator.
        "total_tasks": len(valid_results),
        "successful_tasks": len(successful),
        "success_rate": _rate(len(successful), len(valid_results)),
        "failed_tasks": len(failed),
        "truncated_tasks": len(truncated),
        "average_model_turns": _rate(sum(model_turns), len(valid_results)),
        "average_environment_steps": _rate(sum(environment_steps), len(valid_results)),
        "median_environment_steps": (
            sorted_environment_steps[len(sorted_environment_steps) // 2]
            if sorted_environment_steps
            else 0
        ),
        "total_reward": sum(float(result.get("reward", 0.0)) for result in valid_results),
        "total_generations": len(all_turns),
        "nonempty_generation_rate": _rate(len(nonempty), len(all_turns)),
        "strict_json_success_rate": _rate(len(strict_json), len(all_turns)),
        "fallback_parse_rate": _rate(len(fallback), len(all_turns)),
        "effective_json_success_rate": _rate(len(effective_json), len(all_turns)),
        "action_schema_valid_rate": _rate(len(schema_valid), len(all_turns)),
        "target_valid_rate": _rate(len(has_action), len(all_turns)),
        "environment_action_success_rate": _rate(
            sum(1 for turn in environment_turns if turn["action_result"].get("success")),
            len(environment_turns),
        ),
        "premature_finish_count": sum(
            1 for result in valid_results
            if result.get("termination_reason") == "premature_finish"
        ),
        "model_output_failure_limit_count": sum(
            1 for result in valid_results
            if result.get("termination_reason") == "model_output_failure_limit"
        ),
        "no_submission_count": sum(
            1 for result in valid_results
            if result.get("termination_reason") in {"max_model_turns", "max_environment_steps"}
        ),
        "action_distribution": action_distribution,
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "mean_input_tokens_per_turn": _rate(sum(input_tokens), len(all_turns)),
        "mean_output_tokens_per_turn": _rate(sum(output_tokens), len(all_turns)),
        "mean_generation_latency_ms": _rate(sum(latencies), len(all_turns)),
        "task_type_breakdown": task_type_breakdown,
        "termination_reason_breakdown": termination_breakdown,
        "infrastructure_failure_breakdown": _termination_counts(infrastructure_results),
        "per_task": [
            {
                "task_id": result["task_id"],
                "success": result.get("success", False),
                "reward": result.get("reward"),
                "rollout_valid": result.get("rollout_valid", True),
                "failure_origin": result.get("failure_origin", "policy"),
                "model_turns": result.get("model_turns", 0),
                "environment_steps": result.get("environment_steps", 0),
                "termination_reason": result.get("termination_reason", ""),
                "failure_reasons": result.get("failure_reasons", []),
                "elapsed_s": result.get("elapsed_s", 0.0),
            }
            for result in task_results
        ],
    }
    return metrics


def _termination_counts(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        reason = result.get("termination_reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts
