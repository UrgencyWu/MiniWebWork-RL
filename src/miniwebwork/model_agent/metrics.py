"""Compute aggregate metrics from model agent evaluation results."""

import json


def compute_metrics(task_results: list, trajectories: list = None) -> dict:
    """Compute comprehensive metrics from task results."""
    total = len(task_results)
    if total == 0:
        return {"total_tasks": 0}

    successful = [r for r in task_results if r.get("success")]
    failed = [r for r in task_results if not r.get("success")]
    truncated = [r for r in task_results if r.get("termination_reason") == "truncated"]

    # Count model turns
    model_turns = [r.get("model_turns", 0) for r in task_results]
    env_steps = [r.get("environment_steps", 0) for r in task_results]
    sorted_env = sorted(env_steps)

    # Parse stats
    all_turns = []
    for r in task_results:
        for t in r.get("turns", []):
            all_turns.append(t)

    total_gens = len(all_turns)
    nonempty = [t for t in all_turns if t.get("raw_output", "").strip()]
    strict_json = [t for t in all_turns if t.get("strict_json_success")]
    fallback = [t for t in all_turns if t.get("fallback_used")]
    schema_ok = [t for t in all_turns if t.get("schema_valid")]
    has_action = [t for t in all_turns if t.get("action")]

    # Action distribution
    action_dist = {}
    for t in all_turns:
        a = t.get("action", {})
        if a:
            act_type = a.get("action", "unknown")
            action_dist[act_type] = action_dist.get(act_type, 0) + 1

    # Token stats
    input_tokens = [t.get("input_tokens", 0) for t in all_turns]
    output_tokens = [t.get("output_tokens", 0) for t in all_turns]
    latencies = [t.get("latency_ms", 0) for t in all_turns]

    # Task type breakdown
    task_type_results = {}
    for r in task_results:
        tt = r.get("task_type", "unknown")
        if tt not in task_type_results:
            task_type_results[tt] = {"total": 0, "success": 0}
        task_type_results[tt]["total"] += 1
        if r.get("success"):
            task_type_results[tt]["success"] += 1

    # Termination breakdown
    term_reasons = {}
    for r in task_results:
        reason = r.get("termination_reason", "unknown")
        term_reasons[reason] = term_reasons.get(reason, 0) + 1

    return {
        # Task metrics
        "total_tasks": total,
        "successful_tasks": len(successful),
        "success_rate": len(successful) / total,
        "failed_tasks": len(failed),
        "truncated_tasks": len(truncated),
        "average_model_turns": sum(model_turns) / max(total, 1),
        "average_environment_steps": sum(env_steps) / max(total, 1),
        "median_environment_steps": sorted_env[len(sorted_env) // 2] if sorted_env else 0,
        "total_reward": sum(r.get("reward", 0) for r in task_results),

        # Output format metrics
        "total_generations": total_gens,
        "nonempty_generation_rate": len(nonempty) / max(total_gens, 1),
        "strict_json_success_rate": len(strict_json) / max(total_gens, 1),
        "fallback_parse_rate": len(fallback) / max(total_gens, 1),
        "effective_json_success_rate": (len(strict_json) + len([t for t in fallback if t.get("parsed_payload")])) / max(total_gens, 1),
        "action_schema_valid_rate": len(schema_ok) / max(total_gens, 1),
        "target_valid_rate": len(has_action) / max(total_gens, 1),

        # Behavior metrics
        "premature_finish_count": sum(1 for r in task_results if r.get("termination_reason") == "premature_finish"),
        "model_output_failure_limit_count": sum(1 for r in task_results if r.get("termination_reason") == "model_output_failure_limit"),
        "no_submission_count": sum(1 for r in task_results if r.get("termination_reason") == "max_model_turns"),

        # Action distribution
        "action_distribution": action_dist,

        # Token efficiency
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "mean_input_tokens_per_turn": sum(input_tokens) / max(total_gens, 1),
        "mean_output_tokens_per_turn": sum(output_tokens) / max(total_gens, 1),
        "mean_generation_latency_ms": sum(latencies) / max(total_gens, 1),

        # Task type breakdown
        "task_type_breakdown": task_type_results,

        # Termination breakdown
        "termination_reason_breakdown": term_reasons,

        # Per task
        "per_task": [{
            "task_id": r["task_id"],
            "success": r.get("success", False),
            "reward": r.get("reward", 0),
            "model_turns": r.get("model_turns", 0),
            "environment_steps": r.get("environment_steps", 0),
            "termination_reason": r.get("termination_reason", ""),
            "failure_reasons": r.get("failure_reasons", []),
            "elapsed_s": r.get("elapsed_s", 0),
        } for r in task_results],
    }
