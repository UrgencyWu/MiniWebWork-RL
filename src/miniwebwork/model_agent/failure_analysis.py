"""Failure analysis for canonical browser-Agent evaluation results."""

from __future__ import annotations


OUTPUT_CODES = {
    "empty_generation",
    "non_json_output",
    "multiple_json_objects",
    "malformed_json",
    "schema_invalid",
    "unknown_action",
    "extra_fields",
    "missing_target",
}
ACTION_CODES = {
    "invalid_target",
    "stale_target",
    "incompatible_action",
    "disabled_element",
    "value_required",
    "value_too_long",
}
INFRASTRUCTURE_TERMINATIONS = {
    "model_backend_error",
    "environment_or_runner_error",
    "environment_step_error",
    "environment_cleanup_error",
    "model_load_error",
    "task_source_error",
}


def classify_failures(task_results: list[dict]) -> dict:
    """Classify policy failures without folding infrastructure into them."""
    analyzed: list[dict] = []

    for result in task_results:
        task_id = result.get("task_id", "unknown")
        if not result.get("rollout_valid", True):
            analyzed.append(
                {
                    "task_id": task_id,
                    "success": False,
                    "rollout_valid": False,
                    "failure_origin": "infrastructure",
                    "tags": ["infrastructure_failure"],
                    "primary_failure": "infrastructure_failure",
                    "termination_reason": result.get("termination_reason", "unknown"),
                    "failure_reasons": result.get("failure_reasons", []),
                    "model_turns": result.get("model_turns", 0),
                    "environment_steps": result.get("environment_steps", 0),
                    "error": result.get("error", ""),
                }
            )
            continue

        if result.get("success"):
            analyzed.append(
                {
                    "task_id": task_id,
                    "success": True,
                    "rollout_valid": True,
                    "failure_origin": "none",
                    "tags": [],
                    "primary_failure": None,
                }
            )
            continue

        turns = result.get("turns", [])
        termination = result.get("termination_reason", "")
        verifier_failures = result.get("failure_reasons", [])
        tags: set[str] = set()

        for turn in turns:
            errors = set(turn.get("errors", []))
            if not turn.get("strict_json_success"):
                tags.add("non_json_output")
            if not turn.get("schema_valid"):
                tags.add("schema_invalid")
            tags.update(error for error in errors if error in OUTPUT_CODES)

            action_result = turn.get("action_result")
            if isinstance(action_result, dict) and not action_result.get("success", False):
                error_code = action_result.get("error_code", "")
                if error_code in ACTION_CODES:
                    tags.add(error_code)

        if termination == "premature_finish":
            tags.add("premature_finish")
        if termination == "model_output_failure_limit":
            tags.add("model_output_failure_limit")
        if termination in {"max_model_turns", "max_environment_steps"}:
            tags.add("no_submission")
        if _has_repeated_action(turns):
            tags.add("repeated_action")

        for failure in verifier_failures:
            if failure in {"wrong_product", "wrong_decision_type"}:
                tags.add("wrong_product")
            elif failure == "objective_not_optimal":
                tags.add("objective_not_optimal")
            elif failure in {"false_no_solution", "expected_no_solution"}:
                tags.add(failure)
            elif failure == "missing_submission":
                tags.add("no_submission")
            elif "constraint" in failure or failure in {
                "out_of_stock",
                "supplier_rating_failed",
                "supplier_certification_failed",
                "region_constraint_failed",
                "warranty_constraint_failed",
            }:
                tags.add("constraint_failure")

        analyzed.append(
            {
                "task_id": task_id,
                "success": False,
                "rollout_valid": True,
                "failure_origin": "policy",
                "tags": sorted(tags),
                "primary_failure": _primary_failure(tags, termination, turns),
                "termination_reason": termination,
                "failure_reasons": verifier_failures,
                "model_turns": result.get("model_turns", 0),
                "environment_steps": result.get("environment_steps", 0),
            }
        )

    return {
        "analyzed_tasks": analyzed,
        "summary": _summarize(analyzed),
    }


def _has_repeated_action(turns: list[dict], window: int = 3) -> bool:
    actions = [turn.get("action") for turn in turns if turn.get("action")]
    if len(actions) < window:
        return False
    for start in range(len(actions) - window + 1):
        if all(action == actions[start] for action in actions[start : start + window]):
            return True
    return False


def _primary_failure(tags: set[str], termination: str, turns: list[dict]) -> str:
    if termination in INFRASTRUCTURE_TERMINATIONS:
        return "infrastructure_failure"
    if not turns:
        return "no_turns_recorded"
    if termination == "model_output_failure_limit":
        return "consecutive_output_failures"
    if tags.intersection(OUTPUT_CODES | {"schema_invalid"}):
        return "output_format_failure"
    if tags.intersection(ACTION_CODES):
        return "element_grounding_failure"
    if "premature_finish" in tags:
        return "premature_finish"
    if "no_submission" in tags:
        return "no_submission_reached"
    if "false_no_solution" in tags:
        return "false_no_solution"
    if "expected_no_solution" in tags:
        return "missed_no_solution"
    if tags.intersection({"wrong_product", "objective_not_optimal"}):
        return "incorrect_product_selection"
    if "constraint_failure" in tags:
        return "constraint_violation"
    if "repeated_action" in tags:
        return "repeated_action"
    return "other_policy_failure"


def _summarize(analyzed: list[dict]) -> dict:
    primary_counts: dict[str, int] = {}
    policy_failures = infrastructure_failures = successes = 0
    for item in analyzed:
        if item.get("success"):
            successes += 1
            continue
        if item.get("failure_origin") == "infrastructure":
            infrastructure_failures += 1
        else:
            policy_failures += 1
        primary = item.get("primary_failure") or "unknown"
        primary_counts[primary] = primary_counts.get(primary, 0) + 1
    return {
        "successes": successes,
        "policy_failures": policy_failures,
        "infrastructure_failures": infrastructure_failures,
        "primary_failure_counts": primary_counts,
    }
