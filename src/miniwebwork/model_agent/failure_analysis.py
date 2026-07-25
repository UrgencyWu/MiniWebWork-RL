"""Taxonomy-based failure analysis from model agent results."""

FAILURE_CATEGORIES = {
    "output": ["empty_generation", "non_json_output", "multiple_json_objects", "malformed_json",
               "schema_invalid", "unknown_action", "extra_fields"],
    "action": ["invalid_target", "stale_target", "incompatible_action", "disabled_element",
               "value_too_long", "environment_action_error"],
    "planning": ["repeated_action", "navigation_loop", "premature_finish", "no_submission",
                 "wrong_page", "ignored_task_constraint", "failed_to_apply_filter",
                 "failed_to_compare_candidates"],
    "terminal": ["wrong_product", "objective_not_optimal", "false_no_solution",
                 "expected_no_solution", "constraint_failure", "max_model_turns",
                 "max_environment_steps", "browser_error", "model_error"],
}


def classify_failures(task_results: list) -> dict:
    """Classify each failed task into taxonomy categories."""
    analyzed = []

    for r in task_results:
        if r.get("success"):
            analyzed.append({"task_id": r["task_id"], "success": True, "tags": [], "primary_failure": None})
            continue

        tags = set()
        turns = r.get("turns", [])
        term_reason = r.get("termination_reason", "")
        failure_reasons = r.get("failure_reasons", [])

        # Output failures
        parse_fails = [t for t in turns if not t.get("strict_json_success")]
        if parse_fails:
            tags.add("non_json_output")
        schema_fails = [t for t in turns if t.get("parsed_payload") and not t.get("schema_valid")]
        if schema_fails:
            tags.add("schema_invalid")

        # Action failures
        for t in turns:
            ar = t.get("action_result", {})
            if ar and not ar.get("success"):
                ec = ar.get("error_code", "")
                if ec == "invalid_target":
                    tags.add("invalid_target")
                elif ec == "stale_target":
                    tags.add("stale_target")
                elif ec == "incompatible_action":
                    tags.add("incompatible_action")
                elif ec == "disabled_element":
                    tags.add("disabled_element")

        # Planning failures
        if term_reason == "premature_finish":
            tags.add("premature_finish")
        if term_reason in ("max_model_turns", "max_environment_steps"):
            tags.add("no_submission")
        if len(turns) >= 3 and all(t.get("action", {}).get("action") == turns[0].get("action", {}).get("action") for t in turns[:3]):
            tags.add("repeated_action")

        # Terminal failures
        for fr in failure_reasons:
            if fr in ("wrong_product", "wrong_decision_type"):
                tags.add("wrong_product")
            elif fr == "objective_not_optimal":
                tags.add("objective_not_optimal")
            elif fr in ("false_no_solution", "expected_no_solution"):
                tags.add("false_no_solution" if fr == "false_no_solution" else "expected_no_solution")
            elif "constraint" in fr:
                tags.add("constraint_failure")
            elif fr == "missing_submission":
                tags.add("no_submission")

        # Determine primary failure
        primary = _determine_primary(tags, term_reason, turns)

        analyzed.append({
            "task_id": r["task_id"],
            "success": False,
            "tags": sorted(tags),
            "primary_failure": primary,
            "termination_reason": term_reason,
            "failure_reasons": failure_reasons,
            "model_turns": r.get("model_turns", 0),
            "environment_steps": r.get("environment_steps", 0),
        })

    return {
        "analyzed_tasks": analyzed,
        "summary": _summarize(analyzed),
    }


def _determine_primary(tags: set, term_reason: str, turns: list) -> str:
    """Determine the primary failure reason."""
    # Hierarchy: output > action > planning > terminal
    if not turns:
        return "no_turns_recorded"
    if "non_json_output" in tags or "schema_invalid" in tags:
        return "output_format_failure"
    if any(t in tags for t in ["invalid_target", "stale_target", "incompatible_action"]):
        return "element_grounding_failure"
    if term_reason == "model_output_failure_limit":
        return "consecutive_output_failures"
    if "no_submission" in tags:
        return "no_submission_reached"
    if "premature_finish" in tags:
        return "premature_finish"
    if "wrong_product" in tags or "objective_not_optimal" in tags:
        return "incorrect_product_selection"
    if "constraint_failure" in tags:
        return "constraint_violation"
    return "other"


def _summarize(analyzed: list) -> dict:
    """Summarize failure distribution."""
    counts = {}
    for a in analyzed:
        pf = a.get("primary_failure", "unknown")
        counts[pf] = counts.get(pf, 0) + 1
    return {"primary_failure_counts": counts}
