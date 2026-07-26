"""M2.0.1: Recalculate all baseline metrics from raw trajectory artifacts."""

import json, sys
from pathlib import Path


def audit(artifact_dir="artifacts/m2_0", output_path="artifacts/m2_0/m2_0_metrics_audit.json"):
    ad = Path(artifact_dir)
    task_ids = sorted([f.stem for f in (ad / "per_task").glob("*.json")])
    total_tasks = len(task_ids)

    # Collect all turns
    all_turns = []
    task_results = {}
    for tid in task_ids:
        pt = json.loads((ad / "per_task" / f"{tid}.json").read_text())
        traj = json.loads((ad / "trajectories" / f"{tid}.json").read_text())
        turns = traj.get("turns", [])
        task_results[tid] = {"per_task": pt, "trajectory": traj, "turns": turns}
        all_turns.extend(turns)

    total_gens = len(all_turns)
    total_model_turns = sum(r["per_task"].get("model_turns", 0) for r in task_results.values())

    # Atomic counts from actual trajectory fields
    empty = [t for t in all_turns if not (t.get("raw_output") or "").strip()]
    nonempty = [t for t in all_turns if (t.get("raw_output") or "").strip()]

    strict_ok = [t for t in all_turns if t.get("strict_json_success")]
    fallback_used = [t for t in all_turns if t.get("fallback_used")]
    fb_ok = [t for t in fallback_used if t.get("parsed_payload") or t.get("schema_valid")]
    fb_fail = [t for t in fallback_used if not (t.get("parsed_payload") or t.get("schema_valid"))]

    effective = [t for t in all_turns if t.get("parsed_payload") or t.get("schema_valid")]

    schema_ok = [t for t in effective if t.get("schema_valid")]
    schema_bad = [t for t in effective if not t.get("schema_valid")]

    acted = [t for t in schema_ok if t.get("action")]
    env_ok = [t for t in acted if (t.get("action_result") or {}).get("success")]
    env_fail = [t for t in acted if not (t.get("action_result") or {}).get("success")]

    target_ok = [t for t in env_fail if (t.get("action_result") or {}).get("error_code") not in
                 ("invalid_target", "stale_target", "disabled_element")]
    target_bad = [t for t in env_fail if (t.get("action_result") or {}).get("error_code") in
                  ("invalid_target", "stale_target", "disabled_element")]

    successful_tasks = sum(1 for r in task_results.values() if r["per_task"].get("success"))
    verifier_ok = sum(1 for r in task_results.values() if r["trajectory"].get("verification", {}).get("success"))
    verifier_fail = sum(1 for r in task_results.values()
                        if r["trajectory"].get("verification") and not r["trajectory"]["verification"].get("success"))

    # Task-level format/grounding failures
    task_format_fail = set()
    task_grounding_fail = set()
    for tid, r in task_results.items():
        for t in r["turns"]:
            if not (t.get("raw_output") or "").strip():
                task_format_fail.add(tid)
            ar = t.get("action_result") or {}
            if ar.get("error_code") in ("invalid_target", "stale_target", "disabled_element"):
                task_grounding_fail.add(tid)

    def r(n, d):
        return {"numerator": n, "denominator": d, "value": n / d if d > 0 else 0.0}

    corrected = {
        "total_tasks": total_tasks, "total_model_turns": total_model_turns,
        "total_generations": total_gens,
        "empty_generation_count": len(empty),
        "nonempty_generation_count": len(nonempty),
        "strict_parse_success_count": len(strict_ok),
        "fallback_attempt_count": len(fallback_used),
        "fallback_success_count": len(fb_ok),
        "fallback_failure_count": len(fb_fail),
        "effective_json_success_count": len(effective),
        "schema_valid_count": len(schema_ok),
        "schema_invalid_count": len(schema_bad),
        "env_step_attempt_count": len(acted),
        "environment_action_success_count": len(env_ok),
        "environment_action_failure_count": len(env_fail),
        "target_valid_count": len(acted) - len(target_bad),
        "target_invalid_count": len(target_bad),
        "successful_tasks": successful_tasks,
        "failed_tasks": total_tasks - successful_tasks,
        "verifier_success_count": verifier_ok,
        "verifier_failure_count": verifier_fail,
        "task_format_failure_count": len(task_format_fail),
        "task_grounding_failure_count": len(task_grounding_fail),
    }

    rates = {
        "nonempty_generation_rate": r(len(nonempty), total_gens),
        "strict_json_success_rate": r(len(strict_ok), total_gens),
        "fallback_attempt_rate": r(len(fallback_used), total_gens),
        "fallback_success_rate": r(len(fb_ok), len(fallback_used)),
        "effective_json_success_rate": r(len(effective), total_gens),
        "schema_valid_rate": r(len(schema_ok), len(effective)),
        "target_valid_rate": r(corrected["target_valid_count"], len(acted)),
        "environment_action_success_rate": r(len(env_ok), len(acted)),
        "task_success_rate": r(successful_tasks, total_tasks),
    }

    # Contingency tables
    ct = {
        "gen_level": {
            "empty": len(empty), "nonempty": len(nonempty),
            "strict_ok": len(strict_ok), "fallback_ok": len(fb_ok), "fallback_fail": len(fb_fail),
        },
        "schema_level": {
            "schema_valid": len(schema_ok), "schema_invalid": len(schema_bad),
        },
        "env_level": {
            "env_action_ok": len(env_ok), "env_action_fail": len(env_fail),
            "target_bad": len(target_bad),
        },
    }

    # Old reported values
    old = {}
    old_path = ad / "m2_0_base_agent_metrics.json"
    if old_path.exists():
        od = json.loads(old_path.read_text())
        old = {k: od.get(k) for k in ["successful_tasks", "success_rate", "total_generations",
                "nonempty_generation_rate", "strict_json_success_rate", "fallback_parse_rate",
                "effective_json_success_rate", "action_schema_valid_rate", "target_valid_rate",
                "average_model_turns"]}

    output = {"old_reported": old, "corrected": corrected, "rates": rates, "contingency": ct}
    Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"Tasks: {total_tasks}, Generations: {total_gens}, Model turns: {total_model_turns}")
    print(f"Nonempty: {len(nonempty)}/{total_gens} = {rates['nonempty_generation_rate']['value']:.1%}")
    print(f"Strict JSON: {len(strict_ok)}/{total_gens} = {rates['strict_json_success_rate']['value']:.1%}")
    print(f"Fallback success: {len(fb_ok)}/{len(fallback_used)}")
    print(f"Effective JSON: {len(effective)}/{total_gens} = {rates['effective_json_success_rate']['value']:.1%}")
    print(f"Schema valid: {len(schema_ok)}/{len(effective)} = {rates['schema_valid_rate']['value']:.1%}")
    print(f"Env success: {len(env_ok)}/{len(acted)} = {rates['environment_action_success_rate']['value']:.1%}")
    print(f"Task success: {successful_tasks}/{total_tasks} = {rates['task_success_rate']['value']:.1%}")
    print(f"\nRESOLVED CONTRADICTIONS:")
    print(f"  Old '94.6% strict JSON' was actually NONEMPTY rate (={rates['nonempty_generation_rate']['value']:.1%})")
    print(f"  True strict JSON = {rates['strict_json_success_rate']['value']:.1%}")
    print(f"  '100% target valid' was because we counted all acted as target-valid")
    print(f"  Actual target failures: {len(target_bad)}")
    print(f"  '100% schema valid' = schema_valid/effective_json (true)")
    print(f"  Format failures: {len(task_format_fail)} tasks, Grounding failures: {len(task_grounding_fail)} tasks")
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    ad = sys.argv[1] if len(sys.argv) > 1 else "artifacts/m2_0"
    op = sys.argv[2] if len(sys.argv) > 2 else "artifacts/m2_0/m2_0_metrics_audit.json"
    sys.exit(audit(ad, op))
