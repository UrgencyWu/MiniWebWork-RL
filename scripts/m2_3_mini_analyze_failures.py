#!/usr/bin/env python3
"""M2.3-mini: Analyze Phase 3 failure trajectories and classify failure modes.

Usage:
    python scripts/m2_3_mini_analyze_failures.py
    python scripts/m2_3_mini_analyze_failures.py --task-id TASK-004
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def classify_trajectory(traj: dict) -> dict:
    """Classify the first deviation point in a trajectory."""
    turns = traj.get("turns", [])
    pages_visited = []
    actions = []
    first_failure_stage = "unknown"
    first_failure_turn = -1
    details = ""

    for i, turn in enumerate(turns):
        obs = turn.get("observation") or {}
        page_type = obs.get("page_type", "unknown")
        pages_visited.append(page_type)

        action = turn.get("action")
        if action:
            act_type = action.get("action", "unknown")
            act_target = action.get("target", "")
            actions.append(f"turn{i}: {act_type}({act_target[:40]})")
        else:
            actions.append(f"turn{i}: None (schema_invalid)")

        # Check for parse/schema failure
        if not turn.get("schema_valid") or action is None:
            if first_failure_turn < 0:
                first_failure_stage = "output_format_failure"
                first_failure_turn = i
                details = f"schema_valid={turn.get('schema_valid')}, errors={turn.get('errors', [])}"
            continue

        # Check action result
        action_result = turn.get("action_result") or {}
        action_success = action_result.get("success", False)

        if not action_success and first_failure_turn < 0:
            first_failure_stage = "action_execution_failure"
            first_failure_turn = i
            details = f"action={action.get('action')} target={action.get('target', '')[:50]}"

        act_type = action.get("action", "") if action else ""

        # Detect no-solution selection
        if page_type == "products" and act_type == "click":
            target = (action.get("target") or "").lower()
            if "no-solution" in target or "no_solution" in target or "declar" in target:
                if first_failure_turn < 0:
                    first_failure_stage = "no_solution_selection"
                    first_failure_turn = i
                    details = f"clicked no-solution button"

        # Detect form completion
        if page_type == "procurement_form":
            if first_failure_turn < 0:
                first_failure_stage = "form_reached"
                first_failure_turn = i
                details = f"reached procurement form"

        # Detect premature finish
        if act_type == "finish" and page_type != "procurement_result":
            if first_failure_turn < 0:
                first_failure_stage = "premature_finish"
                first_failure_turn = i
                details = f"finished on {page_type}"

    # Classify based on termination reason if no earlier failure found
    term_reason = traj.get("termination_reason", "")
    if first_failure_turn < 0:
        if term_reason == "model_output_failure_limit":
            first_failure_stage = "output_format_failure"
        elif term_reason == "max_model_turns":
            first_failure_stage = "max_turns_reached"
        elif "products" in str(pages_visited):
            first_failure_stage = "empty_result_loop"
        else:
            first_failure_stage = "unknown_failure"

    return {
        "first_failure_stage": first_failure_stage,
        "first_failure_turn": first_failure_turn,
        "details": details,
        "actions": actions,
        "pages_visited": pages_visited,
        "termination_reason": term_reason,
        "model_turns": traj.get("model_turns", 0),
        "success": traj.get("success", False),
        "reward": traj.get("reward", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="M2.3-mini: Analyze failure trajectories")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "outputs/m3_0a/phase3_rollout_probe.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/m2_3_mini")
    parser.add_argument("--task-id", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    all_trajs = []
    for g in data.get("groups", []):
        for t in g.get("trajectories", []):
            if args.task_id and t.get("task_id") != args.task_id:
                continue
            all_trajs.append(t)

    print(f"Analyzing {len(all_trajs)} trajectories...")

    classifications = []
    stage_counts = defaultdict(int)
    for t in all_trajs:
        c = classify_trajectory(t)
        classifications.append(c)
        stage_counts[c["first_failure_stage"]] += 1

    print("\n=== Failure Stage Distribution ===")
    for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"  {stage:40s}: {count:3d} ({count/len(classifications)*100:.1f}%)")

    print("\n=== Detailed Examples ===")
    seen_stages = set()
    for c in classifications:
        stage = c["first_failure_stage"]
        if stage not in seen_stages:
            seen_stages.add(stage)
            print(f"\n--- {stage} ---")
            print(f"  Turn {c['first_failure_turn']}: {c['details']}")
            print(f"  Pages: {' -> '.join(c['pages_visited'][:6])}")
            print(f"  Actions:")
            for a in c["actions"][:8]:
                print(f"    {a}")

    result = {
        "input_file": str(args.input),
        "total_trajectories": len(all_trajs),
        "stage_distribution": dict(stage_counts),
        "classifications": classifications,
    }

    out_path = args.output_dir / "failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    print("\n=== Markdown Table ===")
    print("| 首个失败位置 | 数量 |")
    print("|------------|-----:|")
    for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"| {stage} | {count} |")


if __name__ == "__main__":
    main()
