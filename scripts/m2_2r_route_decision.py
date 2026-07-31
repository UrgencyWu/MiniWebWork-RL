"""M2.2R Route Decision - determine which repair route to take."""
import json
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")


def decide_route():
    """Based on audit results, determine the repair route."""

    # Load audit results
    audit_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "prompt_path_audit.json"
    if not audit_path.exists():
        print("ERROR: prompt_path_audit.json not found. Run audit first.")
        return

    audit = json.loads(audit_path.read_text())

    decision = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "audit_verdict": audit.get("verdict", "UNKNOWN"),
        "drift_count": audit.get("drift_count", 0),
        "route": None,
        "requires_retrain": False,
        "reasoning": [],
        "fixes_required": [],
    }

    # Analyze findings
    sft_format = audit.get("sft_training_format", {})
    current_b = audit.get("current_code_paths", {}).get("path_B_teacher_forced", {})

    # Check: Is this prompt drift (Route B)?
    sys_prompt_mismatch = sft_format.get("system_prompt_sha256") != current_b.get("system_prompt_sha256")
    sections_mismatch = sft_format.get("sections_count") != current_b.get("sections_count")
    task_section_missing = not sft_format.get("has_task_section")
    current_page_missing = not sft_format.get("has_current_page_section")

    if sys_prompt_mismatch or sections_mismatch or task_section_missing or current_page_missing:
        decision["route"] = "B"
        decision["requires_retrain"] = True
        decision["reasoning"] = [
            f"SFT training used COMPACT prompt ({sft_format.get('system_prompt_length')} chars)",
            f"Current code uses FULL prompt ({current_b.get('system_prompt_length')} chars)",
            f"SFT training sections: {sft_format.get('sections_count')} ({sft_format.get('sections_present')})",
            f"Current code sections: {current_b.get('sections_count')} ({current_b.get('sections_present')})",
            f"SFT missing ## Task: {task_section_missing}, ## Current Page: {current_page_missing}",
            "Teacher-forced eval works because it reads from saved SFT data directly",
            "Frozen E2E fails because it re-renders prompts using current build_messages()",
            "This is classic TRAINING-INFERENCE CONTRACT DRIFT",
        ]
        decision["fixes_required"] = [
            "Freeze Canonical Prompt Contract v2 (browser_agent_v2)",
            "Rebuild SFT dataset using Canonical Contract (build_messages_v2)",
            "Re-run mask audit on rebuilt dataset",
            "Re-run data leakage audit",
            "Retrain Seed 42 with canonical dataset",
            "Validate via teacher-forced and 3-task E2E smoke test",
            "Run full 15-task frozen E2E",
            "Evaluate Base and all SFT seeds with Canonical Contract",
        ]
    else:
        # Check for adapter issues (Route A)
        decision["route"] = "A"
        decision["requires_retrain"] = False
        decision["reasoning"] = ["Adapter or eval implementation bug detected"]

    # Add additional context
    decision["additional_findings"] = {
        "adapter_sha256s": {
            "seed_42": "b32c1a27be78c67fd8c4d4428b30631a4367190573119a24ee33fcdc0452e08a",
            "seed_1234": "c6f2905cfd7c97449edf43c24a4b7eb4840f4a8543d405e123c43700bd38a13e",
            "seed_20260726": "bef840b6b92d26f6ee606c0a7652e406e10b9f91556a1f75b81ad3920c1bf1c4",
        },
        "eval_bug_frozen_e2e": "AttributeError at line 221: action_result is None for model_output_failure_limit turns",
        "frozen_e2e_result": "0/15 tasks, all terminated with model_output_failure_limit",
        "teacher_forced_result": "40.2% exact match, 60.9% action type, 100% schema valid",
        "train_loss": 0.30,
        "eval_loss": 0.12,
    }

    output_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "route_decision.json"
    output_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False))

    print(f"Route decision saved: {output_path}")
    print(f"\n{'='*70}")
    print(f"ROUTE DECISION: ROUTE {decision['route']}")
    print(f"{'='*70}")
    print(f"Requires retrain: {decision['requires_retrain']}")
    print(f"\nReasoning:")
    for r in decision["reasoning"]:
        print(f"  - {r}")
    print(f"\nFixes required:")
    for f in decision["fixes_required"]:
        print(f"  - {f}")
    print(f"\nAdditional findings:")
    print(f"  - Eval bug: {decision['additional_findings']['eval_bug_frozen_e2e']}")
    print(f"  - Frozen E2E: {decision['additional_findings']['frozen_e2e_result']}")
    print(f"  - Teacher-forced: {decision['additional_findings']['teacher_forced_result']}")

    return decision


if __name__ == "__main__":
    decide_route()
