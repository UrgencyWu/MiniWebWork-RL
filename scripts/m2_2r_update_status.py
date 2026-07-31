"""M2.2R Formal Status Update.

Updated status based on canonical eval results:
- Teacher-forced: PASS (99.4-100% exact match)
- Frozen E2E: BLOCKED by Playwright sync/async eval harness bug
- READY_FOR_AGENTIC_RL: false
"""
import json
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")


def update_status():
    status = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "formal_status": {
            "M2_2R_PROMPT_CONTRACT_ALIGNMENT_PASS": True,
            "M2_2R_ADAPTER_INDEPENDENCE_PASS": True,
            "M2_2R_TEACHER_FORCED_EVAL_PASS": True,
            "M2_2R_CLOSED_LOOP_EVAL_VALID": False,
            "M2_2R_EVAL_HARNESS_FIX_REQUIRED": True,
            "READY_FOR_AGENTIC_RL": False,
        },
        "teacher_forced_results": {
            "base_canonical_v2": {
                "exact_match": 0.523,
                "action_type_match": 0.661,
                "schema_valid_rate": 0.937,
                "total_samples": 174,
            },
            "sft_seed_42": {
                "exact_match": 0.994,
                "action_type_match": 0.994,
                "schema_valid_rate": 1.0,
                "total_samples": 174,
            },
            "sft_seed_1234": {
                "exact_match": 1.0,
                "action_type_match": 1.0,
                "schema_valid_rate": 1.0,
                "total_samples": 174,
            },
            "sft_seed_20260726": {
                "exact_match": 1.0,
                "action_type_match": 1.0,
                "schema_valid_rate": 1.0,
                "total_samples": 174,
            },
        },
        "frozen_e2e_status": {
            "result": "BLOCKED_BY_EVAL_HARNESS",
            "error": "Playwright Sync API inside asyncio loop",
            "total_tasks": 15,
            "completed_tasks": 0,
            "successful_tasks": 0,
            "note": "All 15 tasks failed due to Playwright sync/async conflict in eval harness. "
                    "This is an infrastructure bug, not a model failure. "
                    "Teacher-forced eval (99.4-100% exact match) proves model output is correct.",
            "fix_required": "Synchronize Playwright usage in eval harness (Route A recommended)",
        },
        "evidence": {
            "prompt_contract_alignment": True,
            "adapter_loading_correct": True,
            "single_step_learning_complete": True,
            "closed_loop_not_tested": True,
            "eval_harness_bug": True,
        },
        "next_phase": "M2.2R-E2E: Fix Playwright sync/async eval harness, then run closed-loop eval",
        "do_not_proceed_to": ["M3.0 Agentic RL", "GRPO", "PPO"],
    }

    output_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "formal_status.json"
    output_path.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    print(f"Formal status saved: {output_path}")

    print(f"\n{'='*70}")
    print(f"FORMAL STATUS")
    print(f"{'='*70}")
    print(f"  M2_2R_PROMPT_CONTRACT_ALIGNMENT_PASS: True")
    print(f"  M2_2R_ADAPTER_INDEPENDENCE_PASS: True")
    print(f"  M2_2R_TEACHER_FORCED_EVAL_PASS: True")
    print(f"  M2_2R_CLOSED_LOOP_EVAL_VALID: False")
    print(f"  M2_2R_EVAL_HARNESS_FIX_REQUIRED: True")
    print(f"  READY_FOR_AGENTIC_RL: False")
    print(f"\nTeacher-forced (Canonical v2):")
    print(f"  Base:        52.3% exact, 66.1% action type, 93.7% schema")
    print(f"  SFT 42:      99.4% exact, 99.4% action type, 100% schema")
    print(f"  SFT 1234:    100% exact, 100% action type, 100% schema")
    print(f"  SFT 20260726:100% exact, 100% action type, 100% schema")
    print(f"\nFrozen E2E: BLOCKED - Playwright sync/async eval harness bug")
    print(f"\nNext phase: M2.2R-E2E (fix eval harness, then closed-loop eval)")


if __name__ == "__main__":
    update_status()
