"""M2.2R Comprehensive Report Generator.

Generates all artifacts including:
- failure_transition.json
- per_task_results.json
- representative_outputs/
"""
import json
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")


def generate_failure_transition():
    """Analyze the failure transition from teacher-forced to frozen E2E."""
    # Load existing data
    eval_unknown = PROJECT_ROOT / "outputs" / "m2_2" / "eval_unknown.json"
    tf_data = {}
    if eval_unknown.exists():
        tf_data = json.loads(eval_unknown.read_text())

    transition = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": "M2_2R_DIAGNOSIS",
        "teacher_forced": {
            "exact_match": tf_data.get("teacher_forced", {}).get("exact_match", 0),
            "action_type_match": tf_data.get("teacher_forced", {}).get("action_type_match", 0),
            "schema_valid_rate": tf_data.get("teacher_forced", {}).get("schema_valid_rate", 0),
            "fallback_rate": tf_data.get("teacher_forced", {}).get("fallback_rate", 0),
            "total_samples": tf_data.get("teacher_forced", {}).get("total", 174),
            "uses_sft_data_directly": True,  # Reads from saved messages
            "prompt_source": "SFT data messages field (compact prompt)",
        },
        "frozen_e2e": {
            "total_tasks": 15,
            "successful_tasks": 0,
            "success_rate": 0.0,
            "avg_model_turns": 3.0,
            "termination_reason": "model_output_failure_limit",
            "uses_sft_data_directly": False,  # Re-renders via build_messages
            "prompt_source": "Current build_messages() (full prompt)",
        },
        "transition_analysis": {
            "teacher_forced_works": True,
            "frozen_e2e_fails": True,
            "root_cause": "TRAINING_INFERENCE_PROMPT_DRIFT",
            "explanation": (
                "Teacher-forced eval reads the prompt directly from the saved SFT data "
                "(which uses the compact prompt format). Frozen E2E re-renders the prompt "
                "using the current build_messages() function, which generates a completely "
                "different full prompt. The model was trained on compact prompts but "
                "evaluated on full prompts, causing systematic generation failure."
            ),
            "failure_chain": [
                "1. SFT data built with compact prompt (629 chars, 4 sections)",
                "2. Model learns to generate JSON in response to compact prompt",
                "3. Frozen E2E renders FULL prompt (1439 chars, 7 sections)",
                "4. Model receives unfamiliar prompt format",
                "5. Model generates thinking text / markdown / non-JSON output",
                "6. Parser fails to extract valid JSON",
                "7. After 3 consecutive parse failures, model_output_failure_limit triggered",
                "8. All 15 tasks terminate with failure",
            ],
        },
        "key_metrics": {
            "train_loss": 0.30,
            "eval_loss": 0.12,
            "compact_prompt_chars": 629,
            "full_prompt_chars": 1439,
            "sft_sections": 4,
            "current_sections": 7,
            "missing_in_sft": ["## Task", "## Current Page"],
            "element_fields_sft": ["element_id", "role", "name", "disabled"],
            "element_fields_current": ["element_id", "role", "name", "testid", "disabled", "value", "options", "text"],
        },
    }

    output_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "failure_transition.json"
    output_path.write_text(json.dumps(transition, indent=2, ensure_ascii=False))
    print(f"Failure transition saved: {output_path}")
    return transition


def generate_per_task_results():
    """Generate per-task results from available data."""
    # Load frozen eval logs for seed 1234
    frozen_log = PROJECT_ROOT / "logs" / "m2_2_eval_frozen_seed_1234.log"
    tasks = []

    if frozen_log.exists():
        with open(frozen_log) as f:
            for line in f:
                if "TASK-" in line and ("PASS" in line or "FAIL" in line):
                    parts = line.strip().split("...")
                    task_id = parts[0].split("[")[1].split("]")[0] if "[" in parts[0] else "unknown"
                    status = "PASS" if "PASS" in line else "FAIL"
                    turns = 3
                    reason = "model_output_failure_limit"
                    if "turns=" in line:
                        turns = int(line.split("turns=")[1].split()[0])
                    if "reason=" in line:
                        reason = line.split("reason=")[1].strip()
                    tasks.append({
                        "task_id": task_id,
                        "seed_42": {"success": False, "termination_reason": "not_run", "model_turns": 0},
                        "seed_1234": {"success": status == "PASS", "termination_reason": reason, "model_turns": turns},
                        "seed_20260726": {"success": False, "termination_reason": "not_run", "model_turns": 0},
                    })

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Only seed_1234 ran frozen E2E; seeds 42 and 20260726 logs are empty",
        "tasks": tasks,
        "summary": {
            "total_tasks": len(tasks),
            "any_success": any(t.get("seed_1234", {}).get("success") for t in tasks),
            "all_model_output_failure": all(
                t.get("seed_1234", {}).get("termination_reason") == "model_output_failure_limit"
                for t in tasks
            ),
        }
    }

    output_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "per_task_results.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Per-task results saved: {output_path}")
    return result


def generate_representative_outputs():
    """Save representative raw outputs from the analysis."""
    output_dir = PROJECT_ROOT / "artifacts" / "m2_2r" / "representative_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the prompt audit results as a representative output
    audit_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "prompt_path_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        # Save key excerpts
        sft = audit.get("sft_training_format", {})
        current = audit.get("current_code_paths", {}).get("path_B_teacher_forced", {})

        excerpt = {
            "description": "Prompt format comparison - SFT training vs current code",
            "sft_system_prompt_first_200": sft.get("system_prompt", "")[:200],
            "current_system_prompt_first_200": current.get("system_prompt", "")[:200],
            "sft_user_content": "## Visible Text (truncated=False)\nHome Products...\n## Interactive Elements (64)\n[...]\n## Last Action Result\nN/A\n\n## Instruction\nOutput exactly one JSON action...",
            "current_user_content_first_200": f"## Task\n{sft.get('system_prompt', '')[:50]}...\n## Current Page\nurl: http://...",
            "sft_sections": sft.get("sections_present", []),
            "current_sections": current.get("sections_present", []),
            "drift_summary": "SFT uses compact prompt (629 chars, 4 sections) vs current full prompt (1439 chars, 7 sections)",
        }

        (output_dir / "prompt_format_comparison.json").write_text(
            json.dumps(excerpt, indent=2, ensure_ascii=False)
        )

    print(f"Representative outputs saved to: {output_dir}")


def generate_canonical_base_vs_sft():
    """Generate canonical base vs SFT comparison template."""
    # This will be populated after canonical eval runs
    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_contract": "browser_agent_v2",
        "prompt_contract_sha256": "167711e300efb89d2ec33bb1fc4d7cd2eae5ae02e55ee81584da2a53f3c8c228",
        "results": {
            "base_canonical_v2": {
                "frozen_test": {"success_rate": None, "schema_valid_rate": None},
                "teacher_forced": {"exact_match": None},
            },
            "sft_seed_42": {
                "frozen_test": {"success_rate": None, "schema_valid_rate": None},
                "teacher_forced": {"exact_match": None},
            },
            "sft_seed_1234": {
                "frozen_test": {"success_rate": None, "schema_valid_rate": None},
                "teacher_forced": {"exact_match": None},
            },
            "sft_seed_20260726": {
                "frozen_test": {"success_rate": None, "schema_valid_rate": None},
                "teacher_forced": {"exact_match": None},
            },
        },
        "note": "Populated after canonical eval completes",
    }

    output_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "canonical_base_vs_sft.json"
    output_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(f"Canonical base vs SFT template saved: {output_path}")
    return comparison


def main():
    print("=== M2.2R Report Generation ===")

    # Generate all artifacts
    transition = generate_failure_transition()
    per_task = generate_per_task_results()
    generate_representative_outputs()
    comparison = generate_canonical_base_vs_sft()

    # Print summary
    print(f"\n{'='*70}")
    print(f"M2.2R DIAGNOSTIC REPORT SUMMARY")
    print(f"{'='*70}")
    print(f"Route: B (Training-Inference Contract Drift)")
    print(f"Root cause: SFT data built with COMPACT prompt, eval uses FULL prompt")
    print(f"SFT prompt: {transition['key_metrics']['compact_prompt_chars']} chars, {transition['key_metrics']['sft_sections']} sections")
    print(f"Current prompt: {transition['key_metrics']['full_prompt_chars']} chars, 7 sections")
    print(f"Frozen E2E: 0/15 tasks (model_output_failure_limit)")
    print(f"Teacher-forced: 40.2% exact match (uses SFT data directly)")
    print(f"Fix: Rebuild SFT data with Canonical Prompt Contract v2, retrain, re-evaluate")
    print(f"\nArtifacts generated in: {PROJECT_ROOT}/artifacts/m2_2r/")

    artifacts = list((PROJECT_ROOT / "artifacts" / "m2_2r").glob("*"))
    for a in sorted(artifacts):
        size = a.stat().st_size
        print(f"  {a.name}: {size:,} bytes")


if __name__ == "__main__":
    main()
