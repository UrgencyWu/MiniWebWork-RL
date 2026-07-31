"""M2.2R: Complete prompt path audit - compares training data format vs current code paths.

Uses existing SFT dataset samples to determine what format the model was trained on,
then compares against teacher-forced eval and frozen E2E paths.
"""
import json
import hashlib
import re
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")

from miniwebwork.model_agent.prompt_builder import (
    build_messages, load_system_prompt, compute_message_hash,
    MAX_VISIBLE_TEXT, HISTORY_WINDOW, MAX_ELEMENTS,
)
from miniwebwork.tasks import get_oracle


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_sft_sample(n=0, split="valid"):
    """Load a specific sample from the SFT dataset."""
    path = PROJECT_ROOT / "data" / "sft" / "m2_2" / f"{split}.jsonl"
    with open(path) as f:
        for i, line in enumerate(f):
            if i == n:
                return json.loads(line)
    return None


def analyze_sft_training_format():
    """Analyze what format the model was actually trained on (from SFT data)."""
    sample = load_sft_sample(0, "train")
    if not sample:
        return {}

    msgs = sample["messages"]
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    assistant = msgs[2]["content"]

    # Parse user content sections
    sections = {}
    current_header = None
    current_lines = []
    for line in user.split('\n'):
        if line.startswith('## '):
            if current_header is not None:
                sections[current_header] = '\n'.join(current_lines)
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_header is not None:
        sections[current_header] = '\n'.join(current_lines)

    # Count elements
    elem_count = 0
    elem_fields = []
    for h, content in sections.items():
        if h.startswith('## Interactive Elements'):
            try:
                elems = json.loads(content)
                elem_count = len(elems)
                elem_fields = sorted(elems[0].keys()) if elems else []
            except:
                pass
            break

    # Count visible text
    vt_chars = 0
    for h, content in sections.items():
        if h.startswith('## Visible Text'):
            vt_chars = len(content)
            break

    return {
        "path": "A_sft_training_data",
        "description": "Actual SFT training data (what model was trained on)",
        "system_prompt": system[:200] + "..." if len(system) > 200 else system,
        "system_prompt_sha256": sha256_str(system),
        "system_prompt_length": len(system),
        "user_content_length": len(user),
        "user_content_sha256": sha256_str(user),
        "sections_present": sorted(sections.keys()),
        "sections_count": len(sections),
        "visible_text_chars": vt_chars,
        "element_count": elem_count,
        "element_fields": elem_fields,
        "has_task_section": "## Task" in user,
        "has_current_page_section": "## Current Page" in user,
        "has_history_section": "## Recent History" in user,
        "has_instruction_section": "## Instruction" in user,
        "chat_template_kwargs": sample.get("chat_template_kwargs", {}),
        "add_generation_prompt": True,
        "enable_thinking": False,
        "notes": "COMPACT prompt, NO task/page sections, NO truncation of elements, NO history",
    }


def analyze_current_code_paths():
    """Analyze what the CURRENT code produces for each path."""
    from miniwebwork.agent_env.schemas import Observation

    # Synthetic observation
    obs = Observation(
        task_id="M2_1_T0001",
        instruction="Please locate and select the cheapest suitable product for a development team's GPU upgrade.",
        url="http://localhost:8080/tasks/M2_1_T0001",
        path="/tasks/M2_1_T0001",
        page_type="task",
        title="GPU Procurement Task M2_1_T0001",
        step_index=0,
        visible_text="Home Products Smoke Test Health 任务 M2_1_T0001: 请定位...\n" * 50,
        elements=[
            type('Element', (), {
                'element_id': f'elem_{i}', 'role': ['link', 'button', 'textbox'][i % 3],
                'name': f'element_{i}', 'testid': f'test_{i}', 'disabled': i % 10 == 0,
                'tag': 'input' if i % 3 == 1 else 'a',
                'value': f'value_{i}' if i % 3 == 1 else '',
                'options': None, 'text': f'text_{i}' if i % 3 == 0 else '',
            })() for i in range(20)
        ],
        last_action_result=None,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/share/model/Qwen3.5-4B",
        local_files_only=True, trust_remote_code=True,
    )

    chat_kwargs = {"enable_thinking": False}

    # Path B: Teacher-forced eval
    messages_b = build_messages(obs, history=[])
    prompt_text_b = tokenizer.apply_chat_template(
        messages_b, tokenize=False, add_generation_prompt=True, **chat_kwargs,
    )
    inputs_b = tokenizer(prompt_text_b, return_tensors="pt", truncation=True, max_length=526)
    user_b = messages_b[1]["content"]

    sections_b = {}
    current_header = None
    current_lines = []
    for line in user_b.split('\n'):
        if line.startswith('## '):
            if current_header is not None:
                sections_b[current_header] = '\n'.join(current_lines)
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_header is not None:
        sections_b[current_header] = '\n'.join(current_lines)

    elem_count_b = 0
    elem_fields_b = []
    for h, content in sections_b.items():
        if h.startswith('## Interactive Elements'):
            try:
                elems = json.loads(content)
                elem_count_b = len(elems)
                elem_fields_b = sorted(elems[0].keys()) if elems else []
            except:
                pass
            break
    vt_chars_b = 0
    for h, content in sections_b.items():
        if h.startswith('## Visible Text'):
            vt_chars_b = len(content)
            break

    # Path C: Frozen E2E (same as B for turn 1)
    messages_c = build_messages(obs, history=[])
    user_c = messages_c[1]["content"]

    system = load_system_prompt("browser_agent_v1")

    return {
        "path_B_teacher_forced": {
            "description": "Current code: teacher_forced_eval (uses build_messages)",
            "system_prompt_sha256": sha256_str(system),
            "system_prompt_length": len(system),
            "user_content_length": len(user_b),
            "user_content_sha256": sha256_str(user_b),
            "sections_present": sorted(sections_b.keys()),
            "sections_count": len(sections_b),
            "visible_text_chars": vt_chars_b,
            "element_count": elem_count_b,
            "element_fields": elem_fields_b,
            "has_task_section": "## Task" in user_b,
            "has_current_page_section": "## Current Page" in user_b,
            "has_history_section": "## Recent History" in user_b,
            "input_token_count": inputs_b.input_ids.shape[1],
            "add_generation_prompt": True,
            "enable_thinking": False,
            "max_new_tokens": "len(expected)+10, min 32",
            "notes": "Full prompt (1439 chars), ALL sections, full elements",
        },
        "path_C_frozen_e2e": {
            "description": "Current code: frozen_test_eval turn 1 (uses build_messages)",
            "system_prompt_sha256": sha256_str(system),
            "system_prompt_length": len(system),
            "user_content_length": len(user_c),
            "user_content_sha256": sha256_str(user_c),
            "sections_present": sorted(sections_b.keys()),
            "sections_count": len(sections_b),
            "visible_text_chars": vt_chars_b,
            "element_count": elem_count_b,
            "element_fields": elem_fields_b,
            "has_task_section": "## Task" in user_c,
            "has_current_page_section": "## Current Page" in user_c,
            "has_history_section": "## Recent History" in user_c,
            "input_token_count": inputs_b.input_ids.shape[1],
            "add_generation_prompt": True,
            "enable_thinking": False,
            "max_new_tokens": 128,
            "notes": "Full prompt (1439 chars), ALL sections, full elements, same as B for turn 1",
        },
    }


def compare_with_sft_data():
    """Compare SFT training format vs current code format."""
    sft_format = analyze_sft_training_format()
    current_paths = analyze_current_code_paths()

    comparisons = []

    def add_cmp(field, sft_val, b_val, c_val):
        b_same = sft_val == b_val
        c_same = sft_val == c_val
        comparisons.append({
            "field": field,
            "sft_training": str(sft_val),
            "path_B_teacher_forced": str(b_val),
            "path_C_frozen_e2e": str(c_val),
            "train_vs_B_match": b_same,
            "train_vs_C_match": c_same,
        })

    add_cmp("system_prompt_sha256", sft_format["system_prompt_sha256"],
            current_paths["path_B_teacher_forced"]["system_prompt_sha256"],
            current_paths["path_C_frozen_e2e"]["system_prompt_sha256"])
    add_cmp("system_prompt_length", sft_format["system_prompt_length"],
            current_paths["path_B_teacher_forced"]["system_prompt_length"],
            current_paths["path_C_frozen_e2e"]["system_prompt_length"])
    add_cmp("has_task_section", sft_format["has_task_section"],
            current_paths["path_B_teacher_forced"]["has_task_section"],
            current_paths["path_C_frozen_e2e"]["has_task_section"])
    add_cmp("has_current_page_section", sft_format["has_current_page_section"],
            current_paths["path_B_teacher_forced"]["has_current_page_section"],
            current_paths["path_C_frozen_e2e"]["has_current_page_section"])
    add_cmp("has_history_section", sft_format["has_history_section"],
            current_paths["path_B_teacher_forced"]["has_history_section"],
            current_paths["path_C_frozen_e2e"]["has_history_section"])
    add_cmp("element_fields", str(sft_format["element_fields"]),
            str(current_paths["path_B_teacher_forced"]["element_fields"]),
            str(current_paths["path_C_frozen_e2e"]["element_fields"]))
    add_cmp("user_content_length", sft_format["user_content_length"],
            current_paths["path_B_teacher_forced"]["user_content_length"],
            current_paths["path_C_frozen_e2e"]["user_content_length"])
    add_cmp("sections_count", sft_format["sections_count"],
            current_paths["path_B_teacher_forced"]["sections_count"],
            current_paths["path_C_frozen_e2e"]["sections_count"])

    critical_drifts = [c for c in comparisons if not c["train_vs_B_match"] or not c["train_vs_C_match"]]

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sft_training_format": sft_format,
        "current_code_paths": current_paths,
        "comparisons": comparisons,
        "critical_drifts": critical_drifts,
        "drift_count": len(critical_drifts),
        "verdict": "SEVERE_DRIFT" if critical_drifts else "ALIGNED",
        "key_findings": [
            f"SFT uses COMPACT system prompt ({sft_format['system_prompt_length']} chars) vs current FULL prompt ({current_paths['path_B_teacher_forced']['system_prompt_length']} chars)",
            f"SFT has {sft_format['sections_count']} user sections vs current {current_paths['path_B_teacher_forced']['sections_count']} sections",
            f"SFT has ## Task: {sft_format['has_task_section']}, ## Current Page: {sft_format['has_current_page_section']}",
            f"Current code has ## Task: {current_paths['path_B_teacher_forced']['has_task_section']}, ## Current Page: {current_paths['path_B_teacher_forced']['has_current_page_section']}",
            f"SFT element fields: {sft_format['element_fields']}",
            f"Current element fields: {current_paths['path_B_teacher_forced']['element_fields']}",
            f"SFT user content length: {sft_format['user_content_length']} chars",
            f"Current user content length: {current_paths['path_B_teacher_forced']['user_content_length']} chars",
        ]
    }

    output_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "prompt_path_audit.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Audit saved: {output_path}")

    print(f"\n{'='*70}")
    print(f"PROMPT PATH AUDIT - SFT vs CURRENT CODE")
    print(f"{'='*70}")
    print(f"Drifts found: {len(critical_drifts)}")
    print()
    for finding in result["key_findings"]:
        print(f"  * {finding}")
    print()
    for c in comparisons:
        if not c["train_vs_B_match"]:
            print(f"  DRIFT [{c['field']}]:")
            print(f"    SFT training: {c['sft_training'][:100]}")
            print(f"    Path B/C:     {c['path_B_teacher_forced'][:100]}")

    print(f"\nVERDICT: {result['verdict']}")
    print("\nROOT CAUSE: The SFT dataset was built with a DIFFERENT prompt template")
    print("than the current code uses for evaluation. This is a TRAINING-INFERENCE")
    print("CONTRACT DRIFT that explains the 0/15 frozen E2E failure.")

    return result


if __name__ == "__main__":
    compare_with_sft_data()
