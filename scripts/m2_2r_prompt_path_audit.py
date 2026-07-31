"""M2.2R Step 1: Prompt Path Audit - Compare the three execution paths.

A. SFT data construction path (build_m2_2_dataset.py)
B. Teacher-forced Valid eval path (evaluate_m2_2.py teacher_forced_eval)
C. Frozen E2E path (evaluate_m2_2.py frozen_test_eval)
"""
import json
import hashlib
import sys
import os
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.model_agent.prompt_builder import (
    build_messages, load_system_prompt, compute_message_hash,
    prompt_sha256, MAX_VISIBLE_TEXT, HISTORY_WINDOW, MAX_ELEMENTS,
)
from miniwebwork.tasks import get_public_task, get_oracle
from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent
from miniwebwork.agent_env.schemas import AgentAction


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def analyze_training_path(task_id: str) -> dict:
    """Simulate what build_m2_2_dataset.py does for a single turn."""
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks" / "m2_1")

    with ProcurementBrowserEnv(max_steps=25, run_id="audit_train", headless=True) as env:
        obs = env.reset(task_id)
        oracle = get_oracle(task_id)
        expert = OracleExpertProcurementAgent(oracle, max_steps=25)
        expert.reset()
        expert_action = expert.act(obs)

        # Step 1: Build full messages
        messages = build_messages(obs, history=[])

        # Step 2: Apply training truncation (same as build_m2_2_dataset.py lines 67-137)
        user_msg = messages[1]
        user_content = user_msg["content"]

        sections = {}
        current_header = None
        current_lines = []
        for line in user_content.split('\n'):
            if line.startswith('## '):
                if current_header is not None:
                    sections[current_header] = '\n'.join(current_lines)
                current_header = line.strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_header is not None:
            sections[current_header] = '\n'.join(current_lines)

        # 1. Truncate visible text to 300 chars
        orig_vt_len = 0
        for h in sections:
            if h.startswith('## Visible Text'):
                orig_vt_len = len(sections[h])
                if len(sections[h]) > 300:
                    sections[h] = sections[h][:300] + "..."
                break

        # 2. Truncate elements to 8, keep only essential fields
        orig_elem_count = 0
        orig_elem_fields = []
        for h in sections:
            if h.startswith('## Interactive Elements'):
                elements_str = sections[h]
                try:
                    elements_list = json.loads(elements_str)
                    orig_elem_count = len(elements_list)
                    orig_elem_fields = sorted(elements_list[0].keys()) if elements_list else []
                    if len(elements_list) > 8:
                        minimal = []
                        for e in elements_list[:8]:
                            minimal.append({
                                "element_id": e.get("element_id", ""),
                                "role": e.get("role", ""),
                                "name": (e.get("name") or "")[:20],
                                "disabled": e.get("disabled", False),
                            })
                        sections[h] = json.dumps(minimal, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
                break

        # 3. Remove history section entirely
        sections.pop('## Recent History', None)
        for h in list(sections.keys()):
            if h.startswith('## Recent History'):
                del sections[h]

        # Rebuild with canonical order
        parts = []
        header_order = [
            '## Task', '## Current Page', '## Visible Text',
            '## Interactive Elements', '## Last Action Result', '## Instruction',
        ]
        for expected in header_order:
            for h in sections:
                if h == expected or h.startswith(expected):
                    parts.append(h)
                    parts.append(sections[h])
                    break
        user_content_truncated = '\n'.join(parts)
        messages[1] = {"role": "user", "content": user_content_truncated}

        system_prompt = load_system_prompt("browser_agent_v1")
        elem_count_truncated = 0
        elem_section = None
        for h, content in sections.items():
            if h.startswith('## Interactive Elements'):
                elem_section = content
                break
        if elem_section:
            try:
                elem_count_truncated = len(json.loads(elem_section))
            except:
                pass

        vt_after = 0
        for h, content in sections.items():
            if h.startswith('## Visible Text'):
                vt_after = len(content)
                break

        return {
            "path": "A_sft_data_construction",
            "description": "build_m2_2_dataset.py with training truncation",
            "system_prompt_sha256": sha256_str(system_prompt),
            "system_prompt_length": len(system_prompt),
            "user_content_original_length": len(user_content),
            "user_content_truncated_length": len(user_content_truncated),
            "user_content_sha256": sha256_str(user_content_truncated),
            "messages_sha256": compute_message_hash(messages),
            "visible_text_original_chars": orig_vt_len,
            "visible_text_truncated_chars": vt_after,
            "element_count_original": orig_elem_count,
            "element_count_truncated": elem_count_truncated,
            "element_fields": orig_elem_fields,
            "history_turns": 0,
            "has_history_section": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "add_generation_prompt": True,
            "enable_thinking": False,
            "max_new_tokens": None,
            "notes": "Training data: vt=300, elements=8 (minimal fields), no history",
        }


def analyze_valid_eval_path(task_id: str) -> dict:
    """Simulate evaluate_m2_2.py teacher_forced_eval for a single sample."""
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks" / "m2_1")

    with ProcurementBrowserEnv(max_steps=25, run_id="audit_valid", headless=True) as env:
        obs = env.reset(task_id)
        oracle = get_oracle(task_id)
        expert = OracleExpertProcurementAgent(oracle, max_steps=25)
        expert.reset()
        expert_action = expert.act(obs)

        # Build messages - NO truncation (teacher_forced_eval uses messages from dataset)
        messages = build_messages(obs, history=[])

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/share/model/Qwen3.5-4B",
            local_files_only=True, trust_remote_code=True,
        )

        chat_kwargs = {"enable_thinking": False}
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=526)
        input_token_count = inputs.input_ids.shape[1]

        system_prompt = load_system_prompt("browser_agent_v1")
        els_raw = messages[1]["content"]

        elem_match = re.search(r'## Interactive Elements \((\d+)\)', els_raw)
        elem_count = int(elem_match.group(1)) if elem_match else 0

        vt_text_after = els_raw.split('## Visible Text (truncated=')[1].split('\n', 1)[1]
        vt_text_after = vt_text_after.split('\n## ')[0]
        vt_chars = len(vt_text_after)

        return {
            "path": "B_teacher_forced_eval",
            "description": "evaluate_m2_2.py teacher_forced_eval (no truncation)",
            "system_prompt_sha256": sha256_str(system_prompt),
            "system_prompt_length": len(system_prompt),
            "user_content_length": len(messages[1]["content"]),
            "user_content_sha256": sha256_str(messages[1]["content"]),
            "messages_sha256": compute_message_hash(messages),
            "visible_text_chars": vt_chars,
            "element_count": elem_count,
            "element_fields": ["element_id", "role", "name", "testid", "disabled", "value", "options", "text"],
            "history_turns": 0,
            "has_history_section": True,  # section exists but empty for turn 1
            "history_turns_actual": 0,
            "chat_template_kwargs": chat_kwargs,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "max_new_tokens": "len(expected)+10, min 32",
            "input_token_count": input_token_count,
            "notes": "Valid eval: NO truncation, full 8000 vt, 100 elements, all fields",
        }


def analyze_frozen_e2e_path(task_id: str) -> dict:
    """Simulate evaluate_m2_2.py frozen_test_eval for turn 1."""
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks" / "m2_1")

    with ProcurementBrowserEnv(max_steps=15, run_id="audit_frozen", headless=True) as env:
        obs = env.reset(task_id)

        # Turn 1: no history yet (same as other paths)
        messages = build_messages(obs, history=[])

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/share/model/Qwen3.5-4B",
            local_files_only=True, trust_remote_code=True,
        )

        chat_kwargs = {"enable_thinking": False}
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_token_count = inputs.input_ids.shape[1]

        system_prompt = load_system_prompt("browser_agent_v1")
        els_raw = messages[1]["content"]

        elem_match = re.search(r'## Interactive Elements \((\d+)\)', els_raw)
        elem_count = int(elem_match.group(1)) if elem_match else 0

        vt_text_after = els_raw.split('## Visible Text (truncated=')[1].split('\n', 1)[1]
        vt_text_after = vt_text_after.split('\n## ')[0]
        vt_chars = len(vt_text_after)

        return {
            "path": "C_frozen_e2e",
            "description": "evaluate_m2_2.py frozen_test_eval (agent_loop turn 1)",
            "system_prompt_sha256": sha256_str(system_prompt),
            "system_prompt_length": len(system_prompt),
            "user_content_length": len(messages[1]["content"]),
            "user_content_sha256": sha256_str(messages[1]["content"]),
            "messages_sha256": compute_message_hash(messages),
            "visible_text_chars": vt_chars,
            "element_count": elem_count,
            "element_fields": ["element_id", "role", "name", "testid", "disabled", "value", "options", "text"],
            "history_turns": 0,
            "has_history_section": True,
            "history_turns_actual": 0,
            "chat_template_kwargs": chat_kwargs,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "max_new_tokens": 128,
            "input_token_count": input_token_count,
            "notes": "Frozen E2E: NO truncation, full 8000 vt, 100 elements, all fields",
        }


def compare_field(name, val_a, val_b, val_c):
    all_same = (val_a == val_b == val_c)
    return {
        "field": name,
        "train_data": str(val_a),
        "valid_eval": str(val_b),
        "frozen_e2e": str(val_c),
        "consistent": all_same,
    }


def main():
    task_id = "TASK-001"
    print(f"Auditing prompt paths for task: {task_id}")

    train_path = analyze_training_path(task_id)
    valid_path = analyze_valid_eval_path(task_id)
    frozen_path = analyze_frozen_e2e_path(task_id)

    comparisons = [
        compare_field("system_prompt_sha256", train_path["system_prompt_sha256"], valid_path["system_prompt_sha256"], frozen_path["system_prompt_sha256"]),
        compare_field("system_prompt_length", train_path["system_prompt_length"], valid_path["system_prompt_length"], frozen_path["system_prompt_length"]),
        compare_field("user_content_sha256", train_path["user_content_sha256"], valid_path["user_content_sha256"], frozen_path["user_content_sha256"]),
        compare_field("messages_sha256", train_path["messages_sha256"], valid_path["messages_sha256"], frozen_path["messages_sha256"]),
        compare_field("visible_text_chars", train_path["visible_text_truncated_chars"], valid_path["visible_text_chars"], frozen_path["visible_text_chars"]),
        compare_field("element_count", train_path["element_count_truncated"], valid_path["element_count"], frozen_path["element_count"]),
        compare_field("element_fields", str(train_path["element_fields"]), str(valid_path["element_fields"]), str(frozen_path["element_fields"])),
        compare_field("history_turns", train_path["history_turns"], valid_path["history_turns"], frozen_path["history_turns"]),
        compare_field("has_history_section", train_path["has_history_section"], valid_path["has_history_section"], frozen_path["has_history_section"]),
        compare_field("chat_template_kwargs", str(train_path["chat_template_kwargs"]), str(valid_path["chat_template_kwargs"]), str(frozen_path["chat_template_kwargs"])),
        compare_field("add_generation_prompt", train_path["add_generation_prompt"], valid_path["add_generation_prompt"], frozen_path["add_generation_prompt"]),
        compare_field("enable_thinking", train_path["enable_thinking"], valid_path["enable_thinking"], frozen_path["enable_thinking"]),
        compare_field("max_new_tokens", str(train_path["max_new_tokens"]), str(valid_path["max_new_tokens"]), str(frozen_path["max_new_tokens"])),
        compare_field("input_token_count", train_path.get("input_token_count", "N/A"), valid_path.get("input_token_count", "N/A"), frozen_path.get("input_token_count", "N/A")),
    ]

    inconsistencies = [c for c in comparisons if not c["consistent"]]

    result = {
        "task_id": task_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paths": {
            "A_sft_data_construction": train_path,
            "B_teacher_forced_eval": valid_path,
            "C_frozen_e2e": frozen_path,
        },
        "comparison_table": comparisons,
        "inconsistencies": inconsistencies,
        "inconsistency_count": len(inconsistencies),
        "verdict": "DRIFT_DETECTED" if inconsistencies else "ALL_CONSISTENT",
    }

    output_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "prompt_path_audit.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nAudit saved: {output_path}")

    print(f"\n{'='*70}")
    print(f"PROMPT PATH AUDIT RESULTS")
    print(f"{'='*70}")
    print(f"Task: {task_id}")
    print(f"Inconsistencies: {len(inconsistencies)}")
    print()

    for c in comparisons:
        status = "OK" if c["consistent"] else "DRIFT!"
        print(f"  [{status}] {c['field']}")
        if not c["consistent"]:
            print(f"         Train:    {c['train_data']}")
            print(f"         Valid:    {c['valid_eval']}")
            print(f"         Frozen:   {c['frozen_e2e']}")

    print(f"\nVerdict: {result['verdict']}")
    return result


if __name__ == "__main__":
    main()
