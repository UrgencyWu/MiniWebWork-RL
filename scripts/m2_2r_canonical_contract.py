"""M2.2R Canonical Prompt Contract v2.

This contract defines the single source of truth for prompt construction.
ALL paths (SFT data construction, teacher-forced eval, frozen E2E) MUST use this.
"""
import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")

# System Prompt
SYSTEM_PROMPT = """You are a web browser procurement agent. You interact with a website by outputting exactly one JSON action per turn.

## IMPORTANT RULES
- Output ONLY a single JSON object. No markdown, no explanation, no thinking.
- Only use element_id values that exist in the current observation's elements list.
- Do NOT invent element IDs. If you don't see a suitable element, use finish.
- Each turn you can perform exactly ONE action.
- To complete a procurement task, you MUST navigate to a product, select it, fill the form, and submit.
- The finish action does NOT submit a procurement -- it just ends the episode.
- If you cannot find what you need, navigate differently or use finish.

## SUPPORTED ACTIONS
click: {"action":"click","target":"<element_id>"}
fill: {"action":"fill","target":"<element_id>","value":"<text>"}
select: {"action":"select","target":"<element_id>","value":"<option_value>"}
check: {"action":"check","target":"<element_id>","checked":true}
back: {"action":"back"}
submit: {"action":"submit","target":"<element_id>"}
finish: {"action":"finish"}

## PAGE TYPES
- task: Start page with task description and start button
- products: Product listing with search/filter form and product cards
- product_detail: Single product with details and select button
- supplier_detail: Supplier information
- procurement_form: Form to finalize procurement (fill justification, submit)
- procurement_result: Final submission result page"""

# Action Schema
ACTION_SCHEMA = {
    "type": "object",
    "required": ["action"],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["click", "fill", "select", "check", "back", "submit", "finish"],
            "description": "The action to perform"
        },
        "target": {
            "type": "string",
            "description": "element_id from the current observation's elements list"
        },
        "value": {
            "type": "string",
            "maxLength": 500,
            "description": "Text value for fill/select actions"
        },
        "checked": {
            "type": "boolean",
            "description": "Boolean for check action"
        }
    },
    "additionalProperties": False
}

# Observation limits
MAX_VISIBLE_TEXT = 8000
MAX_ELEMENTS = 100
HISTORY_WINDOW = 5

# Observation fields included in prompt
OBSERVATION_SECTIONS = [
    "## Task",           # task_id, instruction
    "## Current Page",   # url, path, page_type, title, step
    "## Visible Text",   # truncated to MAX_VISIBLE_TEXT
    "## Interactive Elements",  # up to MAX_ELEMENTS, all fields
    "## Recent History", # up to HISTORY_WINDOW turns
    "## Last Action Result",    # result of previous action
    "## Instruction",    # output format instruction
]

# Element fields serialized
ELEMENT_FIELDS = ["element_id", "role", "name", "testid", "disabled", "value", "options", "text"]

# Chat template config
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
ADD_GENERATION_PROMPT = True

# Generation config
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
USE_CACHE = True


def build_contract_manifest() -> dict:
    """Build the canonical contract manifest."""
    contract = {
        "contract_version": "v2",
        "contract_name": "browser_agent_v2",
        "created_at": "2026-07-27",
        "description": "Canonical prompt contract for M2.2R SFT training and evaluation",

        "system_prompt": {
            "text": SYSTEM_PROMPT,
            "sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "length_chars": len(SYSTEM_PROMPT),
        },

        "action_schema": ACTION_SCHEMA,

        "observation": {
            "max_visible_text": MAX_VISIBLE_TEXT,
            "max_elements": MAX_ELEMENTS,
            "history_window": HISTORY_WINDOW,
            "element_fields": ELEMENT_FIELDS,
            "sections": OBSERVATION_SECTIONS,
            "section_order": OBSERVATION_SECTIONS,
        },

        "chat_template": {
            "kwargs": CHAT_TEMPLATE_KWARGS,
            "add_generation_prompt": ADD_GENERATION_PROMPT,
            "enable_thinking": False,
        },

        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "use_cache": USE_CACHE,
            "num_beams": 1,
        },

        "truncation_policy": {
            "order": ["visible_text", "elements", "history", "total_tokens"],
            "visible_text_chars": MAX_VISIBLE_TEXT,
            "elements_max": MAX_ELEMENTS,
            "history_max_turns": HISTORY_WINDOW,
            "max_sequence_length": 8192,
        },

        "paths": {
            "prompt_file": "prompts/browser_agent_v2/system.txt",
            "observation_schema": "prompts/browser_agent_v2/observation_schema.json",
            "action_schema": "prompts/browser_agent_v2/action_schema.json",
            "manifest": "prompts/browser_agent_v2/manifest.json",
        },
    }
    return contract


def save_contract():
    """Save the canonical contract files."""
    contract_dir = PROJECT_ROOT / "prompts" / "browser_agent_v2"
    contract_dir.mkdir(parents=True, exist_ok=True)

    contract = build_contract_manifest()

    # Save system prompt
    (contract_dir / "system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")

    # Save observation schema
    obs_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "instruction": {"type": "string"},
            "url": {"type": "string"},
            "path": {"type": "string"},
            "page_type": {"type": "string"},
            "title": {"type": "string"},
            "step_index": {"type": "integer"},
            "visible_text": {"type": "string", "maxLength": MAX_VISIBLE_TEXT},
            "elements": {
                "type": "array",
                "maxItems": MAX_ELEMENTS,
                "items": {"type": "object", "properties": {
                    "element_id": {"type": "string"},
                    "role": {"type": "string"},
                    "name": {"type": "string"},
                    "testid": {"type": ["string", "null"]},
                    "disabled": {"type": "boolean"},
                    "value": {"type": "string"},
                    "options": {"type": ["array", "null"]},
                    "text": {"type": "string"},
                }}
            },
            "last_action_result": {"type": ["object", "null"]},
        },
        "required": ["task_id", "instruction", "url", "page_type", "elements"],
    }
    (contract_dir / "observation_schema.json").write_text(
        json.dumps(obs_schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save action schema
    (contract_dir / "action_schema.json").write_text(
        json.dumps(ACTION_SCHEMA, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save manifest
    (contract_dir / "manifest.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Canonical contract saved to: {contract_dir}")
    print(f"  System prompt SHA256: {contract['system_prompt']['sha256']}")
    print(f"  System prompt length: {contract['system_prompt']['length_chars']}")

    # Also save to artifacts
    artifacts_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "canonical_prompt_manifest.json"
    artifacts_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Also saved to: {artifacts_path}")

    return contract


if __name__ == "__main__":
    save_contract()
