"""Prompt builder: constructs chat messages from Observation + history."""

import hashlib
import json
from pathlib import Path
from typing import Optional

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"

# M4 v2 compact context contract.  These are character/element limits rather
# than a tokenizer-dependent runtime truncation so SFT, online collection and
# frozen evaluation can all build byte-identical prompts from an Observation.
MAX_VISIBLE_TEXT = 5000
HISTORY_WINDOW = 5
MAX_CONTROL_ELEMENTS = 16
MAX_LINK_ELEMENTS = 16
PROMPT_VERSION = "browser_agent_v3_compact"


def load_system_prompt(version: str = None) -> str:
    """Load the system prompt for the versioned shared context contract."""
    version = version or PROMPT_VERSION
    path = PROMPTS_DIR / f"{version}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "You are a web browser procurement agent."


def prompt_sha256(version: str = None) -> str:
    version = version or PROMPT_VERSION
    return hashlib.sha256(load_system_prompt(version).encode()).hexdigest()


def _serialize_one_element(e) -> dict:
    """Serialize the bounded, model-visible fields of one interactive element."""
    d = {"element_id": e.element_id, "role": e.role, "name": e.name,
         "testid": e.testid if e.testid else None, "disabled": e.disabled}
    if e.tag in ("input", "textarea"):
        d["value"] = e.value[:100] if e.value else ""
    if e.tag == "select" and e.options:
        d["options"] = e.options[:10]
    if e.role in ("link", "button"):
        d["text"] = e.text[:100] if e.text else ""
    return d


def _serialize_elements(observation) -> list:
    """Prioritize usable controls, then a bounded prefix of usable links.

    A raw 100-element listing can consume most of the model context on a
    product page.  The task filter/form controls are always more actionable
    than the long product-link tail, so preserve them first and use a fixed
    link budget.  The exact same deterministic selector is used by SFT and
    every rollout path.
    """
    raw_elements = list(observation.elements or [])
    controls = [
        element
        for element in raw_elements
        if element.role != "link" and not element.disabled
    ][:MAX_CONTROL_ELEMENTS]
    links = [
        element
        for element in raw_elements
        if element.role == "link" and not element.disabled
    ][:MAX_LINK_ELEMENTS]
    els = []
    for e in [*controls, *links]:
        els.append(_serialize_one_element(e))
    return els


def visible_element_ids(observation) -> set[str]:
    """Expose the generic compact selector for dataset-contract validation."""
    return {element["element_id"] for element in _serialize_elements(observation)}


def context_contract() -> dict[str, int | str]:
    """Return the versioned prompt limits recorded in M4 provenance."""
    return {
        "prompt_version": PROMPT_VERSION,
        "max_visible_text_characters": MAX_VISIBLE_TEXT,
        "history_window": HISTORY_WINDOW,
        "max_control_elements": MAX_CONTROL_ELEMENTS,
        "max_link_elements": MAX_LINK_ELEMENTS,
    }


def _serialize_history(history: list, window: int = HISTORY_WINDOW) -> list:
    entries = []
    for h in history[-window:]:
        entries.append({
            "turn": h.get("model_turn_index", 0),
            "action": h.get("action"),
            "parse_ok": h.get("parse_ok", False),
            "result": h.get("result", ""),
            "page_type": h.get("page_type", ""),
        })
    return entries


def build_messages(observation, history: list = None, version: str = None,
                   max_text: int = MAX_VISIBLE_TEXT, history_window: int = HISTORY_WINDOW) -> list:
    """Build chat messages for the model under the versioned compact contract."""
    version = version or PROMPT_VERSION
    system = load_system_prompt(version)
    history = history or []

    # Serialize current observation
    els = _serialize_elements(observation)
    visible_text = observation.visible_text or ""
    text_truncated = len(visible_text) > max_text
    if text_truncated:
        visible_text = visible_text[:max_text]

    hist = _serialize_history(history, history_window)
    last_result = observation.last_action_result

    user_content = f"""## Task
task_id: {observation.task_id}
instruction: {observation.instruction}

## Current Page
url: {observation.url}
path: {observation.path}
page_type: {observation.page_type}
title: {observation.title}
step: {observation.step_index}

## Visible Text (truncated={text_truncated})
{visible_text}

## Interactive Elements ({len(els)})
{json.dumps(els, ensure_ascii=False)}

## Recent History ({len(hist)} turns)
{json.dumps(hist, ensure_ascii=False)}

## Last Action Result
{json.dumps(last_result, ensure_ascii=False) if last_result else 'N/A'}

## Instruction
Output exactly one JSON action. Only use element_id from the elements list above."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def compute_message_hash(messages: list) -> str:
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()
