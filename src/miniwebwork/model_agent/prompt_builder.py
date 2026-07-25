"""Prompt builder: constructs chat messages from Observation + history."""

import hashlib
import json
from pathlib import Path
from typing import Optional

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"

MAX_VISIBLE_TEXT = 8000
HISTORY_WINDOW = 5
MAX_ELEMENTS = 100


def load_system_prompt(version: str = "browser_agent_v1") -> str:
    path = PROMPTS_DIR / f"{version}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "You are a web browser procurement agent."


def prompt_sha256(version: str = "browser_agent_v1") -> str:
    return hashlib.sha256(load_system_prompt(version).encode()).hexdigest()


def _serialize_elements(observation) -> list:
    """Compact element serialization for prompt."""
    els = []
    for e in (observation.elements or [])[:MAX_ELEMENTS]:
        d = {"element_id": e.element_id, "role": e.role, "name": e.name,
             "testid": e.testid if e.testid else None, "disabled": e.disabled}
        if e.tag in ("input", "textarea"):
            d["value"] = e.value[:100] if e.value else ""
        if e.tag == "select" and e.options:
            d["options"] = e.options[:10]
        if e.role in ("link", "button"):
            d["text"] = e.text[:100] if e.text else ""
        els.append(d)
    return els


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


def build_messages(observation, history: list = None, version: str = "browser_agent_v1",
                   max_text: int = MAX_VISIBLE_TEXT, history_window: int = HISTORY_WINDOW) -> list:
    """Build chat messages for the model."""
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
