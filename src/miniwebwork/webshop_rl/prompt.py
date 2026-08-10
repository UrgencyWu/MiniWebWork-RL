"""Versioned public-only prompt contract for M5 WebShop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROMPT_CONTRACT = "webshop_agent_v1_compact"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = PROJECT_ROOT / "prompts" / f"{PROMPT_CONTRACT}.txt"
MAX_OBSERVATION_CHARACTERS = 8000
MAX_HISTORY_TURNS = 5


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _safe_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "turn": int(item.get("turn", 0)),
            "command": str(item.get("command", ""))[:512],
            "success": bool(item.get("success", False)),
            "error_code": str(item.get("error_code", ""))[:100],
            "page_type": str(item.get("page_type", ""))[:100],
        }
        for item in history[-MAX_HISTORY_TURNS:]
        if isinstance(item, dict)
    ]


def build_messages(observation: Any, history: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    visible_text = str(observation.visible_text or "")
    truncated = bool(getattr(observation, "text_truncated", False)) or len(visible_text) > MAX_OBSERVATION_CHARACTERS
    visible_text = visible_text[:MAX_OBSERVATION_CHARACTERS]
    actions = [str(action) for action in observation.available_actions]
    user = f"""## Shopping task
task_id: {observation.task_id}
instruction: {observation.instruction}

## Current public state
page_type: {observation.page_type}
step: {observation.step_index}
observation_truncated: {str(truncated).lower()}
{visible_text}

## Executable actions ({len(actions)})
{json.dumps(actions, ensure_ascii=False)}

## Recent public history
{json.dumps(_safe_history(history or []), ensure_ascii=False)}

## Required output
Return exactly one compact JSON object: {{"command":"..."}}.
For search, replace <your query> with concise keywords. For click, copy one listed action exactly.
Do not explain and do not invent a click target."""
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": user},
    ]


def compute_message_hash(messages: list[dict[str, str]]) -> str:
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
