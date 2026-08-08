"""Prompt builder: constructs chat messages from Observation + history."""

import hashlib
import json
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"

# M4 v2 compact context contract.  These are character/element limits rather
# than a tokenizer-dependent runtime truncation so SFT, online collection and
# frozen evaluation can all build byte-identical prompts from an Observation.
MAX_VISIBLE_TEXT = 5000
HISTORY_WINDOW = 5
MAX_CONTROL_ELEMENTS = 16
MAX_LINK_ELEMENTS = 16
PROMPT_VERSION = "browser_agent_v3_compact"
LONG_MEMORY_PROMPT_VERSION = "browser_agent_v4_long_memory"
MAX_EVIDENCE_ENTRIES = 8
MAX_EVIDENCE_TEXT = 1000
EVIDENCE_PAGE_TYPES = frozenset({"supplier_detail", "product_detail"})


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


def context_contract(version: str = None) -> dict[str, int | str | list[str]]:
    """Return the versioned prompt limits recorded in M4 provenance."""
    version = version or PROMPT_VERSION
    contract: dict[str, int | str | list[str]] = {
        "prompt_version": version,
        "max_visible_text_characters": MAX_VISIBLE_TEXT,
        "history_window": HISTORY_WINDOW,
        "max_control_elements": MAX_CONTROL_ELEMENTS,
        "max_link_elements": MAX_LINK_ELEMENTS,
    }
    if version == LONG_MEMORY_PROMPT_VERSION:
        contract.update(
            {
                "evidence_memory_contract": "public_observation_v1",
                "max_evidence_entries": MAX_EVIDENCE_ENTRIES,
                "max_evidence_text_characters_per_entry": MAX_EVIDENCE_TEXT,
                "evidence_page_types": sorted(EVIDENCE_PAGE_TYPES),
                "current_url_contract": "path_only_without_origin_query_or_fragment",
            }
        )
    return contract


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


def _canonical_public_path(value: object) -> str:
    """Remove query/fragment state that may contain run-specific identifiers."""

    parsed = urlsplit(str(value or ""))
    return (parsed.path or "/")[:300]


def _public_evidence_entry(observation) -> dict | None:
    """Summarize only fields that the policy can see in the current browser state."""

    page_type = str(getattr(observation, "page_type", ""))
    if page_type not in EVIDENCE_PAGE_TYPES:
        return None
    visible_text = " ".join(str(getattr(observation, "visible_text", "") or "").split())
    return {
        "page_type": page_type,
        "path": _canonical_public_path(getattr(observation, "path", "")),
        "title": str(getattr(observation, "title", "") or "")[:200],
        "visible_text": visible_text[:MAX_EVIDENCE_TEXT],
    }


def update_evidence_memory(
    evidence_memory: list[dict],
    observation,
    *,
    version: str = None,
) -> list[dict]:
    """Update bounded public-state memory in place under the v4 contract.

    The summary never inspects an oracle, verifier result, expected answer, or
    arbitrary attributes on ``observation``. Revisited public states replace
    their older entry and move to the newest position deterministically.
    """

    version = version or PROMPT_VERSION
    if version != LONG_MEMORY_PROMPT_VERSION:
        return evidence_memory
    if not isinstance(evidence_memory, list):
        raise TypeError("evidence_memory must be a list")
    entry = _public_evidence_entry(observation)
    if entry is None:
        return evidence_memory
    identity = (entry["page_type"], entry["path"])
    evidence_memory[:] = [
        existing
        for existing in evidence_memory
        if (existing.get("page_type"), existing.get("path")) != identity
    ]
    evidence_memory.append(entry)
    del evidence_memory[:-MAX_EVIDENCE_ENTRIES]
    return evidence_memory


def _serialize_evidence_memory(evidence_memory: list[dict] | None) -> list[dict]:
    """Fail-safe serializer: retain only the four public fields of each entry."""

    serialized = []
    for entry in (evidence_memory or [])[-MAX_EVIDENCE_ENTRIES:]:
        if not isinstance(entry, dict):
            continue
        serialized.append(
            {
                "page_type": str(entry.get("page_type", ""))[:100],
                "path": _canonical_public_path(entry.get("path", "")),
                "title": str(entry.get("title", ""))[:200],
                "visible_text": str(entry.get("visible_text", ""))[:MAX_EVIDENCE_TEXT],
            }
        )
    return serialized


def build_messages(observation, history: list = None, version: str = None,
                   max_text: int = MAX_VISIBLE_TEXT, history_window: int = HISTORY_WINDOW,
                   evidence_memory: list[dict] = None) -> list:
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

    if version == LONG_MEMORY_PROMPT_VERSION:
        current_path = _canonical_public_path(observation.path)
        current_url = _canonical_public_path(observation.url)
        evidence = _serialize_evidence_memory(evidence_memory)
        evidence_block = f"""## Public Evidence Memory ({len(evidence)} states; policy-visible only)
{json.dumps(evidence, ensure_ascii=False)}

"""
    else:
        current_path = observation.path
        current_url = observation.url
        evidence_block = ""

    user_content = f"""## Task
task_id: {observation.task_id}
instruction: {observation.instruction}

## Current Page
url: {current_url}
path: {current_path}
page_type: {observation.page_type}
title: {observation.title}
step: {observation.step_index}

## Visible Text (truncated={text_truncated})
{visible_text}

## Interactive Elements ({len(els)})
{json.dumps(els, ensure_ascii=False)}

{evidence_block}## Recent History ({len(hist)} turns)
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
