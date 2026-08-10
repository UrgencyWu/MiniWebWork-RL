"""Strict, bounded WebShop command parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

COMMAND_RE = re.compile(r"^(search|click)\[(.*)\]$", re.IGNORECASE | re.DOTALL)
MAX_COMMAND_CHARACTERS = 512
MAX_SEARCH_QUERY_CHARACTERS = 200


@dataclass(frozen=True)
class WebShopCommand:
    command: str

    def to_dict(self) -> dict[str, str]:
        return {"command": self.command}


@dataclass
class WebShopCommandParseResult:
    raw_output: str = ""
    strict_json_success: bool = False
    fallback_used: bool = False
    parsed_payload: dict[str, Any] | None = None
    schema_valid: bool = False
    action: WebShopCommand | None = None
    errors: list[str] = field(default_factory=list)


def normalize_command(command: str) -> str:
    if not isinstance(command, str):
        raise ValueError("WebShop command must be a string")
    value = " ".join(command.strip().split())
    if not value or len(value) > MAX_COMMAND_CHARACTERS:
        raise ValueError("WebShop command length is invalid")
    match = COMMAND_RE.fullmatch(value)
    if match is None:
        raise ValueError("WebShop command must be search[...] or click[...]")
    verb = match.group(1).lower()
    argument = match.group(2).strip()
    if not argument:
        raise ValueError("WebShop command argument is empty")
    if verb == "search" and len(argument) > MAX_SEARCH_QUERY_CHARACTERS:
        raise ValueError("WebShop search query is too long")
    return f"{verb}[{argument}]"


def _single_balanced_object(raw: str) -> str | None:
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        character = raw[index]
        if escaped:
            escaped = False
            continue
        if in_string and character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return None


def parse_command_output(raw: str) -> WebShopCommandParseResult:
    result = WebShopCommandParseResult(raw_output=str(raw or ""))
    cleaned = result.raw_output.strip()
    if not cleaned:
        result.errors.append("empty_generation")
        return result
    payload: Any = None
    try:
        payload = json.loads(cleaned)
        result.strict_json_success = isinstance(payload, dict)
    except json.JSONDecodeError:
        candidate = _single_balanced_object(cleaned)
        if candidate is not None:
            result.fallback_used = True
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                payload = None
    if not isinstance(payload, dict):
        result.errors.append("non_json_object")
        return result
    result.parsed_payload = payload
    if set(payload) != {"command"}:
        result.errors.append("command_schema_keys")
        return result
    try:
        action = WebShopCommand(normalize_command(payload["command"]))
    except (TypeError, ValueError) as exc:
        result.errors.append(f"invalid_command: {exc}")
        return result
    result.action = action
    result.schema_valid = True
    return result
