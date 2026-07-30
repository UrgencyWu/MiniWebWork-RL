"""Two-level output parser: strict JSON + bounded fallback extraction."""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParseResult:
    """Result of parsing a model output."""
    raw_output: str = ""
    strict_json_success: bool = False
    fallback_used: bool = False
    fallback_type: str = ""  # "fenced" or "balanced_brace"
    extracted_json: str = ""
    fallback_json_success: bool = False
    parsed_payload: Optional[dict] = None
    schema_valid: bool = False
    schema_error: str = ""
    errors: list = field(default_factory=list)


# Supported actions
VALID_ACTIONS = {"click", "fill", "select", "check", "back", "submit", "finish"}


def parse(raw: str) -> ParseResult:
    """Parse model output with strict-first, fallback-second strategy."""
    result = ParseResult(raw_output=raw)

    if not raw or not raw.strip():
        result.errors.append("empty_generation")
        return result

    cleaned = raw.strip()

    # Level 1: Strict parse — entire text must be a single JSON object
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            result.strict_json_success = True
            result.parsed_payload = payload
            result.extracted_json = cleaned
    except json.JSONDecodeError:
        pass

    # Level 2: Fallback extraction
    if not result.strict_json_success:
        # Try fenced code block
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if m:
            result.fallback_used = True
            result.fallback_type = "fenced"
            try:
                payload = json.loads(m.group(1).strip())
                if isinstance(payload, dict):
                    result.fallback_json_success = True
                    result.parsed_payload = payload
                    result.extracted_json = m.group(1).strip()
            except json.JSONDecodeError:
                pass

    if not result.strict_json_success and not result.fallback_json_success:
        # Try balanced brace extraction
        brace_start = raw.find("{")
        if brace_start >= 0:
            depth = 0
            brace_end = -1
            for i in range(brace_start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end = i
                        break
            if brace_end > brace_start:
                result.fallback_used = True
                result.fallback_type = "balanced_brace"
                candidate = raw[brace_start:brace_end + 1]
                try:
                    payload = json.loads(candidate)
                    if isinstance(payload, dict):
                        result.fallback_json_success = True
                        result.parsed_payload = payload
                        result.extracted_json = candidate
                except json.JSONDecodeError:
                    pass

    if not result.parsed_payload:
        result.errors.append("non_json_output" if not result.strict_json_success else "empty_parsed")
        return result

    # Normalize field names: accept element_id as alias for target
    payload = result.parsed_payload
    if "element_id" in payload and "target" not in payload:
        payload["target"] = payload.pop("element_id")

    # Normalize action names: accept fill_text as alias for fill
    if payload.get("action") == "fill_text":
        payload["action"] = "fill"

    # Schema validation
    _validate_schema(result)

    return result


def _validate_schema(result: ParseResult):
    """Validate parsed payload against action schema."""
    payload = result.parsed_payload
    if not isinstance(payload, dict):
        result.errors.append("not_a_dict")
        return

    action = payload.get("action", "")
    if not isinstance(action, str) or action not in VALID_ACTIONS:
        result.schema_error = f"unknown_action: {action!r}"
        result.errors.append("unknown_action")
        return

    # Check for unknown fields
    allowed_fields = {"action", "target", "value", "checked"}
    extra = set(payload.keys()) - allowed_fields
    if extra:
        result.schema_error = f"extra_fields: {extra}"
        result.errors.append("extra_fields")

    # Action-specific validation
    if action in ("click", "fill", "select", "check", "submit"):
        if "target" not in payload or not payload["target"]:
            result.schema_error = "missing_target"
            result.errors.append("missing_target")
            return

    if action == "fill" and not payload.get("value"):
        result.schema_error = "fill_missing_value"
        result.errors.append("fill_missing_value")
        return

    if action == "select" and "value" not in payload:
        result.schema_error = "select_missing_value"
        result.errors.append("select_missing_value")
        return

    if action == "check" and "checked" not in payload:
        result.schema_error = "check_missing_checked"
        result.errors.append("check_missing_checked")
        return

    # Value length check
    if payload.get("value") and len(str(payload["value"])) > 500:
        result.errors.append("value_too_long")

    if not result.errors:
        result.schema_valid = True
