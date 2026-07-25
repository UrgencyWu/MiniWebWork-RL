"""Versioned schemas for Observation, Action, StepResult, and Trajectory."""

from dataclasses import dataclass, field
from typing import Any, Optional

OBSERVATION_SCHEMA_VERSION = "1.0"
ACTION_SCHEMA_VERSION = "1.0"
TRAJECTORY_SCHEMA_VERSION = "1.0"

# Supported page types
VALID_PAGE_TYPES = {
    "home", "task", "products", "product_detail", "supplier_detail",
    "procurement_form", "procurement_result", "smoke", "error", "unknown",
}

# Supported action types
VALID_ACTION_TYPES = {"click", "fill", "select", "check", "back", "submit", "finish"}

# Role-to-action compatibility
ROLE_ACTION_COMPAT = {
    "link": {"click"},
    "button": {"click"},
    "textbox": {"fill"},
    "searchbox": {"fill"},
    "checkbox": {"check"},
    "combobox": {"select"},
    "spinbutton": {"fill"},
    "textarea": {"fill"},
}

# Error codes
ERROR_CODES = {
    "invalid_action_type",
    "malformed_action",
    "invalid_target",
    "stale_target",
    "incompatible_action",
    "disabled_element",
    "value_required",
    "value_too_long",
    "navigation_blocked",
    "browser_error",
    "environment_closed",
    "episode_finished",
}


@dataclass
class ElementDescriptor:
    """An interactive element visible to the agent."""
    element_id: str
    role: str          # link, button, textbox, searchbox, checkbox, combobox, spinbutton, textarea
    tag: str           # a, button, input, select, textarea
    name: str          # accessible name or label
    text: str          # visible text content
    value: str         # current value for inputs
    input_type: str    # for <input>: text, number, checkbox
    testid: str        # data-testid if present
    options: list      # for <select>: list of {value, label}
    disabled: bool


@dataclass
class Observation:
    """Textual observation of the current page state."""
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    task_id: str = ""
    episode_id: str = ""
    instruction: str = ""
    step_index: int = 0
    url: str = ""
    path: str = ""
    page_type: str = "unknown"
    title: str = ""
    visible_text: str = ""
    text_truncated: bool = False
    elements: list = field(default_factory=list)
    last_action_result: Optional[dict] = None
    terminal: bool = False

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "step_index": self.step_index,
            "url": self.url,
            "path": self.path,
            "page_type": self.page_type,
            "title": self.title,
            "visible_text": self.visible_text,
            "text_truncated": self.text_truncated,
            "elements": [{
                "element_id": e.element_id,
                "role": e.role,
                "tag": e.tag,
                "name": e.name,
                "text": e.text,
                "value": e.value,
                "input_type": e.input_type,
                "testid": e.testid,
                "options": e.options,
                "disabled": e.disabled,
            } for e in self.elements],
            "last_action_result": self.last_action_result,
            "terminal": self.terminal,
        }


@dataclass
class AgentAction:
    """A validated, typed action from the agent."""
    action: str
    target: str = ""
    value: str = ""
    checked: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: dict) -> "AgentAction":
        return cls(
            action=data.get("action", ""),
            target=data.get("target", ""),
            value=str(data.get("value", "")),
            checked=data.get("checked"),
        )

    def to_dict(self) -> dict:
        d = {"action": self.action}
        if self.target:
            d["target"] = self.target
        if self.value:
            d["value"] = self.value
        if self.checked is not None:
            d["checked"] = self.checked
        return d


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    error_code: str = ""
    message: str = ""
    page_changed: bool = False

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "error_code": self.error_code,
            "message": self.message,
            "page_changed": self.page_changed,
        }


@dataclass
class StepResult:
    """Returned by environment.step()."""
    observation: Optional[Observation] = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "observation": self.observation.to_dict() if self.observation else None,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }
