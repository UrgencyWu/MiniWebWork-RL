"""Versioned schemas for Observation, Action, StepResult, and Trajectory."""

from dataclasses import dataclass, field
from typing import Optional

OBSERVATION_SCHEMA_VERSION = "1.0"
ACTION_SCHEMA_VERSION = "1.1"
TRAJECTORY_SCHEMA_VERSION = "1.0"

VALID_PAGE_TYPES = {
    "home",
    "task",
    "products",
    "product_detail",
    "supplier_detail",
    "procurement_form",
    "procurement_result",
    "smoke",
    "error",
    "unknown",
}

VALID_ACTION_TYPES = {"click", "fill", "select", "check", "back", "submit", "finish"}

# `submit` targets the same visible button role as `click`, but remains a
# distinct semantic action for trajectory analysis and prompting.
ROLE_ACTION_COMPAT = {
    "link": {"click"},
    "button": {"click", "submit"},
    "textbox": {"fill"},
    "searchbox": {"fill"},
    "checkbox": {"check"},
    "combobox": {"select"},
    "spinbutton": {"fill"},
    "textarea": {"fill"},
}

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
    """An interactive element visible to the Agent."""

    element_id: str
    role: str
    tag: str
    name: str
    text: str
    value: str
    input_type: str
    testid: str
    options: list
    disabled: bool


@dataclass
class Observation:
    """Textual observation of the current browser state."""

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
            "elements": [
                {
                    "element_id": element.element_id,
                    "role": element.role,
                    "tag": element.tag,
                    "name": element.name,
                    "text": element.text,
                    "value": element.value,
                    "input_type": element.input_type,
                    "testid": element.testid,
                    "options": element.options,
                    "disabled": element.disabled,
                }
                for element in self.elements
            ],
            "last_action_result": self.last_action_result,
            "terminal": self.terminal,
        }


@dataclass
class AgentAction:
    """A typed action emitted by the Agent."""

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
        payload = {"action": self.action}
        if self.target:
            payload["target"] = self.target
        if self.value:
            payload["value"] = self.value
        if self.checked is not None:
            payload["checked"] = self.checked
        return payload


@dataclass
class ActionResult:
    """Deterministic validation or browser execution result."""

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
    """Result returned by `ProcurementBrowserEnv.step`."""

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
