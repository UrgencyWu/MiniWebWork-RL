"""Minimal duck-typed episode schemas without a browser-runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ElementDescriptor:
    element_id: str
    role: str
    tag: str
    name: str
    text: str
    value: str
    input_type: str
    testid: str
    options: list[Any]
    disabled: bool


@dataclass
class Observation:
    schema_version: str = "m5-webshop-1.0"
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
    elements: list[ElementDescriptor] = field(default_factory=list)
    last_action_result: dict[str, Any] | None = None
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
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
            "elements": [vars(element).copy() for element in self.elements],
            "last_action_result": self.last_action_result,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class ActionResult:
    success: bool
    error_code: str = ""
    message: str = ""
    page_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass
class StepResult:
    observation: Observation | None = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
