"""Synchronous, leak-resistant HTTP adapter for the pinned WebShop server."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Mapping

import httpx

from ..m5_webshop_protocol import eligible_goal_indices, split_for_goal_index, task_id_for_goal_index
from .actions import WebShopCommand, normalize_command
from .prompt import MAX_OBSERVATION_CHARACTERS
from .schemas import ActionResult, ElementDescriptor, Observation, StepResult

TASK_ID_RE = re.compile(r"^webshop_goal_([0-9]{5})$")
MAX_PUBLIC_ACTIONS = 256
PAGE_TYPES = {"home", "search_results", "item", "subpage", "done"}
CONTROL_ACTIONS = {
    "click[Description]",
    "click[Features]",
    "click[Reviews]",
    "click[Buy Now]",
    "click[Back to Search]",
    "click[Back to Item]",
    "click[Next >]",
    "click[< Prev]",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=3)
def _eligible_roster(split: str) -> frozenset[int]:
    return frozenset(eligible_goal_indices(split))


@dataclass
class WebShopObservation(Observation):
    available_actions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["available_actions"] = list(self.available_actions)
        return payload


@dataclass
class WebShopTrajectoryState:
    verification: dict[str, Any] = field(default_factory=dict)


def goal_index_from_task_id(task_id: str) -> int:
    match = TASK_ID_RE.fullmatch(str(task_id))
    _require(match is not None, "invalid M5 WebShop task id")
    goal_index = int(match.group(1))
    _require(task_id_for_goal_index(goal_index) == task_id, "non-canonical M5 WebShop task id")
    return goal_index


def _bounded_public_actions(raw: Any) -> tuple[str, ...]:
    _require(isinstance(raw, list), "WebShop response lacks available_actions")
    values = []
    seen = set()
    for item in raw:
        _require(isinstance(item, str) and item.strip(), "WebShop available action is malformed")
        action = " ".join(item.strip().split())
        if action in seen:
            continue
        seen.add(action)
        values.append(action)
    if len(values) <= MAX_PUBLIC_ACTIONS:
        return tuple(values)
    essential = [value for value in values if value == "search[<your query>]" or value in CONTROL_ACTIONS]
    capacity = MAX_PUBLIC_ACTIONS - len(essential)
    _require(capacity >= 0, "WebShop control action set exceeds the public bound")
    selected_nonessential = [value for value in values if value not in essential][:capacity]
    selected = set(selected_nonessential) | set(essential)
    return tuple(value for value in values if value in selected)


def _element_for_action(action: str) -> ElementDescriptor:
    return ElementDescriptor(
        element_id=action,
        role="button",
        tag="button",
        name=action,
        text=action,
        value="",
        input_type="",
        testid="",
        options=[],
        disabled=False,
    )


def _public_page_fingerprint(observation: WebShopObservation) -> tuple[Any, ...]:
    """Compare environment page state without policy-history feedback fields."""

    return (
        observation.page_type,
        observation.visible_text,
        observation.text_truncated,
        observation.available_actions,
        observation.terminal,
    )


class WebShopHTTPEnvironment:
    """One stateful lane over Agent-R1's stateless WebShop HTTP API.

    Only the observation text, page type and bounded available-action list
    become policy-visible. Target ASINs and verifier internals returned by the
    upstream service are intentionally never copied into an observation.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:44151",
        split: str = "train",
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ):
        _require(split in {"train", "dev", "test"}, "invalid WebShop environment split")
        _require(timeout_seconds > 0, "invalid WebShop HTTP timeout")
        self.split = split
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds)
        self._owns_client = client is None
        self._goal_index: int | None = None
        self._task_id = ""
        self._instruction = ""
        self._env_state: dict[str, Any] = {}
        self._observation: WebShopObservation | None = None
        self._episode_id = ""
        self._step_index = 0
        self._finished = False
        self.trajectory: WebShopTrajectoryState | None = None

    def set_agent_name(self, _: str) -> None:
        return None

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = self._client.post(path, json=dict(payload))
        response.raise_for_status()
        body = response.json()
        _require(isinstance(body, dict), "WebShop HTTP response is not an object")
        return body

    def _make_observation(
        self,
        payload: Mapping[str, Any],
        *,
        last_action_result: dict[str, Any] | None,
        terminal: bool,
    ) -> WebShopObservation:
        raw_text = payload.get("observation")
        state = payload.get("env_state")
        info = payload.get("info")
        _require(isinstance(raw_text, str) and raw_text, "WebShop response lacks observation text")
        _require(isinstance(state, Mapping), "WebShop response lacks env_state")
        _require(isinstance(info, Mapping), "WebShop response lacks info")
        page_type = str(state.get("page_type") or "unknown")
        _require(page_type in PAGE_TYPES, "WebShop page type drift")
        actions = _bounded_public_actions(info.get("available_actions"))
        text_truncated = len(raw_text) > MAX_OBSERVATION_CHARACTERS
        text = raw_text[:MAX_OBSERVATION_CHARACTERS]
        title = text.splitlines()[0][:300] if text else "WebShop"
        return WebShopObservation(
            task_id=self._task_id,
            episode_id=self._episode_id,
            instruction=self._instruction,
            step_index=self._step_index,
            url=f"/webshop/{page_type}",
            path=f"/webshop/{page_type}",
            page_type=page_type,
            title=title,
            visible_text=text,
            text_truncated=text_truncated,
            elements=[_element_for_action(action) for action in actions],
            last_action_result=last_action_result,
            terminal=terminal,
            available_actions=actions,
        )

    def reset(self, task_id: str) -> WebShopObservation:
        goal_index = goal_index_from_task_id(task_id)
        _require(split_for_goal_index(goal_index) == self.split, "WebShop task/split mismatch")
        _require(goal_index in _eligible_roster(self.split), "WebShop task is excluded by the split lock")
        self._goal_index = goal_index
        self._task_id = task_id
        self._episode_id = uuid.uuid4().hex
        self._step_index = 0
        self._finished = False
        self.trajectory = WebShopTrajectoryState()
        payload = self._post("/reset", {"goal_index": goal_index})
        info = payload.get("info")
        _require(isinstance(info, Mapping), "WebShop reset lacks info")
        instruction = info.get("instruction")
        _require(isinstance(instruction, str) and instruction.strip(), "WebShop reset lacks instruction")
        self._instruction = instruction
        state = payload.get("env_state")
        _require(isinstance(state, Mapping), "WebShop reset lacks env_state")
        self._env_state = dict(state)
        self._observation = self._make_observation(payload, last_action_result=None, terminal=False)
        return self._observation

    @staticmethod
    def _is_publicly_executable(command: str, actions: tuple[str, ...]) -> bool:
        if command.lower().startswith("search["):
            return "search[<your query>]" in actions
        return command in actions

    def step(self, action: WebShopCommand) -> StepResult:
        _require(self._goal_index is not None and self._observation is not None, "WebShop environment was not reset")
        _require(not self._finished, "WebShop episode is already finished")
        _require(isinstance(action, WebShopCommand), "WebShop environment received the wrong action type")
        command = normalize_command(action.command)
        self._step_index += 1
        if not self._is_publicly_executable(command, self._observation.available_actions):
            action_result = ActionResult(
                success=False,
                error_code="click_target_not_public",
                message="the command was not in the bounded public action list",
                page_changed=False,
            )
            self._observation = replace(
                self._observation,
                step_index=self._step_index,
                last_action_result=action_result.to_dict(),
            )
            return StepResult(
                observation=self._observation,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"action_result": action_result.to_dict(), "policy_error": "click_target_not_public"},
            )

        previous_observation = self._observation
        payload = self._post(
            "/step",
            {"goal_index": self._goal_index, "env_state": self._env_state, "action": command},
        )
        state = payload.get("env_state")
        info = payload.get("info")
        _require(isinstance(state, Mapping), "WebShop step lacks env_state")
        _require(isinstance(info, Mapping), "WebShop step lacks info")
        self._env_state = dict(state)
        done = payload.get("done")
        upstream_reward = payload.get("reward")
        _require(isinstance(done, bool), "WebShop done flag is invalid")
        _require(
            isinstance(upstream_reward, (int, float))
            and not isinstance(upstream_reward, bool)
            and 0.0 <= float(upstream_reward) <= 1.0,
            "WebShop upstream score drift",
        )
        upstream_score = float(upstream_reward)
        _require(done or upstream_score == 0.0, "WebShop emitted reward before termination")
        upstream_error = str(info.get("error") or "")
        _require(not upstream_error, "WebShop rejected a command that the prior public state listed as executable")
        provisional_observation = self._make_observation(
            payload,
            last_action_result=None,
            terminal=done,
        )
        page_changed = _public_page_fingerprint(previous_observation) != _public_page_fingerprint(
            provisional_observation
        )
        action_result = ActionResult(
            success=True,
            error_code="",
            message="",
            page_changed=page_changed,
        )
        self._observation = replace(
            provisional_observation,
            last_action_result=action_result.to_dict(),
        )
        self._finished = done
        task_score_raw = info.get("task_score", upstream_score)
        _require(
            isinstance(task_score_raw, (int, float))
            and not isinstance(task_score_raw, bool)
            and 0.0 <= float(task_score_raw) <= 1.0,
            "WebShop dense task score drift",
        )
        success_raw = info.get("success", float(task_score_raw) >= 0.999)
        _require(isinstance(success_raw, bool), "WebShop success flag drift")
        _require(success_raw == (float(task_score_raw) >= 0.999), "WebShop success/task-score disagreement")
        _require(
            abs(upstream_score - float(task_score_raw)) <= 1e-9
            or abs(upstream_score - float(success_raw)) <= 1e-9,
            "WebShop upstream reward/task-score disagreement",
        )
        _require(done == (self._observation.page_type == "done"), "WebShop terminal page drift")
        safe_info = {
            "action_result": action_result.to_dict(),
            "task_score": float(task_score_raw),
            "success": success_raw,
            "selected_asin": str(info.get("selected_asin") or "")[:20],
            "termination_reason": "purchase" if done else "continue",
        }
        if done and self.trajectory is not None:
            self.trajectory.verification = {
                "success": safe_info["success"],
                "task_score": safe_info["task_score"],
                "selected_asin": safe_info["selected_asin"],
                "failure_reasons": [] if safe_info["success"] else ["webshop_task_score_below_one"],
            }
        return StepResult(
            observation=self._observation,
            reward=float(success_raw),
            terminated=done,
            truncated=False,
            info=safe_info,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
