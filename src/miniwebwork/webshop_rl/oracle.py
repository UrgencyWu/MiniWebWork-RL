"""Public-action-valid WebShop expert used only to build verified SFT labels."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..long_horizon_rl.contracts import sha256_json
from ..m5_webshop_protocol import split_for_goal_index, task_id_for_goal_index
from . import prompt
from .actions import MAX_SEARCH_QUERY_CHARACTERS, WebShopCommand
from .credit import public_state_anchor_signature


class OraclePolicyFailure(ValueError):
    """A pinned goal cannot be solved through the bounded public action API."""


MAX_ORACLE_TURNS = 15


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _click_argument(command: str) -> str | None:
    if not command.lower().startswith("click[") or not command.endswith("]"):
        return None
    return command[command.find("[") + 1 : -1]


def _goal_option_values(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, Mapping):
        values = raw.values()
    elif isinstance(raw, list):
        values = [item.get("value") if isinstance(item, Mapping) else item for item in raw]
    else:
        values = []
    normalized = tuple(str(value).strip() for value in values if str(value or "").strip())
    if len(set(map(_normalize, normalized))) != len(normalized):
        raise OraclePolicyFailure("duplicate_goal_option")
    return normalized


def _oracle_title_query(goal: Mapping[str, Any]) -> str:
    """Build a deterministic public search action from offline teacher metadata.

    The exact product title is available only to the offline SFT teacher. The
    resulting action still goes through WebShop's normal public search API; the
    target ASIN is removed before the action is emitted and is never inserted
    into a prompt.
    """

    title = str(goal.get("name") or "")
    target_asin = str(goal.get("asin") or "").strip()
    if not title.strip() or not target_asin:
        raise OraclePolicyFailure("missing_oracle_metadata")
    printable = "".join(character if character.isprintable() else " " for character in title)
    without_delimiters = re.sub(r"[\[\]]+", " ", printable)
    without_asin = re.sub(re.escape(target_asin), " ", without_delimiters, flags=re.IGNORECASE)
    query = " ".join(without_asin.split())[:MAX_SEARCH_QUERY_CHARACTERS].strip()
    if not query or target_asin.lower() in query.lower():
        raise OraclePolicyFailure("unsafe_oracle_title_query")
    return query


@dataclass
class VerifiedOracleTrajectory:
    task_id: str
    split: str
    goal_index: int
    turns: list[dict[str, Any]]
    task_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "m5_webshop_verified_oracle_trajectory_v1",
            "task_id": self.task_id,
            "split": self.split,
            "goal_index": self.goal_index,
            "turn_count": len(self.turns),
            "task_score": self.task_score,
            "verified_reward": 1.0,
            "turns": self.turns,
        }
        payload["content_sha256"] = sha256_json(payload)
        return payload


def build_verified_oracle_trajectory(environment: Any, goal: Mapping[str, Any]) -> VerifiedOracleTrajectory:
    goal_index = goal.get("goal_index")
    if not isinstance(goal_index, int) or isinstance(goal_index, bool):
        raise OraclePolicyFailure("invalid_goal_index")
    split = split_for_goal_index(goal_index)
    if split == "test":
        raise OraclePolicyFailure("frozen_test_forbidden")
    task_id = task_id_for_goal_index(goal_index)
    observation = environment.reset(task_id)
    history: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []

    def execute(command: str):
        nonlocal observation
        if len(turns) >= MAX_ORACLE_TURNS:
            raise OraclePolicyFailure("oracle_turn_budget_exhausted")
        messages = prompt.build_messages(observation, history)
        completion = json.dumps({"command": command}, ensure_ascii=False, separators=(",", ":"))
        result = environment.step(WebShopCommand(command))
        action_result = result.info.get("action_result", {})
        if not action_result.get("success", False):
            raise OraclePolicyFailure(f"public_action_rejected:{action_result.get('error_code', 'unknown')}")
        turns.append(
            {
                "schema_version": "m5_webshop_verified_sft_turn_v1",
                "task_id": task_id,
                "split": split,
                "goal_index": goal_index,
                "turn_index": len(turns) + 1,
                "messages": messages,
                "prompt_sha256": prompt.compute_message_hash(messages),
                "completion": completion,
                "command": command,
                "public_state_anchor_sha256": public_state_anchor_signature(observation.to_dict()),
            }
        )
        history.append(
            {
                "turn": len(turns),
                "command": command,
                "success": True,
                "error_code": "",
                "page_type": result.observation.page_type if result.observation is not None else "unknown",
            }
        )
        if result.observation is not None:
            observation = result.observation
        return result

    query = _oracle_title_query(goal)
    target_asin = str(goal.get("asin") or "").strip()
    execute(f"search[{query}]")

    target_action = ""
    for page_index in range(5):
        candidates = [
            action
            for action in observation.available_actions
            if (_click_argument(action) or "").upper() == target_asin.upper()
        ]
        if len(candidates) == 1:
            target_action = candidates[0]
            break
        if len(candidates) > 1:
            raise OraclePolicyFailure("ambiguous_target_action")
        if page_index < 4 and "click[Next >]" in observation.available_actions:
            execute("click[Next >]")
            continue
        break
    if not target_action:
        raise OraclePolicyFailure("target_asin_not_in_public_top50")
    target_result = execute(target_action)
    if target_result.terminated:
        raise OraclePolicyFailure("target_click_terminated_early")

    for desired in _goal_option_values(goal.get("goal_options")):
        candidates = []
        for action in observation.available_actions:
            argument = _click_argument(action)
            if argument is not None and _normalize(argument) == _normalize(desired):
                candidates.append(action)
        if len(candidates) != 1:
            raise OraclePolicyFailure("goal_option_not_uniquely_public")
        option_result = execute(candidates[0])
        if option_result.terminated:
            raise OraclePolicyFailure("option_click_terminated_early")

    if "click[Buy Now]" not in observation.available_actions:
        raise OraclePolicyFailure("buy_now_not_public")
    terminal = execute("click[Buy Now]")
    if not terminal.terminated or terminal.reward != 1.0 or terminal.info.get("success") is not True:
        raise OraclePolicyFailure("oracle_purchase_not_verified")
    task_score = float(terminal.info.get("task_score", 0.0))
    if task_score < 0.999:
        raise OraclePolicyFailure("oracle_dense_score_below_one")
    return VerifiedOracleTrajectory(
        task_id=task_id,
        split=split,
        goal_index=goal_index,
        turns=turns,
        task_score=task_score,
    )
