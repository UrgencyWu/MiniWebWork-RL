"""Small public-input, verifier-checked corpora for Phase10-C SFT Specialists."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from .long_horizon_rl.contracts import sha256_json
from .webshop_rl import prompt
from .webshop_rl.actions import MAX_SEARCH_QUERY_CHARACTERS, WebShopCommand, parse_command_output


CORPUS_SCHEMA = "m6_phase10c_specialist_smoke_corpus_v1"
PUBLIC_QUERY_FORMULA = "public_instruction_candidate_search_v3"
QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "for", "of", "with", "that", "are", "is", "some", "and", "my", "to", "in", "on",
    "high", "quality", "performance", "power", "definition", "looking", "find", "want", "would", "like",
})
ASIN_RE = re.compile(r"^B[0-9A-Z]{9}$", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SPECIALIST_FAMILIES = {
    "S_nav_sft": frozenset({"search", "navigation", "candidate"}),
    "S_match_sft": frozenset({"candidate", "option"}),
    "S_finish_sft": frozenset({"option", "buy"}),
}


class SpecialistDataFailure(ValueError):
    """A frozen smoke task could not produce a learnable verified trajectory."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _click_argument(command: str) -> str | None:
    if not command.casefold().startswith("click[") or not command.endswith("]"):
        return None
    return command[command.find("[") + 1 : -1]


def public_instruction_query(instruction: str) -> str:
    """Create a deterministic search query using instruction tokens only."""

    public_text = str(instruction or "").strip()
    public_text = re.split(r",?\s+and price lower than\b", public_text, maxsplit=1, flags=re.IGNORECASE)[0]
    public_text = re.split(r"\.\s+", public_text, maxsplit=1)[0]
    public_text = re.sub(
        r"^(?:i am looking for|i['’]m looking for|looking for|i would like|i want to find|i want)\s+",
        "",
        public_text,
        count=1,
        flags=re.IGNORECASE,
    )
    tokens = [match.group(0) for match in WORD_RE.finditer(public_text)]
    _require(tokens, "Phase10-C instruction has no public query tokens")
    retained: list[str] = []
    for token in tokens:
        candidate = " ".join([*retained, token])
        if len(candidate) > MAX_SEARCH_QUERY_CHARACTERS:
            break
        retained.append(token)
    query = " ".join(retained).strip()
    _require(query, "Phase10-C public instruction query is empty")
    source = {token.casefold() for token in tokens}
    _require(
        {token.casefold() for token in WORD_RE.findall(query)} <= source,
        "Phase10-C query introduced a non-instruction token",
    )
    return query


def public_instruction_query_candidates(instruction: str) -> tuple[str, ...]:
    """Return at most three fixed candidates, all derived from public tokens."""

    base = public_instruction_query(instruction)
    tokens = WORD_RE.findall(base)
    compact_tokens = [token for token in tokens if token.casefold() not in QUERY_STOPWORDS]
    compact = " ".join(compact_tokens[:12]).strip()
    core = " ".join(compact_tokens[:6]).strip()
    candidates = tuple(dict.fromkeys(value for value in (base, compact, core) if value))
    _require(candidates, "Phase10-C public query candidate set is empty")
    source_tokens = {token.casefold() for token in WORD_RE.findall(str(instruction or ""))}
    for candidate in candidates:
        _require(
            {token.casefold() for token in WORD_RE.findall(candidate)} <= source_tokens,
            "Phase10-C query candidate introduced a non-instruction token",
        )
    return candidates


def action_family(command: str) -> str:
    normalized = _normalize(command)
    if normalized.startswith("search["):
        return "search"
    argument = _click_argument(command)
    _require(argument is not None, "Phase10-C oracle emitted a non-public command")
    value = _normalize(argument)
    if value == "buy now":
        return "buy"
    if value in {"next >", "< prev", "back to search", "back to item", "description", "features", "reviews"}:
        return "navigation"
    if ASIN_RE.fullmatch(argument) is not None:
        return "candidate"
    return "option"


def _goal_options(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, Mapping):
        values = raw.values()
    elif isinstance(raw, list):
        values = [item.get("value") if isinstance(item, Mapping) else item for item in raw]
    else:
        values = []
    result = tuple(str(value).strip() for value in values if str(value or "").strip())
    _require(len({_normalize(value) for value in result}) == len(result), "Phase10-C duplicate goal option")
    return result


def build_verified_public_query_trajectory(environment: Any, goal: Mapping[str, Any]) -> dict[str, Any]:
    """Use hidden metadata only to choose among actions already public on-page."""

    goal_index = goal.get("goal_index")
    _require(isinstance(goal_index, int) and not isinstance(goal_index, bool), "Phase10-C goal index drift")
    task_id = f"webshop_goal_{goal_index:05d}"
    observation = environment.reset(task_id)
    history: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []

    def execute(command: str) -> Any:
        nonlocal observation
        if len(turns) >= 15:
            raise SpecialistDataFailure("environment_step_budget_exhausted")
        available = tuple(str(value) for value in observation.available_actions)
        if not command.casefold().startswith("search[") and command not in available:
            raise SpecialistDataFailure("label_not_in_public_available_actions")
        messages = prompt.build_messages(observation, history)
        completion = json.dumps({"command": command}, ensure_ascii=False, separators=(",", ":"))
        parsed = parse_command_output(completion)
        _require(parsed.strict_json_success and parsed.schema_valid, "Phase10-C canonical label drift")
        result = environment.step(WebShopCommand(command))
        action_result = result.info.get("action_result", {})
        if not action_result.get("success", False):
            raise SpecialistDataFailure(f"public_action_rejected:{action_result.get('error_code', 'unknown')}")
        turns.append({
            "task_id": task_id,
            "turn_index": len(turns) + 1,
            "messages": messages,
            "prompt_sha256": prompt.compute_message_hash(messages),
            "completion": completion,
            "command": command,
            "action_family": action_family(command),
            "public_available_action_sha256": sha256_json(list(available)),
        })
        history.append({
            "turn": len(turns),
            "command": command,
            "success": True,
            "error_code": "",
            "page_type": result.observation.page_type if result.observation is not None else "done",
        })
        if result.observation is not None:
            observation = result.observation
        return result

    target_asin = str(goal.get("asin") or "").strip()
    if not target_asin:
        raise SpecialistDataFailure("missing_offline_verifier_target")

    def target_is_public_for_query(query: str) -> bool:
        probe_observation = environment.reset(task_id)
        probe_result = environment.step(WebShopCommand(f"search[{query}]"))
        if not probe_result.info.get("action_result", {}).get("success", False):
            return False
        if probe_result.observation is not None:
            probe_observation = probe_result.observation
        for page_index in range(5):
            if any(
                _normalize(_click_argument(str(action))) == _normalize(target_asin)
                for action in probe_observation.available_actions
            ):
                return True
            if page_index >= 4 or "click[Next >]" not in probe_observation.available_actions:
                break
            probe_result = environment.step(WebShopCommand("click[Next >]"))
            if not probe_result.info.get("action_result", {}).get("success", False):
                break
            if probe_result.observation is not None:
                probe_observation = probe_result.observation
        return False

    query_candidates = public_instruction_query_candidates(str(goal.get("instruction") or ""))
    query = next((candidate for candidate in query_candidates if target_is_public_for_query(candidate)), "")
    if not query:
        raise SpecialistDataFailure("target_not_in_public_top50_for_instruction_query_candidates")
    observation = environment.reset(task_id)
    history.clear()
    turns.clear()
    execute(f"search[{query}]")
    target_action = ""
    for page_index in range(5):
        candidates = [
            action for action in observation.available_actions
            if _normalize(_click_argument(str(action))) == _normalize(target_asin)
        ]
        if len(candidates) == 1:
            target_action = str(candidates[0])
            break
        if len(candidates) > 1:
            raise SpecialistDataFailure("ambiguous_public_target_action")
        if page_index < 4 and "click[Next >]" in observation.available_actions:
            execute("click[Next >]")
            continue
        break
    if not target_action:
        raise SpecialistDataFailure("target_not_in_public_top50_for_instruction_query")
    if execute(target_action).terminated:
        raise SpecialistDataFailure("target_click_terminated_early")

    for desired in _goal_options(goal.get("goal_options")):
        candidates = [
            str(action) for action in observation.available_actions
            if _normalize(_click_argument(str(action))) == _normalize(desired)
        ]
        if len(candidates) != 1:
            raise SpecialistDataFailure("goal_option_not_uniquely_public")
        if execute(candidates[0]).terminated:
            raise SpecialistDataFailure("option_click_terminated_early")
    if "click[Buy Now]" not in observation.available_actions:
        raise SpecialistDataFailure("buy_now_not_public")
    terminal = execute("click[Buy Now]")
    score = float(terminal.info.get("task_score", 0.0))
    if (
        not terminal.terminated
        or float(terminal.reward) < 0.999
        or terminal.info.get("success") is not True
        or score < 0.999
    ):
        raise SpecialistDataFailure("strict_replay_failed")
    result = {
        "schema_version": "m6_phase10c_public_query_oracle_trajectory_v1",
        "task_id": task_id,
        "goal_index": goal_index,
        "strict_success": True,
        "task_score": score,
        "query_tokens_from_public_instruction_only": True,
        "public_query_formula": PUBLIC_QUERY_FORMULA,
        "public_query_candidate_count": len(query_candidates),
        "selected_public_query_rank": query_candidates.index(query) + 1,
        "offline_verifier_metadata_used_only_for_public_action_selection": True,
        "policy_input_contains_hidden_metadata": False,
        "turns": turns,
    }
    result["content_sha256"] = sha256_json(result)
    return result


def build_specialist_smoke_corpus(
    *,
    specialist: str,
    goals: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str],
    environment_factory: Callable[[], Any],
    phase10c_split_content_sha256: str,
    producer_git_sha: str,
    minimum_verified_tasks: int = 12,
) -> dict[str, Any]:
    _require(specialist in SPECIALIST_FAMILIES, "Phase10-C specialist identity drift")
    _require(len(task_ids) == len(set(task_ids)) == 16, "Phase10-C smoke task count drift")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    rows: list[dict[str, Any]] = []
    verified_tasks: list[str] = []
    rejection_counts: Counter[str] = Counter()
    full_trajectory_hashes: dict[str, str] = {}
    family_counts: Counter[str] = Counter()
    for task_id in task_ids:
        _require(task_id in goal_map, "Phase10-C smoke task missing from goals")
        environment = environment_factory()
        try:
            try:
                trajectory = build_verified_public_query_trajectory(environment, goal_map[task_id])
            except SpecialistDataFailure as exc:
                rejection_counts[str(exc)] += 1
                continue
        finally:
            environment.close()
        verified_tasks.append(task_id)
        full_trajectory_hashes[task_id] = trajectory["content_sha256"]
        for turn in trajectory["turns"]:
            if turn["action_family"] not in SPECIALIST_FAMILIES[specialist]:
                continue
            row = {
                "schema_version": "m6_phase10c_specialist_sft_row_v1",
                "specialist": specialist,
                "task_id": task_id,
                "turn_index": turn["turn_index"],
                "messages": turn["messages"],
                "prompt_sha256": turn["prompt_sha256"],
                "completion": turn["completion"],
                "command": turn["command"],
                "action_family": turn["action_family"],
                "source_trajectory_content_sha256": trajectory["content_sha256"],
                "policy_input_contains_hidden_metadata": False,
            }
            row["content_sha256"] = sha256_json(row)
            rows.append(row)
            family_counts[turn["action_family"]] += 1
    passed = len(verified_tasks) >= minimum_verified_tasks and bool(rows)
    result = {
        "schema_version": CORPUS_SCHEMA,
        "development_only": True,
        "specialist": specialist,
        "phase10c_split_content_sha256": phase10c_split_content_sha256,
        "producer_git_sha": producer_git_sha,
        "requested_task_count": len(task_ids),
        "requested_task_ids": list(task_ids),
        "requested_task_order_sha256": sha256_json(list(task_ids)),
        "verified_task_count": len(verified_tasks),
        "verified_task_ids": verified_tasks,
        "verified_task_id_sha256": sha256_json(verified_tasks),
        "minimum_verified_tasks": minimum_verified_tasks,
        "passed": passed,
        "decision": "ready_for_single_update_probe" if passed else "stop_specialist_data_method",
        "label_row_count": len(rows),
        "label_action_family_counts": dict(sorted(family_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "source_trajectory_content_sha256": dict(sorted(full_trajectory_hashes.items())),
        "query_tokens_from_public_instruction_only": True,
        "public_query_formula": PUBLIC_QUERY_FORMULA,
        "maximum_public_query_candidates": 3,
        "offline_verifier_metadata_used_only_for_public_action_selection": True,
        "policy_input_contains_hidden_metadata": False,
        "fresh_session_strict_replay_required": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "rows": rows,
    }
    result["content_sha256"] = sha256_json(result)
    return result
