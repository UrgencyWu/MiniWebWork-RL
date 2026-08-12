"""Policy-visible M6 SFT corpus construction and conditional-learnability audit."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from ..long_horizon_rl.contracts import sha256_json
from ..m6_posttraining_protocol import load_protocol, validate_protocol
from . import prompt
from .actions import COMMAND_RE, parse_command_output

CORPUS_SCHEMA = "m6_policy_visible_sft_corpus_v1"
AUDIT_SCHEMA = "m6_conditional_learnability_audit_v1"
RETENTION_SCHEMA = "m6_raw_retention_states_v1"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
ASIN_RE = re.compile(r"^B[0-9A-Z]{9}$", re.IGNORECASE)
FORBIDDEN_POLICY_FIELD_RE = re.compile(
    r"(?:target[_ -]?asin|goal\.name|goal[_ -]?name|oracle[_ -]?(?:answer|title|query)|"
    r"expected[_ -]?answer|verifier[_ -]?(?:target|answer|score))\s*[:=]",
    re.IGNORECASE,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tokens(value: Any) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN_RE.finditer(str(value or "")))


def _command(turn: Mapping[str, Any]) -> str:
    raw = turn.get("command")
    if not isinstance(raw, str):
        action = turn.get("action")
        raw = action.get("command") if isinstance(action, Mapping) else ""
    _require(isinstance(raw, str) and COMMAND_RE.fullmatch(raw.strip()) is not None, "M6 corpus command is invalid")
    return " ".join(raw.strip().split())


def _search_query(command: str) -> str | None:
    match = COMMAND_RE.fullmatch(command)
    _require(match is not None, "M6 corpus command is malformed")
    return match.group(2).strip() if match.group(1).casefold() == "search" else None


def _click_argument(command: str) -> str:
    match = COMMAND_RE.fullmatch(command)
    _require(match is not None, "M6 corpus command is malformed")
    return match.group(2).strip() if match.group(1).casefold() == "click" else ""


def action_family(command: str) -> str:
    query = _search_query(command)
    if query is not None:
        return "search"
    argument = _click_argument(command)
    normalized = argument.casefold()
    if normalized == "buy now":
        return "buy"
    if normalized in {"next >", "< prev", "back to search", "back to item"}:
        return "navigation"
    if normalized in {"description", "features", "reviews"}:
        return "information_tab"
    if ASIN_RE.fullmatch(argument) is not None:
        return "candidate"
    return "option"


def _is_recovery(commands: Sequence[str]) -> bool:
    searches = {_search_query(command).casefold() for command in commands if _search_query(command) is not None}
    arguments = [_click_argument(command) for command in commands]
    products = {argument.casefold() for argument in arguments if ASIN_RE.fullmatch(argument) is not None}
    navigation_recovery = any(
        argument.casefold() in {"< prev", "back to search", "back to item"}
        for argument in arguments
    )
    return len(searches) >= 2 or len(products) >= 2 or navigation_recovery


def _policy_payload_text(turn: Mapping[str, Any]) -> str:
    messages = turn.get("messages")
    completion = turn.get("completion")
    _require(isinstance(messages, list) and messages, "M6 corpus turn lacks policy messages")
    _require(isinstance(completion, str) and completion.strip(), "M6 corpus turn has a zero label")
    return json.dumps(
        {"messages": messages, "completion": completion},
        ensure_ascii=False,
        sort_keys=True,
    )


def _public_query_source_from_messages(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    users = [str(message.get("content", "")) for message in messages if message.get("role") == "user"]
    _require(len(users) == 1, "M6 corpus prompt must contain exactly one user message")
    content = users[0]
    instruction_match = re.search(
        r"\ninstruction: (.*?)\n\n## Current public state\n",
        content,
        flags=re.DOTALL,
    )
    visible_match = re.search(
        r"\nobservation_truncated: (?:true|false)\n(.*?)\n\n## Executable actions",
        content,
        flags=re.DOTALL,
    )
    _require(instruction_match is not None and visible_match is not None, "M6 public prompt sections drifted")
    return sorted(
        set(_tokens(instruction_match.group(1))) | set(_tokens(visible_match.group(1)))
    )


def _observation_from_turn(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = turn.get("observation")
    _require(isinstance(observation, Mapping), "M6 source turn lacks public observation")
    required = ("task_id", "instruction", "page_type", "visible_text", "available_actions")
    _require(all(field in observation for field in required), "M6 source observation is incomplete")
    return observation


def _messages_from_source_turns(
    turns: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    rows: list[dict[str, Any]] = []
    selected_commands: list[str] = []
    history: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(turns, start=1):
        observation = dict(_observation_from_turn(turn))
        observation.setdefault("step_index", turn_index - 1)
        observation.setdefault("text_truncated", False)
        namespace = SimpleNamespace(**observation)
        messages = prompt.build_messages(namespace, history)
        observed_prompt_hash = turn.get("rendered_prompt_sha256")
        if isinstance(observed_prompt_hash, str) and observed_prompt_hash:
            _require(
                observed_prompt_hash == prompt.compute_message_hash(messages),
                "M6 source prompt cannot be reproduced from public state/history",
            )
        action = turn.get("action")
        if not isinstance(action, Mapping) or turn.get("schema_valid") is not True:
            continue
        command = _command(turn)
        public_query_source_tokens = _public_query_source_from_messages(messages)
        action_result = turn.get("action_result")
        action_success = bool(action_result.get("success", False)) if isinstance(action_result, Mapping) else False
        if action_success:
            completion = json.dumps({"command": command}, ensure_ascii=False, separators=(",", ":"))
            parsed = parse_command_output(completion)
            _require(parsed.strict_json_success and parsed.schema_valid, "M6 canonical completion is invalid")
            rows.append(
                {
                    "schema_version": "m6_policy_visible_sft_turn_v1",
                    "turn_index": turn_index,
                    "messages": messages,
                    "prompt_sha256": prompt.compute_message_hash(messages),
                    "completion": completion,
                    "command": command,
                    "completion_truncated": False,
                    "source_public_state_sha256": sha256_json(observation),
                    "source_raw_output_sha256": sha256_json({"raw_output": str(turn.get("raw_output", ""))}),
                    "public_query_source_tokens": public_query_source_tokens,
                    "public_query_source_tokens_sha256": sha256_json(public_query_source_tokens),
                }
            )
            selected_commands.append(command)
        next_page_type = "done" if turn.get("terminated") is True else str(observation.get("page_type", ""))
        if turn_index < len(turns):
            next_observation = turns[turn_index].get("observation")
            if isinstance(next_observation, Mapping):
                next_page_type = str(next_observation.get("page_type", next_page_type))
        history.append(
            {
                "turn": turn_index,
                "command": command,
                "success": action_success,
                "error_code": str(action_result.get("error_code", "")) if isinstance(action_result, Mapping) else "",
                "page_type": next_page_type,
            }
        )
    _require(rows, "M6 strict-success trajectory has no replayable policy-visible labels")
    return rows, tuple(selected_commands)


def build_policy_visible_corpus(
    *,
    trajectories: Sequence[Mapping[str, Any]],
    goal_by_task_id: Mapping[str, Mapping[str, Any]],
    completion_token_counter: Callable[[str], int],
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build de-duplicated action-turn labels from replayed strict successes.

    This function never consults goal metadata while constructing messages or
    completions.  Goal metadata is passed only to the subsequent leakage audit.
    """

    contract = dict(protocol or load_protocol()["payload"])
    validate_protocol(contract)
    maximum_per_task = int(contract["corpus"]["maximum_trajectories_per_task"])
    selected: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    seen_sequences: set[tuple[str, tuple[str, ...]]] = set()
    for source in trajectories:
        _require(isinstance(source, Mapping), "M6 source trajectory is malformed")
        task_id = source.get("task_id")
        _require(isinstance(task_id, str) and task_id in goal_by_task_id, "M6 source task identity is unknown")
        score = source.get("task_score", source.get("reward"))
        _require(
            isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)),
            "M6 source task score is invalid",
        )
        if source.get("success") is not True or float(score) < 0.999 or source.get("replay_success") is not True:
            continue
        source_turns = source.get("turns")
        _require(isinstance(source_turns, list) and source_turns, "M6 source trajectory has no turns")
        rows, commands = _messages_from_source_turns(source_turns)
        sequence_key = (task_id, tuple(command.casefold() for command in commands))
        if sequence_key in seen_sequences or task_counts[task_id] >= maximum_per_task:
            continue
        completion_tokens = 0
        for row in rows:
            count = completion_token_counter(str(row["completion"]))
            _require(isinstance(count, int) and not isinstance(count, bool) and count > 0, "M6 completion token count is invalid")
            row["completion_label_tokens"] = count
            completion_tokens += count
        selected.append(
            {
                "schema_version": "m6_policy_visible_sft_trajectory_v1",
                "trajectory_id": str(source.get("trajectory_id", "")),
                "task_id": task_id,
                "strict_success": True,
                "replay_success": True,
                "command_sequence_sha256": sha256_json(list(commands)),
                "recovery": _is_recovery(commands),
                "completion_label_tokens": completion_tokens,
                "turns": rows,
            }
        )
        seen_sequences.add(sequence_key)
        task_counts[task_id] += 1

    payload = {
        "schema_version": CORPUS_SCHEMA,
        "study_id": contract["study_id"],
        "development_only": True,
        "hidden_metadata_used_for_label_construction": False,
        "trajectory_count": len(selected),
        "task_count": len(task_counts),
        "completion_label_tokens": sum(item["completion_label_tokens"] for item in selected),
        "trajectories": selected,
    }
    payload["content_sha256"] = sha256_json(payload)
    return payload


def build_retention_states(
    *,
    trajectories: Sequence[Mapping[str, Any]],
    maximum_states: int,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze policy-visible Raw actions for KL-only capability retention.

    These rows deliberately have no supervised label.  The generated Raw
    completion supplies only the action-token positions at which the current
    SFT policy is compared with the adapter-disabled Raw reference.
    """

    contract = dict(protocol or load_protocol()["payload"])
    validate_protocol(contract)
    _require(maximum_states > 0, "M6 retention-state cap must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for trajectory in trajectories:
        _require(isinstance(trajectory, Mapping), "M6 retention trajectory is malformed")
        _require(trajectory.get("rollout_valid") is True, "M6 retention source is infrastructure-invalid")
        task_id = trajectory.get("task_id")
        turns = trajectory.get("turns")
        _require(isinstance(task_id, str) and task_id, "M6 retention task id is missing")
        _require(isinstance(turns, list) and turns, "M6 retention trajectory has no turns")
        valid_commands = [
            _command(item)
            for item in turns
            if isinstance(item, Mapping)
            and item.get("schema_valid") is True
            and isinstance(item.get("action"), Mapping)
        ]
        source_recovery = _is_recovery(valid_commands)
        for turn_index, turn in enumerate(turns, start=1):
            _require(isinstance(turn, Mapping), "M6 retention turn is malformed")
            prompt_ids = turn.get("prompt_token_ids")
            generated_ids = turn.get("generated_token_ids")
            observation = turn.get("observation")
            _require(isinstance(observation, Mapping), "M6 retention state lacks public observation")
            if not (
                isinstance(prompt_ids, list)
                and prompt_ids
                and isinstance(generated_ids, list)
                and generated_ids
                and all(isinstance(item, int) and item >= 0 for item in [*prompt_ids, *generated_ids])
            ):
                continue
            _require(
                len(prompt_ids) + len(generated_ids) <= int(contract["sft"]["maximum_sequence_tokens"]),
                "M6 retention state exceeds the SFT context",
            )
            page_type = str(observation.get("page_type") or "unknown")
            history = str(turn.get("rendered_prompt_sha256") or "")
            key = (task_id, page_type, history)
            if key in seen:
                continue
            seen.add(key)
            buckets[page_type].append(
                {
                    "schema_version": "m6_raw_retention_state_v1",
                    "task_id": task_id,
                    "trajectory_id": str(trajectory.get("trajectory_id", "")),
                    "turn_index": turn_index,
                    "page_type": page_type,
                    "source_success": trajectory.get("success") is True,
                    "source_recovery": source_recovery,
                    "supervised_label_present": False,
                    "prompt_token_ids": [int(item) for item in prompt_ids],
                    "raw_generated_token_ids": [int(item) for item in generated_ids],
                    "prompt_token_sha256": sha256_json(prompt_ids),
                    "raw_generated_token_sha256": sha256_json(generated_ids),
                    "public_state_sha256": sha256_json(dict(observation)),
                }
            )
    _require(bool(buckets), "M6 retention-state source is empty")
    selected: list[dict[str, Any]] = []
    bucket_names = sorted(buckets)
    offsets = {name: 0 for name in bucket_names}
    while len(selected) < maximum_states:
        made_progress = False
        for name in bucket_names:
            offset = offsets[name]
            if offset < len(buckets[name]):
                selected.append(buckets[name][offset])
                offsets[name] += 1
                made_progress = True
                if len(selected) == maximum_states:
                    break
        if not made_progress:
            break
    _require(selected, "M6 retention-state selection is empty")
    payload = {
        "schema_version": RETENTION_SCHEMA,
        "study_id": contract["study_id"],
        "development_only": True,
        "selection": "deterministic_page_type_round_robin_v1",
        "maximum_states": maximum_states,
        "state_count": len(selected),
        "success_state_count": sum(item["source_success"] for item in selected),
        "failure_state_count": sum(not item["source_success"] for item in selected),
        "recovery_state_count": sum(item["source_recovery"] for item in selected),
        "page_type_counts": dict(sorted(Counter(item["page_type"] for item in selected).items())),
        "supervised_label_count": 0,
        "states": selected,
    }
    payload["content_sha256"] = sha256_json(payload)
    return payload


def validate_retention_states(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == RETENTION_SCHEMA, "M6 retention schema drift")
    states = value.get("states")
    _require(isinstance(states, list) and states, "M6 retention states are missing")
    _require(value.get("state_count") == len(states), "M6 retention state count drift")
    _require(value.get("supervised_label_count") == 0, "M6 retention rows gained labels")
    for state in states:
        _require(state.get("supervised_label_present") is False, "M6 retention row gained a label")
        prompt_ids = state.get("prompt_token_ids")
        generated_ids = state.get("raw_generated_token_ids")
        _require(sha256_json(prompt_ids) == state.get("prompt_token_sha256"), "M6 retention prompt hash drift")
        _require(sha256_json(generated_ids) == state.get("raw_generated_token_sha256"), "M6 retention action hash drift")
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 retention self-hash drift")
    return value


def flatten_corpus_rows(corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return canonical action-turn JSONL rows with trajectory provenance."""

    expected = dict(corpus)
    observed = expected.pop("content_sha256", None)
    _require(corpus.get("schema_version") == CORPUS_SCHEMA, "M6 corpus schema drift")
    _require(observed == sha256_json(expected), "M6 corpus self-hash drift")
    rows: list[dict[str, Any]] = []
    for trajectory in corpus["trajectories"]:
        for turn in trajectory["turns"]:
            row = dict(turn)
            row.update(
                {
                    "task_id": trajectory["task_id"],
                    "trajectory_id": trajectory["trajectory_id"],
                    "trajectory_recovery": trajectory["recovery"],
                    "trajectory_command_sequence_sha256": trajectory["command_sequence_sha256"],
                }
            )
            rows.append(row)
    _require(rows, "M6 flattened corpus is empty")
    return rows


def audit_conditional_learnability(
    *,
    corpus: Mapping[str, Any],
    goal_by_task_id: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any] | None = None,
    mini: bool = True,
) -> dict[str, Any]:
    contract = dict(protocol or load_protocol()["payload"])
    validate_protocol(contract)
    corpus_contract = contract["corpus"]
    trajectories = corpus.get("trajectories")
    _require(corpus.get("schema_version") == CORPUS_SCHEMA, "M6 corpus schema drift")
    expected_corpus = dict(corpus)
    observed_corpus_hash = expected_corpus.pop("content_sha256", None)
    _require(observed_corpus_hash == sha256_json(expected_corpus), "M6 corpus self-hash drift")
    _require(isinstance(trajectories, list), "M6 corpus trajectories are missing")

    hidden_field_count = 0
    target_asin_search_label_count = 0
    zero_label_count = 0
    truncation_count = 0
    strict_success_count = 0
    replay_success_count = 0
    recovery_count = 0
    all_commands: list[str] = []
    query_lengths: list[int] = []
    query_tokens = 0
    public_query_tokens = 0
    unmatched_query_tokens: Counter[str] = Counter()
    sequences: list[str] = []
    task_counts: Counter[str] = Counter()
    duplicate_sequence_count = 0
    seen_sequences: set[tuple[str, str]] = set()
    completion_label_tokens = 0
    per_trajectory: list[dict[str, Any]] = []

    for trajectory_index, trajectory in enumerate(trajectories):
        _require(isinstance(trajectory, Mapping), f"M6 corpus trajectory {trajectory_index} is malformed")
        task_id = trajectory.get("task_id")
        _require(isinstance(task_id, str) and task_id in goal_by_task_id, "M6 corpus task is absent from goal audit map")
        goal = goal_by_task_id[task_id]
        instruction = str(goal.get("instruction") or "")
        target_asin = str(goal.get("asin") or "")
        _require(instruction and target_asin, "M6 goal audit metadata is incomplete")
        strict_success_count += int(trajectory.get("strict_success") is True)
        replay_success_count += int(trajectory.get("replay_success") is True)
        recovery_count += int(trajectory.get("recovery") is True)
        task_counts[task_id] += 1
        turns = trajectory.get("turns")
        _require(isinstance(turns, list) and turns, "M6 corpus trajectory has no turns")
        commands: list[str] = []
        trajectory_public_tokens = 0
        trajectory_query_tokens = 0
        trajectory_hidden = 0
        trajectory_target_leaks = 0
        for turn in turns:
            _require(isinstance(turn, Mapping), "M6 corpus turn is malformed")
            messages = turn.get("messages")
            _require(isinstance(messages, list) and messages, "M6 corpus messages are missing")
            _require(turn.get("prompt_sha256") == prompt.compute_message_hash(messages), "M6 corpus prompt hash drift")
            completion = turn.get("completion")
            if not isinstance(completion, str) or not completion.strip():
                zero_label_count += 1
                completion = ""
            truncation_count += int(turn.get("completion_truncated") is not False)
            policy_text = _policy_payload_text(turn)
            found_hidden = len(FORBIDDEN_POLICY_FIELD_RE.findall(policy_text))
            hidden_field_count += found_hidden
            trajectory_hidden += found_hidden
            command = _command(turn)
            parsed = parse_command_output(str(completion))
            _require(parsed.strict_json_success and parsed.schema_valid, "M6 corpus completion is not strict action JSON")
            _require(parsed.action is not None and parsed.action.command == command, "M6 corpus command/completion drift")
            commands.append(command)
            all_commands.append(command)
            source_tokens = turn.get("public_query_source_tokens")
            _require(
                isinstance(source_tokens, list)
                and source_tokens == sorted(set(source_tokens))
                and all(isinstance(token, str) and token for token in source_tokens),
                "M6 public query provenance tokens are malformed",
            )
            _require(
                turn.get("public_query_source_tokens_sha256") == sha256_json(source_tokens),
                "M6 public query provenance hash drift",
            )
            _require(
                source_tokens == _public_query_source_from_messages(messages),
                "M6 public query provenance does not match policy-visible text",
            )
            public_source = set(source_tokens)
            _require(set(_tokens(instruction)) <= public_source, "M6 query provenance omits instruction tokens")
            query = _search_query(command)
            if query is not None:
                query_lengths.append(len(query))
                if target_asin.casefold() in query.casefold():
                    target_asin_search_label_count += 1
                    trajectory_target_leaks += 1
                current_tokens = _tokens(query)
                query_tokens += len(current_tokens)
                trajectory_query_tokens += len(current_tokens)
                for token in current_tokens:
                    if token in public_source:
                        public_query_tokens += 1
                        trajectory_public_tokens += 1
                    else:
                        unmatched_query_tokens[token] += 1
            label_tokens = turn.get("completion_label_tokens")
            _require(isinstance(label_tokens, int) and not isinstance(label_tokens, bool) and label_tokens > 0, "M6 label token count drift")
            completion_label_tokens += label_tokens
        sequence_sha = sha256_json([command.casefold() for command in commands])
        sequence_key = (task_id, sequence_sha)
        duplicate_sequence_count += int(sequence_key in seen_sequences)
        seen_sequences.add(sequence_key)
        sequences.append(sequence_sha)
        per_trajectory.append(
            {
                "trajectory_id": str(trajectory.get("trajectory_id", "")),
                "task_id": task_id,
                "strict_success": trajectory.get("strict_success") is True,
                "replay_success": trajectory.get("replay_success") is True,
                "recovery": trajectory.get("recovery") is True,
                "query_token_count": trajectory_query_tokens,
                "public_query_token_count": trajectory_public_tokens,
                "hidden_policy_field_count": trajectory_hidden,
                "target_asin_search_label_count": trajectory_target_leaks,
                "command_sequence_sha256": sequence_sha,
            }
        )

    trajectory_count = len(trajectories)
    strict_fraction = strict_success_count / trajectory_count if trajectory_count else 0.0
    replay_fraction = replay_success_count / trajectory_count if trajectory_count else 0.0
    recovery_fraction = recovery_count / trajectory_count if trajectory_count else 0.0
    public_query_fraction = public_query_tokens / query_tokens if query_tokens else 0.0
    unique_sequence_fraction = len(set(sequences)) / trajectory_count if trajectory_count else 0.0
    families = Counter(action_family(command) for command in all_commands)
    maximum_family_fraction = max(families.values(), default=0) / len(all_commands) if all_commands else 0.0
    sorted_lengths = sorted(query_lengths)
    p95_index = max(0, math.ceil(0.95 * len(sorted_lengths)) - 1)
    mean_query_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0.0
    p95_query_length = float(sorted_lengths[p95_index]) if sorted_lengths else 0.0

    checks = {
        "nonempty": trajectory_count > 0,
        "strict_success_fraction": strict_fraction == float(corpus_contract["strict_success_fraction"]),
        "replay_success_fraction": replay_fraction == float(corpus_contract["replay_success_fraction"]),
        "zero_labels": zero_label_count == 0,
        "completion_truncations": truncation_count == 0,
        "hidden_policy_fields": hidden_field_count <= int(corpus_contract["maximum_hidden_field_count"]),
        "target_asin_search_labels": target_asin_search_label_count <= int(corpus_contract["maximum_target_asin_search_label_count"]),
        "public_query_token_fraction": public_query_fraction >= float(corpus_contract["minimum_public_query_token_fraction"]),
        "mean_search_query_characters": mean_query_length <= float(corpus_contract["maximum_mean_search_query_characters"]),
        "p95_search_query_characters": p95_query_length <= float(corpus_contract["maximum_p95_search_query_characters"]),
        "unique_command_sequence_fraction": unique_sequence_fraction >= float(corpus_contract["minimum_unique_command_sequence_fraction"]),
        "recovery_trajectory_fraction": recovery_fraction >= float(corpus_contract["minimum_recovery_trajectory_fraction"]),
        "action_family_fraction": maximum_family_fraction <= float(corpus_contract["maximum_action_family_fraction"]),
        "maximum_trajectories_per_task": max(task_counts.values(), default=0) <= int(corpus_contract["maximum_trajectories_per_task"]),
        "duplicate_task_sequence_count": duplicate_sequence_count == 0,
    }
    if mini:
        mini_contract = contract["mini"]
        checks.update(
            {
                "mini_success_task_count": len(task_counts) >= int(mini_contract["minimum_success_tasks"]),
                "mini_success_trajectory_count": trajectory_count >= int(mini_contract["minimum_success_trajectories"]),
                "mini_completion_label_token_minimum": completion_label_tokens >= int(mini_contract["minimum_completion_label_tokens"]),
                "mini_completion_label_token_maximum": completion_label_tokens <= int(mini_contract["maximum_completion_label_tokens"]),
            }
        )
    else:
        checks.update(
            {
                "formal_success_task_count": len(task_counts) >= int(corpus_contract["minimum_formal_success_tasks"]),
                "formal_trajectory_count_minimum": trajectory_count >= int(corpus_contract["minimum_formal_trajectories"]),
                "formal_trajectory_count_maximum": trajectory_count <= int(corpus_contract["maximum_formal_trajectories"]),
                "formal_completion_label_token_minimum": completion_label_tokens >= int(corpus_contract["minimum_formal_completion_label_tokens"]),
                "formal_completion_label_token_maximum": completion_label_tokens <= int(corpus_contract["maximum_formal_completion_label_tokens"]),
            }
        )

    report = {
        "schema_version": AUDIT_SCHEMA,
        "study_id": contract["study_id"],
        "scope": "mini" if mini else "formal",
        "development_only": bool(mini),
        "passed": all(checks.values()),
        "corpus_content_sha256": corpus["content_sha256"],
        "metrics": {
            "task_count": len(task_counts),
            "trajectory_count": trajectory_count,
            "turn_count": len(all_commands),
            "completion_label_tokens": completion_label_tokens,
            "strict_success_fraction": strict_fraction,
            "replay_success_fraction": replay_fraction,
            "zero_label_count": zero_label_count,
            "completion_truncation_count": truncation_count,
            "hidden_policy_field_count": hidden_field_count,
            "target_asin_search_label_count": target_asin_search_label_count,
            "query_token_count": query_tokens,
            "public_query_token_count": public_query_tokens,
            "public_query_token_fraction": public_query_fraction,
            "mean_search_query_characters": mean_query_length,
            "p95_search_query_characters": p95_query_length,
            "unique_command_sequence_fraction": unique_sequence_fraction,
            "recovery_trajectory_fraction": recovery_fraction,
            "maximum_action_family_fraction": maximum_family_fraction,
            "maximum_trajectories_per_task": max(task_counts.values(), default=0),
            "duplicate_task_sequence_count": duplicate_sequence_count,
        },
        "action_family_counts": dict(sorted(families.items())),
        "unmatched_query_token_counts": dict(sorted(unmatched_query_tokens.items())),
        "checks": checks,
        "trajectory_audits": per_trajectory,
    }
    report["content_sha256"] = sha256_json(report)
    return report


def validate_conditional_learnability_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == AUDIT_SCHEMA, "M6 corpus audit schema drift")
    checks = value.get("checks")
    _require(isinstance(checks, Mapping) and checks, "M6 corpus audit checks are missing")
    _require(value.get("passed") is all(bool(item) for item in checks.values()), "M6 corpus audit decision drift")
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 corpus audit self-hash drift")
    return value
