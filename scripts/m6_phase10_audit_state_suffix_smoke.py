#!/usr/bin/env python3
"""Fresh-replay and suffix-mask audit for the matched Phase10 K2 smoke."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl import prompt  # noqa: E402
from miniwebwork.webshop_rl.actions import WebShopCommand  # noqa: E402
from miniwebwork.webshop_rl.credit import public_state_anchor_signature  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
ASIN_RE = re.compile(r"^B[0-9A-Z]{9}$", re.IGNORECASE)
SEARCH_RE = re.compile(r"^search\[(.*)\]$", re.IGNORECASE | re.DOTALL)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Phase10 smoke artifact is not an object: {path}")
    _require(value.get("content_sha256") == _self_hash(value), f"Phase10 smoke self-hash drift: {path}")
    return value


def _tokens(value: Any) -> set[str]:
    return {match.group(0).casefold() for match in TOKEN_RE.finditer(str(value or ""))}


def _failure_class(score: float, commands: Sequence[str]) -> str:
    if score >= 0.999:
        return "strict_success"
    if any(command.casefold() == "click[buy now]" for command in commands):
        return "partial_purchase" if score > 0.0 else "zero_score_purchase"
    return "no_purchase"


def query_provenance(trajectory: Mapping[str, Any], *, prefix_turn_count: int) -> dict[str, Any]:
    """Require suffix search terms to have appeared in public prompt material."""

    turns = trajectory["turns"]
    allowed_tokens: set[str] = set()
    seen_public_asins: set[str] = set()
    violations: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        observation = turn["observation"]
        allowed_tokens.update(_tokens(observation.get("instruction")))
        allowed_tokens.update(_tokens(observation.get("visible_text")))
        for action in observation.get("available_actions", []):
            argument = str(action).removeprefix("click[").removesuffix("]")
            if ASIN_RE.fullmatch(argument):
                seen_public_asins.add(argument.casefold())
        action = turn.get("action")
        command = str(action.get("command", "")) if isinstance(action, Mapping) else ""
        if index < prefix_turn_count:
            continue
        match = SEARCH_RE.fullmatch(command)
        if match is None:
            continue
        query = match.group(1).strip()
        query_tokens = _tokens(query)
        unknown = sorted(query_tokens - allowed_tokens)
        hidden_asins = sorted(
            token.casefold()
            for token in re.findall(r"B[0-9A-Z]{9}", query, flags=re.IGNORECASE)
            if token.casefold() not in seen_public_asins
        )
        if unknown or hidden_asins or len(query) > 200:
            violations.append({
                "turn_index": index + 1,
                "unknown_query_tokens": unknown,
                "unseen_asins": hidden_asins,
                "query_length": len(query),
            })
    return {
        "passed": not violations,
        "violation_count": len(violations),
        "violations": violations,
    }


def fresh_replay(
    *,
    task_id: str,
    trajectory: Mapping[str, Any],
    base_url: str,
) -> dict[str, Any]:
    """Replay one complete command sequence in a fresh environment session."""

    environment = WebShopHTTPEnvironment(base_url=base_url, split="train", timeout_seconds=120)
    commands: list[str] = []
    score = 0.0
    terminated = False
    try:
        observation = environment.reset(task_id)
        for turn in trajectory["turns"]:
            _require(
                public_state_anchor_signature(observation.to_dict())
                == turn["pre_action_public_state_sha256"],
                "Phase10 fresh replay pre-action public state drift",
            )
            action = turn.get("action")
            if not isinstance(action, Mapping):
                _require(
                    turn["pre_action_public_state_sha256"] == turn["post_action_public_state_sha256"],
                    "Phase10 non-action turn changed public state",
                )
                continue
            command = str(action.get("command", ""))
            commands.append(command)
            result = environment.step(WebShopCommand(command))
            action_result = result.info.get("action_result", {})
            _require(action_result.get("success") is True, "Phase10 fresh replay action failed")
            observation = result.observation
            _require(
                public_state_anchor_signature(observation.to_dict())
                == turn["post_action_public_state_sha256"],
                "Phase10 fresh replay post-action public state drift",
            )
            score = float(result.info.get("task_score", score))
            terminated = bool(result.terminated)
            if terminated:
                break
    finally:
        environment.close()
    replay_class = _failure_class(score, commands)
    return {
        "fresh_session": True,
        "public_state_exact": True,
        "executed_command_count": len(commands),
        "terminated": terminated,
        "task_score": score,
        "strict_success": score >= 0.999,
        "failure_class": replay_class,
        "command_sequence_sha256": sha256_json(commands),
    }


def suffix_mask_rows(
    trajectory: Mapping[str, Any],
    *,
    prefix_turn_count: int,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Retokenize exact policy prompts and mask every non-assistant-action token."""

    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for index, turn in enumerate(trajectory["turns"]):
        observation = SimpleNamespace(**dict(turn["observation"]))
        messages = prompt.build_messages(observation, history)
        _require(
            prompt.compute_message_hash(messages) == turn["rendered_prompt_sha256"],
            "Phase10 mask prompt hash drift",
        )
        if index >= prefix_turn_count and isinstance(turn.get("action"), Mapping):
            completion = json.dumps(
                {"command": str(turn["action"]["command"])},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            prompt_ids = list(tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
            action_ids = list(tokenizer.encode(completion, add_special_tokens=False))
            _require(prompt_ids and action_ids, "Phase10 mask produced an empty token region")
            labels = [-100] * len(prompt_ids) + action_ids
            _require(all(label == -100 for label in labels[: len(prompt_ids)]), "Phase10 prompt label leak")
            _require(all(label >= 0 for label in labels[len(prompt_ids) :]), "Phase10 action label drift")
            rows.append({
                "turn_index": index + 1,
                "prompt_sha256": turn["rendered_prompt_sha256"],
                "prompt_token_count": len(prompt_ids),
                "masked_token_count": len(prompt_ids),
                "action_label_token_count": len(action_ids),
                "labels_sha256": sha256_json(labels),
            })
        action = turn.get("action")
        action_result = turn.get("action_result")
        if isinstance(action, Mapping) and isinstance(action_result, Mapping):
            history.append({
                "turn": index + 1,
                "command": str(action.get("command", "")),
                "success": bool(action_result.get("success", False)),
                "error_code": str(action_result.get("error_code", ""))[:100],
                "page_type": str(turn["post_action_observation"].get("page_type", "unknown")),
            })
    return rows


def _load_collection(root: Path, *, identity: str, manifest_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = _load_hashed(root / "collection_report.json")
    _require(
        report.get("complete") is True
        and report.get("mode") == "phase10_state_suffix_smoke"
        and report.get("phase10_smoke_identity") == identity
        and report.get("K") == 2
        and report.get("task_count") == 8
        and report.get("trajectory_count") == 16
        and report.get("training_updates_allowed") is False
        and report.get("exact_shared_prefix_replay_task_count") == 8
        and report.get("future_policy_loss_scope") == "suffix_tokens_only"
        and report["shared_prefix_source"]["manifest_content_sha256"] == manifest_sha,
        f"Phase10 {identity} collection contract drift",
    )
    groups = [
        validate_committed_group(_load_hashed(root / "groups" / f"g{index:04d}.json"), require_k=2)
        for index in range(8)
    ]
    _require(report["group_content_sha256"] == [group["content_sha256"] for group in groups], "Phase10 group/report drift")
    return report, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--student-tokenizer", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase10 smoke audit output already exists")
    manifest = _load_hashed(args.manifest)
    _require(manifest.get("schema_version") == "m6_phase10_state_suffix_smoke_manifest_v1", "Phase10 manifest schema drift")
    task_specs = {row["task_id"]: row for row in manifest["tasks"]}
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.student_tokenizer.expanduser().resolve()),
        trust_remote_code=True,
    )
    student_report, student_groups = _load_collection(
        args.student_root.expanduser().resolve(), identity="student", manifest_sha=manifest["content_sha256"]
    )
    teacher_report, teacher_groups = _load_collection(
        args.teacher_root.expanduser().resolve(), identity="teacher", manifest_sha=manifest["content_sha256"]
    )
    _require(
        [group["task_id"] for group in student_groups]
        == [group["task_id"] for group in teacher_groups]
        == list(task_specs),
        "Phase10 matched smoke task order drift",
    )

    trajectories: list[dict[str, Any]] = []
    for identity, groups in (("student", student_groups), ("teacher", teacher_groups)):
        for group in groups:
            spec = task_specs[group["task_id"]]
            prefix_count = int(spec["prefix_turn_count"])
            correction_hashes = {
                trajectory["turns"][prefix_count]["rendered_prompt_sha256"]
                for trajectory in group["trajectories"]
            }
            _require(len(correction_hashes) == 1, "Phase10 K2 correction prompt mismatch")
            for trajectory in group["trajectories"]:
                commands = [
                    str(turn["action"]["command"])
                    for turn in trajectory["turns"]
                    if isinstance(turn.get("action"), Mapping)
                ]
                replay = fresh_replay(task_id=group["task_id"], trajectory=trajectory, base_url=args.base_url)
                original_class = _failure_class(float(trajectory["task_score"]), commands)
                _require(replay["failure_class"] == original_class, "Phase10 fresh replay outcome class drift")
                provenance = query_provenance(trajectory, prefix_turn_count=prefix_count)
                masks = suffix_mask_rows(trajectory, prefix_turn_count=prefix_count, tokenizer=tokenizer)
                trajectories.append({
                    "identity": identity,
                    "task_id": group["task_id"],
                    "rollout_index": trajectory["rollout_index"],
                    "trajectory_id": trajectory["trajectory_id"],
                    "prefix_turn_count": prefix_count,
                    "correction_prompt_sha256": next(iter(correction_hashes)),
                    "first_execution_failure_class": original_class,
                    "first_execution_strict": trajectory["success"] is True,
                    "fresh_replay": replay,
                    "query_provenance": provenance,
                    "suffix_mask_rows": masks,
                    "suffix_action_row_count": len(masks),
                    "suffix_action_label_tokens": sum(row["action_label_token_count"] for row in masks),
                })

    student_rows = [row for row in trajectories if row["identity"] == "student"]
    teacher_rows = [row for row in trajectories if row["identity"] == "teacher"]
    teacher_first_strict = [row for row in teacher_rows if row["first_execution_strict"]]
    matched_prompt_tasks = {
        task_id
        for task_id in task_specs
        if {row["correction_prompt_sha256"] for row in trajectories if row["task_id"] == task_id}
        and len({row["correction_prompt_sha256"] for row in trajectories if row["task_id"] == task_id}) == 1
    }
    gates = {
        "exact_prefix_replay_8_of_8": (
            student_report["exact_shared_prefix_replay_task_count"] == 8
            and teacher_report["exact_shared_prefix_replay_task_count"] == 8
        ),
        "matched_correction_prompt_8_of_8": len(matched_prompt_tasks) == 8,
        "teacher_first_strict_at_least_one": len(teacher_first_strict) >= 1,
        "teacher_first_strict_fresh_replay_rate_1": (
            bool(teacher_first_strict)
            and all(row["fresh_replay"]["strict_success"] for row in teacher_first_strict)
        ),
        "all_query_provenance_pass": all(row["query_provenance"]["passed"] for row in trajectories),
        "all_suffix_masks_nonempty_and_prompt_masked": all(
            row["suffix_action_row_count"] > 0 and row["suffix_action_label_tokens"] > 0
            for row in trajectories
        ),
        "student_nonstrict_replay_class_stable": all(
            row["first_execution_failure_class"] == row["fresh_replay"]["failure_class"]
            for row in student_rows
            if not row["first_execution_strict"]
        ),
    }
    report = {
        "schema_version": "m6_phase10_state_suffix_smoke_audit_v1",
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "public_fields_only": True,
        "target_asin_or_hidden_answer_used": False,
        "fresh_session_full_replay": True,
        "student_tokenizer": str(args.student_tokenizer.expanduser().resolve()),
        "manifest_content_sha256": manifest["content_sha256"],
        "student_collection_report_content_sha256": student_report["content_sha256"],
        "teacher_collection_report_content_sha256": teacher_report["content_sha256"],
        "task_count": 8,
        "trajectory_count": len(trajectories),
        "teacher_first_strict_trajectory_count": len(teacher_first_strict),
        "teacher_replay_strict_trajectory_count": sum(
            row["fresh_replay"]["strict_success"] for row in teacher_first_strict
        ),
        "student_first_strict_trajectory_count": sum(row["first_execution_strict"] for row in student_rows),
        "gates": gates,
        "passed": all(gates.values()),
        "trajectories": trajectories,
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(output, report)
    print(json.dumps({
        "output": str(output),
        "content_sha256": report["content_sha256"],
        "passed": report["passed"],
        "gates": gates,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
