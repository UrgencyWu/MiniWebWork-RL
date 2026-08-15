#!/usr/bin/env python3
"""Freeze eight historical student-failure states for the Phase10 engineering smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m6_phase4_build_data_rosters import _bucket  # noqa: E402
from m6_phase7_terminal_contrast_diagnostic import FORBIDDEN_PATH_PARTS, _load_groups  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402

TASK_COUNT = 8
K = 2
SELECTION_SEED = 20260841
BUY_COMMAND = "click[buy now]"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _git_sha() -> str:
    value = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase10 smoke repository Git SHA drift")
    return value


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _failure_class(trajectory: Mapping[str, Any]) -> str | None:
    if trajectory.get("success") is True:
        return None
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        return None
    purchased = any(
        _normalized((turn.get("action") or {}).get("command")) == BUY_COMMAND
        for turn in turns
        if isinstance(turn, Mapping)
    )
    if not purchased:
        return None
    score = float(trajectory.get("task_score", 0.0))
    return "partial_purchase" if score > 0.0 else "zero_score_purchase"


def public_purchase_correction_prefix(trajectory: Mapping[str, Any]) -> dict[str, Any] | None:
    """Select the pre-Buy state without consulting terminal score or hidden goal fields."""

    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        return None
    buy_index = next(
        (
            index
            for index in range(len(turns) - 1, -1, -1)
            if _normalized((turns[index].get("action") or {}).get("command")) == BUY_COMMAND
        ),
        None,
    )
    if buy_index is None or buy_index <= 0:
        return None
    prefix_turns = turns[:buy_index]
    if not all((turn.get("action_result") or {}).get("success") is True for turn in prefix_turns):
        return None
    correction_observation = turns[buy_index].get("observation")
    if not isinstance(correction_observation, Mapping):
        return None
    available_actions = correction_observation.get("available_actions")
    if not isinstance(available_actions, list) or not any(
        _normalized(command) == BUY_COMMAND for command in available_actions
    ):
        return None
    commands = [str((turn.get("action") or {}).get("command", "")).strip() for turn in prefix_turns]
    if not all(commands):
        return None
    token_evidence = [
        {
            "prompt_token_ids": list(turn.get("prompt_token_ids", [])),
            "generated_token_ids": list(turn.get("generated_token_ids", [])),
            "behavior_logprobs": list(turn.get("behavior_logprobs", [])),
            "sampling_logprobs": list(turn.get("sampling_logprobs", [])),
        }
        for turn in prefix_turns
    ]
    _require(
        all(
            row["prompt_token_ids"]
            and row["generated_token_ids"]
            and len(row["generated_token_ids"])
            == len(row["behavior_logprobs"])
            == len(row["sampling_logprobs"])
            for row in token_evidence
        ),
        "Phase10 smoke prefix token evidence drift",
    )
    return {
        "prefix_turn_count": buy_index,
        "prefix_environment_steps": sum(
            (turn.get("action_result") or {}).get("success") is True for turn in prefix_turns
        ),
        "remaining_model_turns": 18 - buy_index,
        "remaining_environment_steps": 15 - len(prefix_turns),
        "correction_public_state_sha256": str(turns[buy_index].get("pre_action_public_state_sha256", "")),
        "prefix_action_sequence_sha256": sha256_json(commands),
        "prefix_token_evidence_sha256": sha256_json(token_evidence),
    }


def _rank(namespace: str, task_id: str) -> bytes:
    return hashlib.sha256(f"{namespace}|{SELECTION_SEED}|{task_id}".encode()).digest()


def _select_stratified(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    goal_map: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    _require(len(candidates) >= TASK_COUNT, "Phase10 smoke has fewer than eight public purchase-failure states")
    by_bucket: dict[str, list[str]] = {}
    for task_id in candidates:
        by_bucket.setdefault(_bucket(goal_map[task_id]), []).append(task_id)
    for task_ids in by_bucket.values():
        task_ids.sort(key=lambda task_id: (_rank("m6-phase10-smoke-bucket", task_id), task_id))
    selected: list[str] = []
    depth = 0
    while len(selected) < TASK_COUNT:
        wave = [task_ids[depth] for _, task_ids in sorted(by_bucket.items()) if depth < len(task_ids)]
        wave.sort(key=lambda task_id: (_rank("m6-phase10-smoke-wave", task_id), task_id))
        _require(wave, "Phase10 smoke stratified selection exhausted unexpectedly")
        selected.extend(wave[: TASK_COUNT - len(selected)])
        depth += 1
    return selected


def _best_candidates(groups: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for group in groups:
        task_id = str(group["task_id"])
        for rollout_index, trajectory in enumerate(group["trajectories"]):
            failure_class = _failure_class(trajectory)
            prefix = public_purchase_correction_prefix(trajectory)
            if failure_class is None or prefix is None:
                continue
            if prefix["remaining_model_turns"] <= 0 or prefix["remaining_environment_steps"] <= 0:
                continue
            candidate = {
                "task_id": task_id,
                "source_group_id": group["group_id"],
                "source_group_content_sha256": group["content_sha256"],
                "source_rollout_index": rollout_index,
                "source_trajectory_id": trajectory["trajectory_id"],
                "source_failure_class": failure_class,
                **prefix,
            }
            key = (
                prefix["prefix_turn_count"],
                prefix["correction_public_state_sha256"],
                str(trajectory["trajectory_id"]),
            )
            previous = output.get(task_id)
            if previous is None or key < previous["_selection_key"]:
                candidate["_selection_key"] = key
                output[task_id] = candidate
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--roster-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    current_git = _git_sha()
    _require(args.producer_git_sha == current_git, "Phase10 smoke producer Git drift")
    protocol = load_protocol()
    split = validate_split_lock(json.loads(args.split_lock.expanduser().resolve().read_text(encoding="utf-8")))
    _require(split["protocol_sha256"] == protocol["sha256"], "Phase10 smoke split/protocol drift")
    goals = json.loads(args.goals.expanduser().resolve().read_text(encoding="utf-8"))
    _require(sha256_json(goals) == split["goals_canonical_sha256"], "Phase10 smoke goals/split drift")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    train_ids = set(split["roles"]["train"]["task_ids"])
    roots = [path.expanduser().resolve() for path in args.source_root]
    groups, bindings = _load_groups(roots, expected_mode="phase4_data_synthesis")
    lineages = {sha256_json(binding["policy_lineage"]): binding["policy_lineage"] for binding in bindings}
    _require(len(lineages) == 1, "Phase10 smoke source policy lineage is not unique")
    source_lineage = next(iter(lineages.values()))
    candidates = _best_candidates(groups)
    _require(set(candidates) <= train_ids, "Phase10 smoke candidate escaped train role")
    selected = _select_stratified(candidates, goal_map=goal_map)

    # _load_groups flattens groups, so bind each selected group by unique content hash.
    bindings_by_group_sha: dict[str, dict[str, Any]] = {}
    for root, binding in zip(roots, bindings):
        report_hashes = set(json.loads((root / "collection_report.json").read_text(encoding="utf-8"))["group_content_sha256"])
        for group_sha in report_hashes:
            bindings_by_group_sha[str(group_sha)] = binding

    task_rows = []
    for task_id in selected:
        row = dict(candidates[task_id])
        row.pop("_selection_key")
        binding = bindings_by_group_sha[row["source_group_content_sha256"]]
        row.update(
            source_collection_root=binding["collection_root"],
            source_collection_report_content_sha256=binding["collection_report_content_sha256"],
            source_policy_lineage=dict(binding["policy_lineage"]),
        )
        task_rows.append(row)

    roster = {
        "schema_version": "m6_rl_curriculum_v1",
        "development_only": True,
        "formal_training": False,
        "purpose": "phase10_historical_student_failure_engineering_smoke",
        "selection": "public_prebuy_purchase_failure_stratified_hash_v1",
        "selection_seed": SELECTION_SEED,
        "source_role": "train",
        "source_split_lock_content_sha256": split["content_sha256"],
        "protocol_sha256": protocol["sha256"],
        "git_sha": current_git,
        "task_count": TASK_COUNT,
        "task_ids": selected,
        "task_order_sha256": sha256_json(selected),
        "K": K,
        "max_model_turns": 18,
        "max_environment_steps": 15,
        "training_performed": False,
        "optimizer_steps": 0,
        "selected_bucket_counts": dict(sorted(Counter(_bucket(goal_map[task_id]) for task_id in selected).items())),
    }
    roster["content_sha256"] = _self_hash(roster)
    manifest = {
        "schema_version": "m6_phase10_state_suffix_smoke_manifest_v1",
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "public_fields_only": True,
        "task_score_used_for_eligibility_and_offline_stratification": True,
        "task_score_used_for_selector": False,
        "target_asin_or_hidden_answer_used": False,
        "post_action_internal_state_used": False,
        "producer_git_sha": current_git,
        "protocol_sha256": protocol["sha256"],
        "split_lock_content_sha256": split["content_sha256"],
        "task_roster_content_sha256": roster["content_sha256"],
        "task_order_sha256": roster["task_order_sha256"],
        "task_count": TASK_COUNT,
        "K": K,
        "source_policy_lineage": dict(source_lineage),
        "source_bindings": bindings,
        "tasks": task_rows,
        "correction_selector": "last_public_prebuy_state_for_all_nonstrict_purchase_v1",
        "future_policy_loss_contract": {
            "scope": "suffix_assistant_action_tokens_only",
            "prefix_policy_loss_eligible": False,
            "system_instruction_observation_labels": -100,
        },
    }
    manifest["content_sha256"] = _self_hash(manifest)
    for destination in (args.roster_output, args.manifest_output):
        _require(not destination.expanduser().resolve().exists(), f"Phase10 smoke output exists: {destination}")
    atomic_write_json(args.roster_output, roster)
    atomic_write_json(args.manifest_output, manifest)
    print(json.dumps({
        "roster": str(args.roster_output.expanduser().resolve()),
        "roster_content_sha256": roster["content_sha256"],
        "manifest": str(args.manifest_output.expanduser().resolve()),
        "manifest_content_sha256": manifest["content_sha256"],
        "candidate_task_count": len(candidates),
        "selected_task_count": len(selected),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
