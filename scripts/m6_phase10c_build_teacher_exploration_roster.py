#!/usr/bin/env python3
"""Build the frozen 32-task Raw Qwen3.5-35B self-exploration roster."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m6_phase10c_data import validate_phase10c_split  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402

ROLE_COUNTS = {"teacher_nav_train": 11, "teacher_match_train": 11, "teacher_finish_train": 10}
USED_PREFIX_TASKS_PER_ROLE = 16
SELECTION_SEED = 20260862


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_roster(
    *,
    phase10c_split: Mapping[str, Any],
    base_split: Mapping[str, Any],
    protocol_sha256: str,
    producer_git_sha: str,
) -> dict[str, Any]:
    selected_by_role: dict[str, list[str]] = {}
    for role, count in ROLE_COUNTS.items():
        tasks = list(phase10c_split["roles"][role]["task_ids"])
        selected = tasks[USED_PREFIX_TASKS_PER_ROLE : USED_PREFIX_TASKS_PER_ROLE + count]
        _require(len(selected) == count, f"Phase10-C exploration role too small: {role}")
        selected_by_role[role] = selected
    task_ids: list[str] = []
    for index in range(max(ROLE_COUNTS.values())):
        for role in ROLE_COUNTS:
            if index < len(selected_by_role[role]):
                task_ids.append(selected_by_role[role][index])
    _require(len(task_ids) == len(set(task_ids)) == 32, "Phase10-C exploration task roster drift")
    base_train = set(base_split["roles"]["train"]["task_ids"])
    _require(set(task_ids) <= base_train, "Phase10-C exploration escaped base train role")
    value = {
        "schema_version": "m6_rl_curriculum_v1",
        "development_only": True,
        "formal_training": False,
        "purpose": "phase10c_raw_qwen35_full_environment_self_exploration",
        "selection": "phase10c_roles_after_failed_16_task_query_smokes_interleaved_v1",
        "selection_seed": SELECTION_SEED,
        "source_role": "train",
        "source_split_lock_content_sha256": base_split["content_sha256"],
        "phase10c_split_lock_content_sha256": phase10c_split["content_sha256"],
        "protocol_sha256": protocol_sha256,
        "git_sha": producer_git_sha,
        "phase10c_role": "teacher_self_exploration",
        "identity": "raw_qwen3.5_35b_a3b",
        "source_role_counts": dict(ROLE_COUNTS),
        "excluded_used_prefix_tasks_per_role": USED_PREFIX_TASKS_PER_ROLE,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "task_order_sha256": sha256_json(task_ids),
        "K": 4,
        "max_model_turns": 18,
        "max_environment_steps": 15,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    value["content_sha256"] = _self_hash(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase10c-split", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current_git = _git_sha()
    _require(args.producer_git_sha == current_git, "Phase10-C exploration roster Git drift")
    phase10c = validate_phase10c_split(json.loads(args.phase10c_split.read_text(encoding="utf-8")))
    base = validate_split_lock(json.loads(args.base_split.read_text(encoding="utf-8")))
    protocol = load_protocol()
    _require(base["protocol_sha256"] == protocol["sha256"], "Phase10-C exploration protocol drift")
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase10-C exploration roster already exists")
    roster = build_roster(
        phase10c_split=phase10c,
        base_split=base,
        protocol_sha256=protocol["sha256"],
        producer_git_sha=current_git,
    )
    atomic_write_json(output, roster)
    print(json.dumps({
        "task_count": roster["task_count"],
        "source_role_counts": roster["source_role_counts"],
        "task_order_sha256": roster["task_order_sha256"],
        "content_sha256": roster["content_sha256"],
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
