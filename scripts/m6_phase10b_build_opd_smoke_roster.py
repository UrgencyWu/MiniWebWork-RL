#!/usr/bin/env python3
"""Build the frozen eight-task Student behavior roster for Phase10-B OPD smoke."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m6_phase10b_opd import validate_phase10b_split  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402

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
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase10-B OPD smoke roster Git SHA drift")
    return value


def build_roster(
    *,
    phase10b_split: Mapping[str, Any],
    base_split: Mapping[str, Any],
    protocol_sha256: str,
    git_sha: str,
) -> dict[str, Any]:
    phase10b = validate_phase10b_split(phase10b_split)
    base = validate_split_lock(base_split)
    task_ids = list(phase10b["roles"]["opd_smoke"]["task_ids"])
    _require(len(task_ids) == len(set(task_ids)) == 8, "Phase10-B OPD smoke role drift")
    _require(set(task_ids) <= set(base["roles"]["train"]["task_ids"]),
             "Phase10-B OPD smoke escaped base train role")
    value = {
        "schema_version": "m6_phase10b_opd_smoke_roster_v1",
        "development_only": True,
        "formal_training": False,
        "purpose": "phase10b_student_behavior_opd_zero_update_smoke",
        "selection": "phase10b_frozen_fresh_role_v1",
        "selection_seed": phase10b["selection_seed"],
        "source_role": "train",
        "source_split_lock_content_sha256": base["content_sha256"],
        "phase10b_split_lock_content_sha256": phase10b["content_sha256"],
        "phase10b_role": "opd_smoke",
        "protocol_sha256": protocol_sha256,
        "git_sha": git_sha,
        "identity": "student",
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "task_order_sha256": sha256_json(task_ids),
        "K": 4,
        "max_model_turns": 18,
        "max_environment_steps": 15,
        "behavior_policy": "pi_0",
        "specialist_actions_executed": False,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    value["content_sha256"] = _self_hash(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase10b-split", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current_git = _git_sha()
    _require(args.producer_git_sha == current_git, "Phase10-B OPD smoke roster producer Git drift")
    protocol = load_protocol()
    base = json.loads(args.base_split.expanduser().resolve().read_text(encoding="utf-8"))
    _require(validate_split_lock(base)["protocol_sha256"] == protocol["sha256"],
             "Phase10-B OPD smoke protocol drift")
    roster = build_roster(
        phase10b_split=json.loads(args.phase10b_split.expanduser().resolve().read_text(encoding="utf-8")),
        base_split=base,
        protocol_sha256=protocol["sha256"],
        git_sha=current_git,
    )
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase10-B OPD smoke roster already exists")
    atomic_write_json(output, roster)
    print(json.dumps({
        "output": str(output),
        "task_count": roster["task_count"],
        "task_order_sha256": roster["task_order_sha256"],
        "content_sha256": roster["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
