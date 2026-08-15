#!/usr/bin/env python3
"""Build paired 96-task Raw35/SFT35 and SFT35/SFT4 evaluation rosters."""

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
from miniwebwork.m6_phase10c_data import validate_phase10c_split  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CAPABILITIES = ("nav", "match", "finish")
STAGES = ("dev", "qualification")


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
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase10-C evaluation roster Git drift")
    return value


def build_roster(
    *,
    stage: str,
    phase10c_split: Mapping[str, Any],
    base_split: Mapping[str, Any],
    protocol_sha256: str,
    git_sha: str,
) -> dict[str, Any]:
    _require(stage in STAGES, "Phase10-C evaluation stage drift")
    by_capability = {
        capability: list(phase10c_split["roles"][f"teacher_{capability}_{stage}"]["task_ids"])
        for capability in CAPABILITIES
    }
    _require(all(len(rows) == 32 for rows in by_capability.values()),
             "Phase10-C evaluation role count drift")
    task_ids = [
        by_capability[capability][index]
        for index in range(32)
        for capability in CAPABILITIES
    ]
    _require(len(task_ids) == len(set(task_ids)) == 96,
             "Phase10-C evaluation role overlap")
    _require(set(task_ids) <= set(base_split["roles"]["train"]["task_ids"]),
             "Phase10-C evaluation roster escaped train role")
    value = {
        "schema_version": "m6_rl_curriculum_v1",
        "development_only": True,
        "formal_training": False,
        "purpose": "phase10c_teacher_stage_paired_evaluation",
        "selection": "phase10c_three_capability_round_robin_v1",
        "selection_seed": phase10c_split["selection_seed"],
        "source_role": "train",
        "source_split_lock_content_sha256": base_split["content_sha256"],
        "phase10c_split_lock_content_sha256": phase10c_split["content_sha256"],
        "phase10c_evaluation_stage": stage,
        "phase10c_roles": [f"teacher_{capability}_{stage}" for capability in CAPABILITIES],
        "capability_task_counts": {capability: 32 for capability in CAPABILITIES},
        "protocol_sha256": protocol_sha256,
        "git_sha": git_sha,
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
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    current_git = _git_sha()
    _require(args.producer_git_sha == current_git, "Phase10-C evaluation roster producer Git drift")
    protocol = load_protocol()
    phase10c = validate_phase10c_split(
        json.loads(args.phase10c_split.expanduser().resolve().read_text(encoding="utf-8"))
    )
    base = validate_split_lock(
        json.loads(args.base_split.expanduser().resolve().read_text(encoding="utf-8"))
    )
    _require(base["protocol_sha256"] == protocol["sha256"],
             "Phase10-C evaluation roster protocol drift")
    root = args.output_root.expanduser().resolve()
    outputs = {}
    for stage in STAGES:
        roster = build_roster(
            stage=stage,
            phase10c_split=phase10c,
            base_split=base,
            protocol_sha256=protocol["sha256"],
            git_sha=current_git,
        )
        destination = root / f"{stage}.json"
        _require(not destination.exists(), f"Phase10-C evaluation roster exists: {stage}")
        atomic_write_json(destination, roster)
        outputs[stage] = {
            "path": str(destination),
            "task_count": roster["task_count"],
            "task_order_sha256": roster["task_order_sha256"],
            "content_sha256": roster["content_sha256"],
        }
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
