#!/usr/bin/env python3
"""Build paired pi0/Specialist qualification rosters from the frozen Phase10-B split."""

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
ROLE_BY_IDENTITY = {
    "S_nav": "specialist_nav_qualification",
    "S_match": "specialist_match_qualification",
    "S_finish": "specialist_finish_qualification",
}


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
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase10-B roster Git SHA drift")
    return value


def _roster(
    *,
    identity: str,
    comparison_specialist: str,
    task_ids: list[str],
    phase10b_split: Mapping[str, Any],
    base_split: Mapping[str, Any],
    protocol_sha256: str,
    git_sha: str,
) -> dict[str, Any]:
    _require(comparison_specialist in ROLE_BY_IDENTITY, "Phase10-B comparison Specialist drift")
    _require(identity in {"student", comparison_specialist}, "Phase10-B paired roster identity drift")
    role = ROLE_BY_IDENTITY[comparison_specialist]
    value = {
        "schema_version": "m6_rl_curriculum_v1",
        "development_only": True,
        "formal_training": False,
        "purpose": f"phase10b_student_vs_{comparison_specialist}_qualification",
        "selection": "phase10b_frozen_specialist_proxy_roles_v1",
        "selection_seed": phase10b_split["selection_seed"],
        "source_role": "train",
        "source_split_lock_content_sha256": base_split["content_sha256"],
        "phase10b_split_lock_content_sha256": phase10b_split["content_sha256"],
        "phase10b_role": role,
        "protocol_sha256": protocol_sha256,
        "git_sha": git_sha,
        "identity": identity,
        "comparison_specialist": comparison_specialist,
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
    parser.add_argument("--phase10b-split", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    current_git = _git_sha()
    _require(args.producer_git_sha == current_git, "Phase10-B roster producer Git drift")
    protocol = load_protocol()
    phase10b = validate_phase10b_split(
        json.loads(args.phase10b_split.expanduser().resolve().read_text(encoding="utf-8"))
    )
    base = validate_split_lock(json.loads(args.base_split.expanduser().resolve().read_text(encoding="utf-8")))
    _require(base["protocol_sha256"] == protocol["sha256"], "Phase10-B roster protocol drift")
    train_ids = set(base["roles"]["train"]["task_ids"])
    specialist_tasks = {
        identity: list(phase10b["roles"][role]["task_ids"])
        for identity, role in ROLE_BY_IDENTITY.items()
    }
    all_tasks = [task_id for identity in ROLE_BY_IDENTITY for task_id in specialist_tasks[identity]]
    _require(len(all_tasks) == len(set(all_tasks)) == 72, "Phase10-B qualification role overlap")
    _require(set(all_tasks) <= train_ids, "Phase10-B qualification roster escaped base train role")
    rosters: dict[str, dict[str, Any]] = {}
    for specialist, task_ids in specialist_tasks.items():
        rosters[f"student_{specialist}"] = _roster(
            identity="student",
            comparison_specialist=specialist,
            task_ids=task_ids,
            phase10b_split=phase10b,
            base_split=base,
            protocol_sha256=protocol["sha256"],
            git_sha=current_git,
        )
        rosters[specialist] = _roster(
            identity=specialist,
            comparison_specialist=specialist,
            task_ids=task_ids,
            phase10b_split=phase10b,
            base_split=base,
            protocol_sha256=protocol["sha256"],
            git_sha=current_git,
        )
    output_root = args.output_root.expanduser().resolve()
    for identity, roster in rosters.items():
        destination = output_root / f"{identity}.json"
        _require(not destination.exists(), f"Phase10-B roster already exists: {identity}")
        atomic_write_json(destination, roster)
    print(json.dumps({
        "output_root": str(output_root),
        "rosters": {
            identity: {
                "task_count": roster["task_count"],
                "task_order_sha256": roster["task_order_sha256"],
                "content_sha256": roster["content_sha256"],
            }
            for identity, roster in rosters.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
