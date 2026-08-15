#!/usr/bin/env python3
"""Build the minimal frozen Phase10-C Specialist-SFT and OPD data split."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json, sha256_json  # noqa: E402
from miniwebwork.m6_phase10c_data import (  # noqa: E402
    build_phase10c_exposure_union,
    build_phase10c_split,
)
from miniwebwork.m6_posttraining_protocol import validate_split_lock  # noqa: E402

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    value = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase10-C repository Git SHA drift")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--base-split-lock", type=Path, required=True)
    parser.add_argument("--phase10b-exposure", type=Path, required=True)
    parser.add_argument("--phase10b-split", type=Path, required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--exposure-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    args = parser.parse_args()

    current_git = _git_sha()
    _require(args.producer_git_sha == current_git, "Phase10-C producer Git drift")
    outputs = [args.exposure_output.expanduser().resolve(), args.split_output.expanduser().resolve()]
    _require(all(not path.exists() for path in outputs), "Phase10-C output already exists")

    goals = json.loads(args.goals.expanduser().resolve().read_text(encoding="utf-8"))
    _require(isinstance(goals, list), "Phase10-C goals payload must be a list")
    base_split = validate_split_lock(
        json.loads(args.base_split_lock.expanduser().resolve().read_text(encoding="utf-8"))
    )
    _require(sha256_json(goals) == base_split["goals_canonical_sha256"], "Phase10-C goals/base split drift")
    phase10b_exposure = json.loads(args.phase10b_exposure.expanduser().resolve().read_text(encoding="utf-8"))
    phase10b_split = json.loads(args.phase10b_split.expanduser().resolve().read_text(encoding="utf-8"))

    exposure = build_phase10c_exposure_union(
        goals=goals,
        phase10b_exposure=phase10b_exposure,
        phase10b_split=phase10b_split,
        producer_git_sha=current_git,
    )
    split = build_phase10c_split(
        goals=goals,
        train_task_ids=base_split["roles"]["train"]["task_ids"],
        exposure_union=exposure,
        base_split_content_sha256=base_split["content_sha256"],
        producer_git_sha=current_git,
    )
    publish_immutable_json(outputs[0], exposure)
    publish_immutable_json(outputs[1], split)
    print(json.dumps({
        "exposure_output": str(outputs[0]),
        "exposed_task_count": exposure["exposed_task_count"],
        "split_output": str(outputs[1]),
        "eligible_after_exposure_count": split["eligible_after_exposure_count"],
        "role_counts": {role: item["count"] for role, item in split["roles"].items()},
        "quick_start": split["teacher_sft_quick_start"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
