#!/usr/bin/env python3
"""Build Phase10's complete exposure union and six fresh task roles."""

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
from miniwebwork.m6_phase10_readiness import build_exposure_union, build_phase10_split  # noqa: E402
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
    _require(GIT_SHA_RE.fullmatch(value) is not None, "Phase10 repository Git SHA drift")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--base-split-lock", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--exposure-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    args = parser.parse_args()

    current_git = _git_sha()
    _require(args.producer_git_sha == current_git, "Phase10 producer Git drift")
    exposure_output = args.exposure_output.expanduser().resolve()
    split_output = args.split_output.expanduser().resolve()
    _require(not exposure_output.exists(), "Phase10 exposure output already exists")
    _require(not split_output.exists(), "Phase10 split output already exists")

    goals_path = args.goals.expanduser().resolve()
    goals = json.loads(goals_path.read_text(encoding="utf-8"))
    _require(isinstance(goals, list), "Phase10 goals payload must be a list")
    base_split = validate_split_lock(
        json.loads(args.base_split_lock.expanduser().resolve().read_text(encoding="utf-8"))
    )
    _require(sha256_json(goals) == base_split["goals_canonical_sha256"], "Phase10 goals/base split drift")
    source_spec_path = args.source_spec.expanduser().resolve()
    source_spec = json.loads(source_spec_path.read_text(encoding="utf-8"))

    exposure = build_exposure_union(
        goals=goals,
        source_spec=source_spec,
        source_base_dir=source_spec_path.parent,
        producer_git_sha=current_git,
    )
    split = build_phase10_split(
        goals=goals,
        train_task_ids=base_split["roles"]["train"]["task_ids"],
        exposure_union=exposure,
        base_split_content_sha256=base_split["content_sha256"],
        producer_git_sha=current_git,
    )
    publish_immutable_json(exposure_output, exposure)
    publish_immutable_json(split_output, split)
    print(json.dumps({
        "exposure_output": str(exposure_output),
        "exposed_task_count": exposure["exposed_task_count"],
        "exposure_content_sha256": exposure["content_sha256"],
        "split_output": str(split_output),
        "split_content_sha256": split["content_sha256"],
        "role_counts": {role: item["count"] for role, item in split["roles"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
