#!/usr/bin/env python3
"""Freeze the union of M6.2 tasks that actually produced optimizer updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402


def _iteration_root(group_path: Path) -> Path:
    return group_path.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    split = validate_split_lock(json.loads(args.split_lock.read_text(encoding="utf-8")))
    if split["protocol_sha256"] != protocol["sha256"]:
        raise ValueError("M6 phase-one roster split/protocol drift")
    updated = set()
    source_groups = []
    for path in sorted(args.online_root.glob("seed_*/**/iteration_*/collection/groups/g*.json")):
        if not (_iteration_root(path) / "learner" / "learner_report.json").is_file():
            continue
        group = validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=8)
        updated.add(group["task_id"])
        source_groups.append(group["content_sha256"])
    if not updated or not updated <= set(split["roles"]["mini_train"]["task_ids"]):
        raise ValueError("M6 phase-one updated-task roster is empty or escapes mini_train")
    task_ids = sorted(updated)
    report = {
        "schema_version": "m6_rl_curriculum_v1",
        "development_only": True,
        "formal_training": False,
        "purpose": "phase1_seen_updated_task_diagnostic_only",
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "source_group_count": len(source_groups),
        "source_group_content_sha256": source_groups,
        "source_split_lock_content_sha256": split["content_sha256"],
        "protocol_sha256": protocol["sha256"],
        "git_sha": protocol["git_sha"],
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
