#!/usr/bin/env python3
"""Freeze a Raw-K8 difficulty curriculum for the M6-mini RL loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402
from miniwebwork.m6_pilot import (  # noqa: E402
    load_medium_rl_authorization,
    validate_pilot_authorization,
)
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-authorization", type=Path)
    parser.add_argument("--medium-authorization", type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    split = validate_split_lock(json.loads(args.split_lock.read_text(encoding="utf-8")))
    _require(split["protocol_sha256"] == protocol["sha256"], "M6 curriculum split/protocol drift")
    mini_train = set(split["roles"]["mini_train"]["task_ids"])
    groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=8)
        for path in sorted(args.groups_dir.expanduser().resolve().glob("g*.json"))
    ]
    _require(groups, "M6 Raw K8 curriculum source is empty")
    _require(
        [group["task_id"] for group in groups] == split["roles"]["mini_train"]["task_ids"],
        "M6 curriculum groups are not the complete frozen mini_train order",
    )
    pilot = None
    if args.pilot_authorization is not None:
        pilot = validate_pilot_authorization(json.loads(args.pilot_authorization.read_text(encoding="utf-8")))
        expected_producer_git = pilot["source_producer_git_sha"]
    else:
        expected_producer_git = protocol["git_sha"]
    _require(all(group.get("git_sha") == expected_producer_git for group in groups), "M6 curriculum group Git drift")
    lineage_fields = ("adapter_sha256", "rollout_adapter_sha256", "adapter_semantic_sha256")
    source_lineage = {field: groups[0][field] for field in lineage_fields}
    _require(
        all(all(group[field] == source_lineage[field] for field in lineage_fields) for group in groups),
        "M6 curriculum source mixes policies",
    )
    rows = []
    for group in groups:
        task_id = group["task_id"]
        _require(task_id in mini_train, "M6 curriculum source escapes mini_train")
        _require(group.get("protocol_sha256") == protocol["sha256"], "M6 curriculum group/protocol drift")
        _require(group.get("training_updates_allowed") is False, "M6 curriculum source is update-authorized")
        success_count = sum(bool(item["success"]) for item in group["trajectories"])
        pass_rate = success_count / 8
        # K8 success of 1..6 is the pre-specified informative region.  It gives
        # a frozen, outcome-based curriculum without looking at later SFT/RL
        # results or hidden goal fields.
        if 1 <= success_count <= 6:
            rows.append(
                {
                    "task_id": task_id,
                    "raw_k8_success_count": success_count,
                    "raw_k8_pass_rate": pass_rate,
                    "source_group_content_sha256": group["content_sha256"],
                }
            )
    minimum = int(protocol["payload"]["mini"]["minimum_mixed_rl_iterations"])
    maximum = int(protocol["payload"]["mini"]["maximum_rl_iterations"])
    medium = None
    if args.medium_authorization is not None:
        _require(pilot is not None, "M6 medium curriculum requires the pilot authorization")
        medium = load_medium_rl_authorization(args.medium_authorization)
        maximum = int(medium["payload"]["shared_controls"]["curriculum_candidate_tasks"])
        minimum = int(medium["payload"]["shared_controls"]["target_mixed_optimizer_updates"])
    seed = int(protocol["payload"]["split"]["selection_seed"])
    rows.sort(
        key=lambda item: (
            hashlib.sha256(f"m6-rl-curriculum-v1|{seed}|{item['task_id']}".encode()).digest(),
            item["task_id"],
        )
    )
    _require(len(rows) >= minimum, "M6 Raw K8 has too few informative tasks for five mixed RL iterations")
    rows = rows[:maximum]
    report = {
        "schema_version": "m6_rl_curriculum_v1",
        "study_id": protocol["payload"]["study_id"],
        "development_only": True,
        "selection": "raw_K8_success_count_in_[1,6]_then_hash_rank_v1",
        "selection_seed": seed,
        "source_split_lock_content_sha256": split["content_sha256"],
        "git_sha": protocol["git_sha"],
        "source_producer_git_sha": expected_producer_git,
        "protocol_sha256": protocol["sha256"],
        "pilot_authorization_content_sha256": pilot["content_sha256"] if pilot is not None else None,
        "medium_authorization_file_sha256": medium["sha256"] if medium is not None else None,
        "source_policy_lineage": source_lineage,
        "source_group_content_sha256": [group["content_sha256"] for group in groups],
        "task_count": len(rows),
        "task_ids": [item["task_id"] for item in rows],
        "rows": rows,
    }
    report["content_sha256"] = sha256_json(report)
    publish_immutable_json(args.output, report)
    print(json.dumps({"path": str(args.output.resolve()), "task_count": len(rows), "content_sha256": report["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
