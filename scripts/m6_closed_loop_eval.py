#!/usr/bin/env python3
"""Summarize one completed M6 mini-dev K4 identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m6_mini import build_evaluation_semantics, summarize_closed_loop_identity  # noqa: E402
from miniwebwork.m6_pilot import validate_pilot_authorization  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", choices=("raw", "mini_sft", "mini_rl"), required=True)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-authorization", type=Path)
    parser.add_argument("--pilot-binding-output", type=Path)
    args = parser.parse_args()
    if (args.pilot_authorization is None) != (args.pilot_binding_output is None):
        raise ValueError("M6 pilot authorization and binding output must be provided together")
    protocol = load_protocol()
    split = validate_split_lock(json.loads(args.split_lock.read_text(encoding="utf-8")))
    if split.get("protocol_sha256") != protocol["sha256"]:
        raise ValueError("M6 evaluation split/protocol drift")
    collection = json.loads(args.collection_report.read_text(encoding="utf-8"))
    expected_collection = dict(collection)
    observed_collection = expected_collection.pop("content_sha256", None)
    if observed_collection != sha256_json(expected_collection):
        raise ValueError("M6 evaluation collection report self-hash drift")
    if collection.get("complete") is not True or collection.get("mode") != "evaluation" or collection.get("K") != 4:
        raise ValueError("M6 evaluation collection report is incomplete")
    if collection.get("role") != "mini_dev":
        raise ValueError("M6 evaluation did not use frozen mini_dev")
    if collection.get("split_lock_content_sha256") != split["content_sha256"]:
        raise ValueError("M6 evaluation collection/split drift")
    paths = sorted(args.groups_dir.expanduser().resolve().glob("g*.json"))
    groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=4)
        for path in paths
    ]
    expected_tasks = split["roles"]["mini_dev"]["task_ids"]
    if [group["task_id"] for group in groups] != expected_tasks:
        raise ValueError("M6 evaluation group roster is not the frozen mini_dev order")
    if collection.get("group_content_sha256") != [group["content_sha256"] for group in groups]:
        raise ValueError("M6 evaluation collection/group binding drift")
    invocation_path = args.collection_report.parent / "invocation.json"
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    expected_invocation = dict(invocation)
    observed_invocation = expected_invocation.pop("content_sha256", None)
    if observed_invocation != sha256_json(expected_invocation):
        raise ValueError("M6 evaluation invocation self-hash drift")
    if collection.get("invocation_content_sha256") != invocation.get("content_sha256"):
        raise ValueError("M6 evaluation invocation/report drift")
    evaluation_semantics = build_evaluation_semantics(
        collection=collection,
        invocation=invocation,
        split_lock_content_sha256=split["content_sha256"],
        protocol=protocol["payload"],
    )
    report = summarize_closed_loop_identity(
        identity=args.identity,
        groups=groups,
        development_only=args.identity != "raw",
        evaluation_contract_sha256=evaluation_semantics["content_sha256"],
        source_bindings={
            "producer_git_sha": collection["git_sha"],
            "report_consumer_git_sha": protocol["git_sha"],
            "collection_report_content_sha256": collection["content_sha256"],
            "invocation_content_sha256": invocation["content_sha256"],
            "split_lock_content_sha256": split["content_sha256"],
            "policy_lineage": collection["policy_lineage"],
            "group_content_sha256": collection["group_content_sha256"],
            "evaluation_semantics": evaluation_semantics,
        },
    )
    atomic_write_json(args.output, report)
    if args.pilot_authorization is not None:
        authorization = validate_pilot_authorization(
            json.loads(args.pilot_authorization.read_text(encoding="utf-8"))
        )
        binding = {
            "schema_version": "m6_mini_pilot_evaluation_binding_v1",
            "identity": report["identity"],
            "development_only": True,
            "identity_report_content_sha256": report["content_sha256"],
            "pilot_authorization_content_sha256": authorization["content_sha256"],
        }
        binding["content_sha256"] = sha256_json(binding)
        atomic_write_json(args.pilot_binding_output, binding)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
