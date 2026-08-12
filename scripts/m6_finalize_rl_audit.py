#!/usr/bin/env python3
"""Aggregate M6-mini RL iterations and enforce the formal-method preflight gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m6_mini import audit_mini_rl  # noqa: E402
from miniwebwork.m6_pilot import validate_pilot_authorization, validate_pilot_method  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group, validate_learner_report  # noqa: E402
from miniwebwork.webshop_rl.verifier_td import FORMULA_VERSION_BY_METHOD, METHODS, assign_group_credit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=Path, action="append", required=True, help="Repeat learner_report.json")
    parser.add_argument("--groups-dir", type=Path, action="append", required=True, help="Repeat matching K8 groups directory")
    parser.add_argument("--collection-report", type=Path, action="append", required=True, help="Repeat every attempted curriculum collection, including no-update groups")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--pilot-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authorization = validate_pilot_authorization(
        json.loads(args.pilot_authorization.read_text(encoding="utf-8"))
    )
    validate_pilot_method(args.method, authorization)
    if len(args.iteration) != len(args.groups_dir):
        raise ValueError("M6 learner/group iteration counts differ")
    reports = [validate_learner_report(path, expected_method=args.method) for path in args.iteration]
    if [report["iteration_index"] for report in reports] != list(range(len(reports))):
        raise ValueError("M6 learner iterations are not contiguous from zero")
    reference_sha = reports[0]["reference_sft_adapter_sha256"]
    for index, report in enumerate(reports):
        if report.get("pilot_authorization_sha256") != authorization["content_sha256"]:
            raise ValueError("M6 learner pilot-authorization binding drift")
        if report["reference_sft_adapter_sha256"] != reference_sha:
            raise ValueError("M6 frozen SFT reference changed across iterations")
        if index:
            previous = reports[index - 1]
            if report["input_adapter_sha256"] != previous["output_adapter_sha256"]:
                raise ValueError("M6 RL adapter byte lineage is broken")
            if report["input_adapter_semantic_sha256"] != previous["output_adapter_semantic_sha256"]:
                raise ValueError("M6 RL adapter semantic lineage is broken")
            if report["input_optimizer_sha256"] != previous["output_optimizer_sha256"]:
                raise ValueError("M6 RL optimizer lineage is broken")
            if report["cumulative_optimizer_updates_before"] != previous["cumulative_optimizer_updates_after"]:
                raise ValueError("M6 RL optimizer update counter is discontinuous")
    credit = []
    iterations = []
    learner_group_hashes: list[list[str]] = []
    for report, group_dir in zip(reports, args.groups_dir):
        groups = [
            validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=8)
            for path in sorted(group_dir.glob("g*.json"))
        ]
        group_credit = [
            assign_group_credit(group["trajectories"], method=args.method)
            for group in groups
        ]
        group_hashes = [group["content_sha256"] for group in groups]
        expected_credit_hashes = report.get("collection_audit", {}).get(
            "credit_assignment_content_sha256"
        )
        if expected_credit_hashes != [item["content_sha256"] for item in group_credit]:
            raise ValueError("M6 learner report/credit evidence binding drift")
        learner_group_hashes.append(group_hashes)
        credit.extend(group_credit)
        iterations.append(
            {
                "iteration_index": report["iteration_index"],
                "loss": report["mean_loss"],
                "gradient_norm": report["mean_gradient_norm"],
                "observed_kl": report["observed_reference_kl"],
                "adaptive_kl_coefficient_before": report["adaptive_kl_coefficient_before"],
                "adaptive_kl_coefficient_after": report["adaptive_kl_coefficient_after"],
                "mixed_strict_reward_signal": any(
                    item["metrics"]["mixed_strict_reward_signal"] for item in group_credit
                ),
                "report_content_sha256": report["content_sha256"],
            }
        )
    latest = reports[-1]
    collection_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.collection_report]
    collection_hashes: set[str] = set()
    collection_by_group_hashes: dict[tuple[str, ...], dict] = {}
    for collection in collection_reports:
        expected_collection = dict(collection)
        observed_collection = expected_collection.pop("content_sha256", None)
        if observed_collection != sha256_json(expected_collection):
            raise ValueError("M6 collection report self-hash drift")
        if collection.get("schema_version") != "m6_rollout_collection_report_v1" or collection.get("complete") is not True:
            raise ValueError("M6 collection report is incomplete")
        if collection.get("mode") != "rl_collection" or collection.get("action_token_budget_respected") is not True:
            raise ValueError("M6 collection report is not a budgeted RL collection")
        if collection.get("training_updates_allowed") is not True:
            raise ValueError("M6 collection report is not update-authorized")
        if collection["content_sha256"] in collection_hashes:
            raise ValueError("M6 collection report was counted more than once")
        collection_hashes.add(collection["content_sha256"])
        group_hashes = collection.get("group_content_sha256")
        if not isinstance(group_hashes, list) or not group_hashes:
            raise ValueError("M6 collection report group binding is missing")
        key = tuple(str(item) for item in group_hashes)
        if key in collection_by_group_hashes:
            raise ValueError("M6 collection group roster was counted more than once")
        collection_by_group_hashes[key] = collection
    for report, group_hashes in zip(reports, learner_group_hashes):
        collection = collection_by_group_hashes.get(tuple(group_hashes))
        if collection is None:
            raise ValueError("M6 learner has no matching collection report")
        audit = report.get("collection_audit", {})
        if (
            audit.get("all_generated_action_tokens")
            != collection.get("all_attempt_generated_action_tokens")
            or audit.get("committed_group_action_tokens")
            != collection.get("generated_action_tokens")
        ):
            raise ValueError("M6 learner/collection token binding drift")
    learner = {
        "development_only": True,
        "method": args.method,
        "credit_formula_version": FORMULA_VERSION_BY_METHOD[args.method],
        "pilot_authorization_sha256": authorization["content_sha256"],
        "group_size": 8,
        "generated_action_tokens": sum(int(report["all_attempt_generated_action_tokens"]) for report in collection_reports),
        "collection_attempt_count": len(collection_reports),
        "collection_report_content_sha256": [report["content_sha256"] for report in collection_reports],
        "optimizer_updates": latest["cumulative_optimizer_updates_after"],
        "parameters_changed": all(report["input_adapter_semantic_sha256"] != report["output_adapter_semantic_sha256"] for report in reports),
        "lineage_chain_complete": True,
        "frozen_reference_sft_adapter_sha256": reference_sha,
        "parameter_sha256_before": reports[0]["input_adapter_semantic_sha256"],
        "parameter_sha256_after": latest["output_adapter_semantic_sha256"],
        "iterations": iterations,
    }
    learner["content_sha256"] = sha256_json(learner)
    report = audit_mini_rl(
        learner_report=learner,
        credit_assignments=credit,
        pilot_authorization=authorization,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
