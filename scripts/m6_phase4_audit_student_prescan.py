#!/usr/bin/env python3
"""Audit student K4 prescan data and freeze RL/retention/teacher partitions."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import m6_phase2_reward_probe as phase2  # noqa: E402
from m6_phase4_build_data_rosters import _bucket  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import validate_split_lock  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402

PARTITIONS = ("a", "b")
TASKS_PER_PARTITION = 64
K = 4
MINIMUM_TOTAL_MIXED_GROUPS = 64
MINIMUM_PARTITION_MIXED_GROUPS = 24
MINIMUM_STRICT_FAILURE_PAIRS = 128
MINIMUM_STRICT_PARTIAL_PAIRS = 64
MINIMUM_FAILURE_QUALITY_PAIRS = 32
MINIMUM_FAILURE_SPREAD_GROUPS = 32
MINIMUM_FAILURE_CLASS_COUNT = 3
MINIMUM_FAILURES_PER_CLASS = 4
TEACHER_TRIGGER_ALL_FAILURE_TASKS = 16


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def _load_hashed_json(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("content_sha256") == _self_hash(value), f"M6 Phase4 self-hash drift: {path}")
    if schema is not None:
        _require(value.get("schema_version") == schema, f"M6 Phase4 schema drift: {path}")
    return value


def _load_jsonl_task_ids(paths: list[Path]) -> set[str]:
    task_ids: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                task_ids.add(str(row["task_id"]))
    return task_ids


def _parse_collections(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        _require(separator == "=" and name in PARTITIONS and path, "M6 Phase4 collection must be a=<path> or b=<path>")
        _require(name not in output, "M6 Phase4 collection partition duplicated")
        output[name] = Path(path).expanduser().resolve()
    _require(set(output) == set(PARTITIONS), "M6 Phase4 requires both student partitions")
    return output


def _annotation(
    trajectory: Mapping[str, Any],
    *,
    partition: str,
    group_id: str,
    task_id: str,
) -> dict[str, Any]:
    reward = phase2.trajectory_rewards(trajectory)
    readiness = [
        phase2._readiness_from_evidence(
            turn["verifier_progress_evidence"],
            formula_version=phase2.EQUAL_BLOCK_FORMULA,
        )
        for turn in trajectory["turns"][:-1]
    ]
    deltas = [value - (readiness[index - 1] if index else 0.0) for index, value in enumerate(readiness)]
    row = {
        "schema_version": "m6_phase4_student_trajectory_annotation_v1",
        "partition": partition,
        "group_id": group_id,
        "task_id": task_id,
        "trajectory_id": trajectory["trajectory_id"],
        "rollout_index": int(trajectory["rollout_index"]),
        "strict_reward": float(reward["strict_reward"]),
        "task_score": float(trajectory.get("task_score", 0.0)),
        "failure_class": reward["failure_class"],
        "failure_quality": float(reward["failure_quality"]),
        "termination_reason": str(trajectory.get("termination_reason", "")),
        "model_turn_count": len(trajectory["turns"]),
        "generated_action_tokens": sum(len(turn["generated_token_ids"]) for turn in trajectory["turns"]),
        "preterminal_readiness": readiness,
        "preterminal_readiness_delta": deltas,
        "final_preterminal_readiness": float(reward["final_preterminal_readiness"]),
        "maximum_preterminal_readiness": float(reward["maximum_preterminal_readiness"]),
        "policy_visible_evidence_only": all(
            turn["verifier_progress_evidence"].get("policy_visible_input_only") is True
            for turn in trajectory["turns"][:-1]
        ),
        "official_dense_task_score_used_as_reward": False,
    }
    row["content_sha256"] = _self_hash(row)
    return row


def _contrast_pair(
    *,
    pair_id: str,
    pair_type: str,
    partition: str,
    group_id: str,
    task_id: str,
    preferred: Mapping[str, Any],
    rejected: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "schema_version": "m6_phase4_student_contrast_pair_v1",
        "pair_id": pair_id,
        "pair_type": pair_type,
        "partition": partition,
        "group_id": group_id,
        "task_id": task_id,
        "preferred_trajectory_id": preferred["trajectory_id"],
        "rejected_trajectory_id": rejected["trajectory_id"],
        "preferred_strict_reward": preferred["strict_reward"],
        "rejected_strict_reward": rejected["strict_reward"],
        "preferred_failure_quality": preferred["failure_quality"],
        "rejected_failure_quality": rejected["failure_quality"],
        "label_source": "official_strict_outcome_then_public_failure_quality",
    }
    row["content_sha256"] = _self_hash(row)
    return row


def _group_type(rows: list[Mapping[str, Any]]) -> str:
    _require(len(rows) == K, "M6 Phase4 subset classification requires one complete K4 group")
    strict_count = sum(float(row["strict_reward"]) == 1.0 for row in rows)
    if strict_count == K:
        return "all_success"
    if strict_count == 0:
        return "all_failure"
    return "mixed"


def _subset_index_row(
    *,
    partition: str,
    group: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assign each frozen-SFT behavior group to exactly one downstream role.

    The prescan is not reusable as an online training stream after the policy
    changes.  Mixed tasks are therefore roster candidates for fresh online
    recollection, rather than pre-authorized GRPO batches.
    """

    group_type = _group_type(rows)
    output = {
        "schema_version": "m6_phase4_student_subset_index_row_v1",
        "partition": partition,
        "group_id": group["group_id"],
        "group_content_sha256": group["content_sha256"],
        "task_id": group["task_id"],
        "group_type": group_type,
        "strict_success_count": sum(float(row["strict_reward"]) == 1.0 for row in rows),
        "student_behavior_group": True,
        "future_online_recollection_candidate": group_type == "mixed",
        "retention_and_cost_candidate": group_type == "all_success",
        "conditional_teacher_supplement_candidate": group_type == "all_failure",
        "prescan_rollouts_reusable_as_online_batches_after_policy_update": False,
        "teacher_generated_rollouts_allowed_for_on_policy_grpo": False,
    }
    output["content_sha256"] = _self_hash(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster-manifest", type=Path, required=True)
    parser.add_argument("--collection", action="append", required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--sft-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--exclude-roster", type=Path, action="append", default=[])
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.roster_manifest.expanduser().resolve()
    manifest = _load_hashed_json(manifest_path, schema="m6_phase4_data_roster_manifest_v1")
    collections = _parse_collections(args.collection)
    split = validate_split_lock(json.loads(args.split_lock.read_text(encoding="utf-8")))
    _require(manifest["source_split_lock_content_sha256"] == split["content_sha256"], "M6 Phase4 manifest/split drift")
    goals = json.loads(args.goals.read_text(encoding="utf-8"))
    _require(sha256_json(goals) == split["goals_canonical_sha256"], "M6 Phase4 goals/split drift")
    goal_map = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    sft_ids = _load_jsonl_task_ids(args.sft_jsonl)
    prior_ids = {
        task_id
        for path in args.exclude_roster
        for task_id in _load_hashed_json(path)["task_ids"]
    }
    nontrain_ids = set().union(*(
        set(value["task_ids"])
        for role, value in split["roles"].items()
        if role != "train"
    ))
    adapter_sha256 = directory_sha256(args.sft_adapter.expanduser().resolve())

    annotations: list[dict[str, Any]] = []
    contrast_pairs: list[dict[str, Any]] = []
    teacher_candidates: list[dict[str, Any]] = []
    subset_index_rows: list[dict[str, Any]] = []
    selected_ids: dict[str, list[str]] = {}
    collection_bindings: dict[str, dict[str, Any]] = {}
    group_type_counts: Counter[str] = Counter()
    partition_group_types: dict[str, Counter[str]] = {name: Counter() for name in PARTITIONS}
    failure_spread_groups = 0

    for partition in PARTITIONS:
        roster_path = manifest_path.parent / manifest["partitions"][partition]["file"]
        roster = _load_hashed_json(roster_path, schema="m6_rl_curriculum_v1")
        _require(roster["content_sha256"] == manifest["partitions"][partition]["roster_content_sha256"], "M6 Phase4 roster/manifest drift")
        _require(roster["partition"] == partition and roster["task_count"] == TASKS_PER_PARTITION, "M6 Phase4 partition roster drift")
        task_ids = list(roster["task_ids"])
        selected_ids[partition] = task_ids
        root = collections[partition]
        report = _load_hashed_json(root / "collection_report.json")
        invocation = _load_hashed_json(root / "invocation.json")
        _require(
            report.get("complete") is True
            and report.get("mode") == "phase4_data_synthesis"
            and report.get("role") == "train"
            and report.get("K") == K
            and report.get("task_count") == TASKS_PER_PARTITION
            and report.get("trajectory_count") == TASKS_PER_PARTITION * K,
            "M6 Phase4 collection shape drift",
        )
        _require(report["task_order_sha256"] == roster["task_order_sha256"], "M6 Phase4 collection task order drift")
        _require(report["task_roster_content_sha256"] == roster["content_sha256"], "M6 Phase4 collection roster binding drift")
        _require(report["split_lock_content_sha256"] == split["content_sha256"], "M6 Phase4 collection split drift")
        _require(report["training_updates_allowed"] is False, "M6 Phase4 collection enabled updates")
        _require(report["policy_lineage"]["adapter_sha256"] == adapter_sha256, "M6 Phase4 collection SFT adapter drift")
        _require(invocation["content_sha256"] == report["invocation_content_sha256"], "M6 Phase4 invocation/report drift")
        _require(invocation["max_model_turns"] == 18 and invocation["max_environment_steps"] == 15, "M6 Phase4 horizon drift")
        _require(invocation["training_updates_allowed"] is False, "M6 Phase4 invocation enabled updates")

        groups = []
        for index, task_id in enumerate(task_ids):
            group = validate_committed_group(
                json.loads((root / "groups" / f"g{index:04d}.json").read_text(encoding="utf-8")),
                require_k=K,
            )
            _require(group["task_id"] == task_id, "M6 Phase4 group task order drift")
            _require(group["content_sha256"] == report["group_content_sha256"][index], "M6 Phase4 group/report hash drift")
            _require(group["training_updates_allowed"] is False, "M6 Phase4 group enabled updates")
            groups.append(group)

        for group in groups:
            group_annotations = [
                _annotation(
                    trajectory,
                    partition=partition,
                    group_id=group["group_id"],
                    task_id=group["task_id"],
                )
                for trajectory in group["trajectories"]
            ]
            annotations.extend(group_annotations)
            strict = [row for row in group_annotations if row["strict_reward"] == 1.0]
            failures = [row for row in group_annotations if row["strict_reward"] == 0.0]
            group_type = _group_type(group_annotations)
            subset_index_rows.append(_subset_index_row(
                partition=partition,
                group=group,
                rows=group_annotations,
            ))
            group_type_counts[group_type] += 1
            partition_group_types[partition][group_type] += 1
            if len(failures) >= 2 and max(row["failure_quality"] for row in failures) - min(row["failure_quality"] for row in failures) >= 0.1:
                failure_spread_groups += 1

            for preferred in strict:
                for rejected in failures:
                    pair_type = f"strict_over_{rejected['failure_class']}"
                    contrast_pairs.append(_contrast_pair(
                        pair_id=f"{partition}:{group['group_id']}:{preferred['rollout_index']}:{rejected['rollout_index']}:strict",
                        pair_type=pair_type,
                        partition=partition,
                        group_id=group["group_id"],
                        task_id=group["task_id"],
                        preferred=preferred,
                        rejected=rejected,
                    ))
            for left, right in itertools.combinations(failures, 2):
                if abs(left["failure_quality"] - right["failure_quality"]) < 0.1:
                    continue
                preferred, rejected = (left, right) if left["failure_quality"] > right["failure_quality"] else (right, left)
                contrast_pairs.append(_contrast_pair(
                    pair_id=f"{partition}:{group['group_id']}:{preferred['rollout_index']}:{rejected['rollout_index']}:quality",
                    pair_type="failure_quality_order",
                    partition=partition,
                    group_id=group["group_id"],
                    task_id=group["task_id"],
                    preferred=preferred,
                    rejected=rejected,
                ))

            if group_type == "all_failure":
                maximum_readiness = max(row["maximum_preterminal_readiness"] for row in failures)
                candidate = {
                    "schema_version": "m6_phase4_teacher_candidate_v1",
                    "partition": partition,
                    "group_id": group["group_id"],
                    "task_id": group["task_id"],
                    "bucket": _bucket(goal_map[group["task_id"]]),
                    "student_K": K,
                    "student_strict_success_count": 0,
                    "maximum_public_readiness": maximum_readiness,
                    "failure_classes": dict(Counter(row["failure_class"] for row in failures)),
                    "teacher_use": "verified_recovery_sft_or_preference_only_not_on_policy_grpo",
                }
                candidate["content_sha256"] = _self_hash(candidate)
                teacher_candidates.append(candidate)

        collection_bindings[partition] = {
            "collection_root": str(root),
            "collection_report_content_sha256": report["content_sha256"],
            "invocation_content_sha256": invocation["content_sha256"],
            "roster_content_sha256": roster["content_sha256"],
            "group_content_sha256": list(report["group_content_sha256"]),
            "policy_lineage": dict(report["policy_lineage"]),
            "report_training_updates_allowed": report["training_updates_allowed"],
            "invocation_training_updates_allowed": invocation["training_updates_allowed"],
        }

    all_selected = set(selected_ids["a"]) | set(selected_ids["b"])
    _require(len(all_selected) == 2 * TASKS_PER_PARTITION, "M6 Phase4 selected task union drift")
    failure_counts = Counter(row["failure_class"] for row in annotations if row["strict_reward"] == 0.0)
    strict_count = sum(row["strict_reward"] == 1.0 for row in annotations)
    pair_counts = Counter(row["pair_type"] for row in contrast_pairs)
    strict_failure_pairs = sum(count for name, count in pair_counts.items() if name.startswith("strict_over_"))
    strict_partial_pairs = pair_counts["strict_over_partial_purchase"]
    failure_quality_pairs = pair_counts["failure_quality_order"]
    represented_failure_classes = sum(count >= MINIMUM_FAILURES_PER_CLASS for count in failure_counts.values())
    bucket_failure_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in teacher_candidates:
        bucket_failure_counts[candidate["bucket"]]["all_failure"] += 1
    for partition, task_ids in selected_ids.items():
        for task_id in task_ids:
            bucket_failure_counts[_bucket(goal_map[task_id])]["total"] += 1
    concentrated_all_failure_bucket = any(
        counts["total"] >= 4 and counts["all_failure"] / counts["total"] >= 0.5
        for counts in bucket_failure_counts.values()
    )

    integrity_checks = {
        "exact_128_tasks_512_trajectories": len(all_selected) == 128 and len(annotations) == 512,
        "partition_task_identity_disjoint": not (set(selected_ids["a"]) & set(selected_ids["b"])),
        "sft_task_identity_disjoint": not (all_selected & sft_ids),
        "prior_p1_task_identity_disjoint": not (all_selected & prior_ids),
        "frozen_nontrain_role_disjoint": not (all_selected & nontrain_ids),
        "all_public_evidence_only": all(row["policy_visible_evidence_only"] for row in annotations),
        "all_trajectories_preserved_without_mixed_filter": sum(group_type_counts.values()) == 128,
        "all_collections_forbid_training_updates": all(
            binding["policy_lineage"]["adapter_sha256"] == adapter_sha256
            and binding["report_training_updates_allowed"] is False
            and binding["invocation_training_updates_allowed"] is False
            for binding in collection_bindings.values()
        ),
        "all_roster_bucket_share_gaps_at_most_5pp": all(
            manifest["partitions"][name]["maximum_bucket_share_gap"] <= 0.05
            for name in PARTITIONS
        ),
    }
    contrast_checks = {
        "at_least_64_mixed_groups": group_type_counts["mixed"] >= MINIMUM_TOTAL_MIXED_GROUPS,
        "each_partition_at_least_24_mixed_groups": all(
            partition_group_types[name]["mixed"] >= MINIMUM_PARTITION_MIXED_GROUPS for name in PARTITIONS
        ),
        "at_least_128_strict_failure_pairs": strict_failure_pairs >= MINIMUM_STRICT_FAILURE_PAIRS,
        "at_least_64_strict_partial_pairs": strict_partial_pairs >= MINIMUM_STRICT_PARTIAL_PAIRS,
        "at_least_32_failure_quality_pairs": failure_quality_pairs >= MINIMUM_FAILURE_QUALITY_PAIRS,
        "at_least_32_failure_spread_groups": failure_spread_groups >= MINIMUM_FAILURE_SPREAD_GROUPS,
        "at_least_3_failure_classes_with_4_examples": represented_failure_classes >= MINIMUM_FAILURE_CLASS_COUNT,
    }
    prescan_integrity_passed = all(integrity_checks.values())
    on_policy_contrast_passed = prescan_integrity_passed and all(contrast_checks.values())
    teacher_supplement_required = (
        len(teacher_candidates) >= TEACHER_TRIGGER_ALL_FAILURE_TASKS
        or concentrated_all_failure_bucket
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_paths = [
        output_dir / "trajectory_annotations.json",
        output_dir / "contrast_pairs.json",
        output_dir / "subset_index.json",
        output_dir / "teacher_candidates.json",
        output_dir / "dataset_audit.json",
    ]
    _require(not any(path.exists() for path in target_paths), "M6 Phase4 audit output already exists")
    annotation_payload = {
        "schema_version": "m6_phase4_student_trajectory_annotations_v1",
        "rows": annotations,
    }
    annotation_payload["content_sha256"] = _self_hash(annotation_payload)
    pair_payload = {
        "schema_version": "m6_phase4_student_contrast_pairs_v1",
        "rows": contrast_pairs,
    }
    pair_payload["content_sha256"] = _self_hash(pair_payload)
    subset_payload = {
        "schema_version": "m6_phase4_student_subset_index_v1",
        "student_behavior_policy": "frozen_sft_adapter",
        "online_training_contract": (
            "Mixed task identities may seed a future roster, but every online update must recollect "
            "from the then-current policy. Prescan trajectories are not a reusable on-policy stream."
        ),
        "rows": subset_index_rows,
    }
    subset_payload["content_sha256"] = _self_hash(subset_payload)
    teacher_payload = {
        "schema_version": "m6_phase4_teacher_candidates_v1",
        "teacher_policy_role": "conditional_verified_data_generator_not_student_policy",
        "teacher_trajectories_allowed_for_on_policy_grpo": False,
        "rows": teacher_candidates,
    }
    teacher_payload["content_sha256"] = _self_hash(teacher_payload)
    report = {
        "schema_version": "m6_phase4_student_prescan_audit_v1",
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "roster_manifest_content_sha256": manifest["content_sha256"],
        "collection_bindings": collection_bindings,
        "sft_adapter_sha256": adapter_sha256,
        "task_count": len(all_selected),
        "trajectory_count": len(annotations),
        "group_type_counts": dict(group_type_counts),
        "partition_group_type_counts": {name: dict(value) for name, value in partition_group_types.items()},
        "strict_success_trajectory_count": strict_count,
        "failure_class_counts": dict(failure_counts),
        "contrast_pair_counts": dict(pair_counts),
        "strict_failure_pair_count": strict_failure_pairs,
        "strict_partial_pair_count": strict_partial_pairs,
        "failure_quality_pair_count": failure_quality_pairs,
        "failure_spread_group_count": failure_spread_groups,
        "subset_group_count": len(subset_index_rows),
        "teacher_candidate_task_count": len(teacher_candidates),
        "teacher_trigger_all_failure_task_threshold": TEACHER_TRIGGER_ALL_FAILURE_TASKS,
        "concentrated_all_failure_bucket": concentrated_all_failure_bucket,
        "integrity_checks": integrity_checks,
        "contrast_checks": contrast_checks,
        "decision": {
            "student_prescan_integrity_passed": prescan_integrity_passed,
            "future_online_mixed_task_roster_ready": on_policy_contrast_passed,
            "prescan_rollouts_reusable_as_online_batches_after_policy_update": False,
            "future_online_updates_require_current_policy_recollection": True,
            "retention_subset_available": group_type_counts["all_success"] > 0,
            "teacher_supplement_required": teacher_supplement_required,
            "teacher_data_must_be_strict_and_replay_verified": True,
            "teacher_data_must_not_enter_on_policy_grpo": True,
        },
        "trajectory_annotations_content_sha256": annotation_payload["content_sha256"],
        "contrast_pairs_content_sha256": pair_payload["content_sha256"],
        "subset_index_content_sha256": subset_payload["content_sha256"],
        "teacher_candidates_content_sha256": teacher_payload["content_sha256"],
        "interpretation_guardrail": "This prescan measures student exploration support. It performs no optimization and does not establish RL improvement.",
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(output_dir / "trajectory_annotations.json", annotation_payload)
    atomic_write_json(output_dir / "contrast_pairs.json", pair_payload)
    atomic_write_json(output_dir / "subset_index.json", subset_payload)
    atomic_write_json(output_dir / "teacher_candidates.json", teacher_payload)
    atomic_write_json(output_dir / "dataset_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
