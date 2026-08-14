#!/usr/bin/env python3
"""Replay-verify the Phase4 larger-model probe and admit a bounded supplement."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import m6_phase2_reward_probe as phase2  # noqa: E402
from m6_phase4_build_teacher_roster import ALLOWED_TEACHER_MODELS  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.actions import WebShopCommand, normalize_command  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402

K = 4
TASK_COUNT = 16
MINIMUM_VERIFIED_TASKS = 4
MINIMUM_VERIFIED_DISTINCT_TRAJECTORIES = 8
MAXIMUM_SUPPLEMENT_PER_TASK = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_hashed(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    _require(value.get("content_sha256") == _self_hash(value), f"M6 Phase4 teacher hash drift: {path}")
    if schema is not None:
        _require(value.get("schema_version") == schema, f"M6 Phase4 teacher schema drift: {path}")
    return value


def _command_sequence(episode: Mapping[str, Any]) -> list[str]:
    commands = []
    for turn in episode["turns"]:
        action = turn.get("action")
        if turn.get("schema_valid") is not True or not isinstance(action, Mapping):
            continue
        commands.append(normalize_command(str(action["command"])))
    _require(commands, "M6 Phase4 strict teacher episode has no replayable commands")
    return commands


def _replay_success(episode: Mapping[str, Any], *, base_url: str) -> bool:
    environment = WebShopHTTPEnvironment(base_url=base_url, split="train", timeout_seconds=120)
    try:
        environment.reset(str(episode["task_id"]))
        last = None
        for command in _command_sequence(episode):
            last = environment.step(WebShopCommand(command))
            if last.terminated or last.truncated:
                break
        return bool(last is not None and last.terminated and last.info.get("success") is True)
    finally:
        environment.close()


def _verified_row(
    episode: Mapping[str, Any],
    *,
    source_sha256: str,
    teacher_model: Path,
) -> dict[str, Any]:
    commands = _command_sequence(episode)
    row = dict(episode)
    row.pop("content_sha256", None)
    row.update({
        "schema_version": "m6_phase4_verified_teacher_trajectory_v1",
        "source_episode_content_sha256": source_sha256,
        "teacher_model": str(teacher_model),
        "strict_success": True,
        "replay_success": True,
        "normalized_command_sequence": commands,
        "normalized_command_sequence_sha256": sha256_json(commands),
        "allowed_for_on_policy_grpo": False,
        "allowed_uses": ["recovery_sft", "preference_learning", "offline_diagnostic"],
    })
    row["content_sha256"] = _self_hash(row)
    return row


def _select_distinct(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    selected = []
    for task_id in sorted(by_task):
        seen_sequences: set[str] = set()
        ranked = sorted(
            by_task[task_id],
            key=lambda row: (
                int(row.get("environment_steps", 0)),
                int(row.get("generated_action_tokens", 0)),
                str(row["normalized_command_sequence_sha256"]),
                str(row["trajectory_id"]),
            ),
        )
        for row in ranked:
            sequence_sha256 = str(row["normalized_command_sequence_sha256"])
            if sequence_sha256 in seen_sequences:
                continue
            seen_sequences.add(sequence_sha256)
            selected.append(dict(row))
            if len(seen_sequences) == MAXIMUM_SUPPLEMENT_PER_TASK:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-roster", type=Path, required=True)
    parser.add_argument("--student-audit", type=Path, required=True)
    parser.add_argument("--teacher-candidates", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    roster = _load_hashed(args.teacher_roster, schema="m6_rl_curriculum_v1")
    teacher_model = Path(str(roster["teacher_model"])).expanduser().resolve()
    _require(teacher_model in ALLOWED_TEACHER_MODELS, "M6 Phase4 teacher model is not a frozen candidate")
    student_audit = _load_hashed(args.student_audit, schema="m6_phase4_student_prescan_audit_v1")
    candidates = _load_hashed(args.teacher_candidates, schema="m6_phase4_teacher_candidates_v1")
    _require(roster["source_student_audit_content_sha256"] == student_audit["content_sha256"], "M6 Phase4 teacher roster/student audit drift")
    _require(roster["source_teacher_candidates_content_sha256"] == candidates["content_sha256"], "M6 Phase4 teacher roster/candidate drift")
    _require(roster["teacher_data_allowed_for_on_policy_grpo"] is False, "M6 Phase4 teacher roster permits on-policy use")
    _require(roster["task_count"] == TASK_COUNT and roster["probe_K"] == K, "M6 Phase4 teacher roster shape drift")
    _require(roster["minimum_verified_strict_success_tasks"] == MINIMUM_VERIFIED_TASKS, "M6 Phase4 teacher task gate drift")
    _require(
        roster["minimum_verified_distinct_strict_success_trajectories"] == MINIMUM_VERIFIED_DISTINCT_TRAJECTORIES,
        "M6 Phase4 teacher trajectory gate drift",
    )
    _require(roster["maximum_supplement_trajectories_per_task"] == MAXIMUM_SUPPLEMENT_PER_TASK, "M6 Phase4 teacher supplement cap drift")
    candidate_by_task = {str(row["task_id"]): row for row in candidates["rows"]}
    _require(set(roster["task_ids"]) <= set(candidate_by_task), "M6 Phase4 teacher roster escaped all-failure candidates")
    _require(all(candidate_by_task[task_id]["student_strict_success_count"] == 0 for task_id in roster["task_ids"]), "M6 Phase4 teacher comparison lacks a zero-success student baseline")

    root = args.collection_root.expanduser().resolve()
    report = _load_hashed(root / "collection_report.json")
    invocation = _load_hashed(root / "invocation.json")
    _require(
        report.get("complete") is True
        and report.get("mode") == "phase4_teacher_probe"
        and report.get("role") == "train"
        and report.get("task_count") == TASK_COUNT
        and report.get("trajectory_count") == TASK_COUNT * K
        and report.get("K") == K,
        "M6 Phase4 teacher collection shape drift",
    )
    _require(report["training_updates_allowed"] is False and invocation["training_updates_allowed"] is False, "M6 Phase4 teacher collection enabled updates")
    _require(report["invocation_content_sha256"] == invocation["content_sha256"], "M6 Phase4 teacher report/invocation drift")
    _require(
        report["git_sha"] == invocation["git_sha"] == roster["git_sha"],
        "M6 Phase4 teacher Git lineage drift",
    )
    _require(
        report["protocol_sha256"] == invocation["protocol_sha256"] == roster["protocol_sha256"],
        "M6 Phase4 teacher protocol lineage drift",
    )
    _require(
        report["split_lock_content_sha256"]
        == invocation["split_lock_content_sha256"]
        == roster["source_split_lock_content_sha256"],
        "M6 Phase4 teacher split lineage drift",
    )
    _require(
        invocation["adapter"] is None and invocation["base_model"] == str(teacher_model),
        "M6 Phase4 teacher policy identity drift",
    )
    _require(invocation["seed"] == 20260826, "M6 Phase4 teacher sampling seed drift")
    _require(invocation["max_model_turns"] == 18 and invocation["max_environment_steps"] == 15, "M6 Phase4 teacher horizon drift")
    _require(invocation["base_model_manifest_sha256"] == roster["teacher_base_model_manifest_sha256"], "M6 Phase4 teacher manifest drift")
    _require(report["policy_lineage"]["adapter_sha256"] == roster["teacher_base_model_manifest_sha256"], "M6 Phase4 teacher collection manifest lineage drift")
    _require(report["policy_lineage"]["rollout_adapter_sha256"] == roster["teacher_base_model_functional_sha256"], "M6 Phase4 teacher functional lineage drift")
    _require(report["task_roster_content_sha256"] == roster["content_sha256"], "M6 Phase4 teacher collection roster drift")
    _require(report["task_order_sha256"] == roster["task_order_sha256"], "M6 Phase4 teacher task order drift")

    trajectory_ids: set[str] = set()
    failure_classes: Counter[str] = Counter()
    for index, task_id in enumerate(roster["task_ids"]):
        group = validate_committed_group(
            json.loads((root / "groups" / f"g{index:04d}.json").read_text(encoding="utf-8")),
            require_k=K,
        )
        _require(group["task_id"] == task_id, "M6 Phase4 teacher group task order drift")
        _require(group["content_sha256"] == report["group_content_sha256"][index], "M6 Phase4 teacher group/report hash drift")
        _require(group["training_updates_allowed"] is False, "M6 Phase4 teacher group enabled updates")
        _require(
            group["git_sha"] == report["git_sha"]
            and group["protocol_sha256"] == report["protocol_sha256"],
            "M6 Phase4 teacher group source lineage drift",
        )
        _require(
            group["adapter_sha256"] == report["policy_lineage"]["adapter_sha256"]
            and group["rollout_adapter_sha256"]
            == report["policy_lineage"]["rollout_adapter_sha256"]
            and group["adapter_semantic_sha256"]
            == report["policy_lineage"]["adapter_semantic_sha256"],
            "M6 Phase4 teacher group policy lineage drift",
        )
        for trajectory in group["trajectories"]:
            trajectory_ids.add(str(trajectory["trajectory_id"]))
            reward = phase2.trajectory_rewards(trajectory)
            if reward["strict_reward"] == 0.0:
                failure_classes[reward["failure_class"]] += 1

    episode_paths = sorted((root / "episodes").glob("*.json"))
    _require(len(episode_paths) == TASK_COUNT * K, "M6 Phase4 teacher episode count drift")
    verified_rows = []
    strict_episode_count = 0
    strict_episode_tasks: set[str] = set()
    replay_failure_count = 0
    seen_episode_ids: set[str] = set()
    for path in episode_paths:
        episode = _load_hashed(path)
        trajectory_id = str(episode.get("trajectory_id"))
        _require(trajectory_id in trajectory_ids and trajectory_id not in seen_episode_ids, "M6 Phase4 teacher episode/group identity drift")
        seen_episode_ids.add(trajectory_id)
        if episode.get("success") is not True or float(episode.get("task_score", 0.0)) < 0.999:
            continue
        strict_episode_count += 1
        strict_episode_tasks.add(str(episode["task_id"]))
        replay_success = _replay_success(episode, base_url=args.base_url)
        if not replay_success:
            replay_failure_count += 1
            continue
        verified_rows.append(
            _verified_row(
                episode,
                source_sha256=episode["content_sha256"],
                teacher_model=teacher_model,
            )
        )
    _require(seen_episode_ids == trajectory_ids, "M6 Phase4 teacher episode roster incomplete")
    _require(
        int(report["strict_success_trajectory_count"]) == strict_episode_count
        and int(report["strict_success_task_count"]) == len(strict_episode_tasks),
        "M6 Phase4 teacher report/episode strict-success count drift",
    )

    selected_rows = _select_distinct(verified_rows)
    verified_tasks = {row["task_id"] for row in verified_rows}
    selected_tasks = {row["task_id"] for row in selected_rows}
    strength_gate = (
        replay_failure_count == 0
        and len(verified_tasks) >= MINIMUM_VERIFIED_TASKS
        and len(selected_tasks) >= MINIMUM_VERIFIED_TASKS
        and len(selected_rows) >= MINIMUM_VERIFIED_DISTINCT_TRAJECTORIES
    )
    admitted_rows = selected_rows if strength_gate else []
    verified_payload = {
        "schema_version": "m6_phase4_verified_teacher_probe_successes_v1",
        "teacher_model": str(teacher_model),
        "allowed_for_on_policy_grpo": False,
        "rows": verified_rows,
    }
    verified_payload["content_sha256"] = _self_hash(verified_payload)
    supplement_payload = {
        "schema_version": "m6_phase4_verified_teacher_supplement_v1",
        "admitted": strength_gate,
        "teacher_model": str(teacher_model),
        "allowed_for_on_policy_grpo": False,
        "maximum_trajectories_per_task": MAXIMUM_SUPPLEMENT_PER_TASK,
        "rows": admitted_rows,
    }
    supplement_payload["content_sha256"] = _self_hash(supplement_payload)
    audit = {
        "schema_version": "m6_phase4_teacher_probe_audit_v1",
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "student_audit_content_sha256": student_audit["content_sha256"],
        "teacher_roster_content_sha256": roster["content_sha256"],
        "collection_report_content_sha256": report["content_sha256"],
        "invocation_content_sha256": invocation["content_sha256"],
        "teacher_model": str(teacher_model),
        "teacher_base_model_manifest_sha256": roster["teacher_base_model_manifest_sha256"],
        "task_count": TASK_COUNT,
        "trajectory_count": TASK_COUNT * K,
        "student_strict_success_task_count_on_probe": 0,
        "student_strict_success_trajectory_count_on_probe": 0,
        "teacher_strict_success_task_count": int(report["strict_success_task_count"]),
        "teacher_strict_success_trajectory_count": strict_episode_count,
        "verified_strict_success_task_count": len(verified_tasks),
        "verified_strict_success_trajectory_count": len(verified_rows),
        "strict_replay_failure_count": replay_failure_count,
        "selected_distinct_task_count": len(selected_tasks),
        "selected_distinct_trajectory_count": len(selected_rows),
        "failure_class_counts": dict(failure_classes),
        "minimum_verified_task_gate": MINIMUM_VERIFIED_TASKS,
        "minimum_verified_distinct_trajectory_gate": MINIMUM_VERIFIED_DISTINCT_TRAJECTORIES,
        "maximum_supplement_per_task": MAXIMUM_SUPPLEMENT_PER_TASK,
        "verified_probe_successes_content_sha256": verified_payload["content_sha256"],
        "teacher_supplement_content_sha256": supplement_payload["content_sha256"],
        "decision": {
            "teacher_demonstrably_stronger_than_student_on_probe": strength_gate,
            "teacher_supplement_admitted": strength_gate,
            "teacher_data_allowed_for_on_policy_grpo": False,
            "next_step": "USE_VERIFIED_TEACHER_SUPPLEMENT" if strength_gate else "STOP_NO_STRONG_TEACHER_EVIDENCE",
        },
    }
    audit["content_sha256"] = _self_hash(audit)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / "verified_probe_successes.json",
        output / "teacher_supplement.json",
        output / "teacher_probe_audit.json",
    ]
    _require(not any(path.exists() for path in targets), "M6 Phase4 teacher audit output already exists")
    atomic_write_json(targets[0], verified_payload)
    atomic_write_json(targets[1], supplement_payload)
    atomic_write_json(targets[2], audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
