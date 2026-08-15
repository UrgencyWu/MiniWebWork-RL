#!/usr/bin/env python3
"""Build matched teacher/rehearsal inputs for the Phase10 one-update probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol  # noqa: E402
from miniwebwork.webshop_rl import prompt  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402
from miniwebwork.webshop_rl.m6_sft_training import M6SFTConfig, tokenize_sft_row  # noqa: E402

SOURCE_WEIGHTS = {"new": 0.60, "old_sft": 0.25, "current_student": 0.15}
SEED = 20260842


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Phase10 probe artifact is not an object: {path}")
    _require(value.get("content_sha256") == _self_hash(value), f"Phase10 probe self-hash drift: {path}")
    return value


def _load_groups(root: Path) -> dict[str, dict[str, Any]]:
    report = _load_hashed(root / "collection_report.json")
    groups = {}
    for index in range(int(report["task_count"])):
        group = validate_committed_group(_load_hashed(root / "groups" / f"g{index:04d}.json"), require_k=2)
        groups[group["task_id"]] = group
    _require(report["group_content_sha256"] == [groups[task_id]["content_sha256"] for task_id in groups], "Phase10 probe group/report drift")
    return groups


def suffix_rows(
    *,
    trajectory: Mapping[str, Any],
    prefix_turn_count: int,
    source: str,
    state_id: str,
) -> list[dict[str, Any]]:
    """Create one student-tokenizer completion row per suffix action turn."""

    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    path_id = str(trajectory["trajectory_id"])
    for index, turn in enumerate(trajectory["turns"]):
        observation = SimpleNamespace(**dict(turn["observation"]))
        messages = prompt.build_messages(observation, history)
        _require(prompt.compute_message_hash(messages) == turn["rendered_prompt_sha256"], "Phase10 probe prompt drift")
        action = turn.get("action")
        if index >= prefix_turn_count and isinstance(action, Mapping):
            completion = json.dumps(
                {"command": str(action["command"])}, ensure_ascii=False, separators=(",", ":")
            )
            rows.append({
                "schema_version": "m6_phase10_probe_action_row_v1",
                "sample_id": f"{source}:{path_id}:{index + 1:03d}",
                "task_id": trajectory["task_id"],
                "state_id": state_id,
                "path_id": path_id,
                "trajectory_id": path_id,
                "turn_index": index + 1,
                "messages": messages,
                "completion": completion,
                "trajectory_recovery": False,
                "source": source,
                "prefix_labels_masked": True,
            })
        action_result = turn.get("action_result")
        if isinstance(action, Mapping) and isinstance(action_result, Mapping):
            history.append({
                "turn": index + 1,
                "command": str(action.get("command", "")),
                "success": bool(action_result.get("success", False)),
                "error_code": str(action_result.get("error_code", ""))[:100],
                "page_type": str(turn["post_action_observation"].get("page_type", "unknown")),
            })
    _require(rows, "Phase10 probe strict suffix has no action rows")
    return rows


def _path_tokens(rows: Sequence[Mapping[str, Any]], tokenizer: Any, config: M6SFTConfig) -> int:
    return sum(tokenize_sft_row(row, tokenizer, config).completion_label_tokens for row in rows)


def _old_paths(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[str(row["trajectory_id"])].append(dict(row))
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["turn_index"]))
    return dict(grouped)


def _decorate_old(rows: Sequence[Mapping[str, Any]], *, source: str) -> list[dict[str, Any]]:
    path_id = str(rows[0]["trajectory_id"])
    task_id = str(rows[0]["task_id"])
    return [
        {
            **dict(row),
            "sample_id": f"{source}:{path_id}:{int(row['turn_index']):03d}",
            "source": source,
            "state_id": f"{source}:{task_id}",
            "path_id": path_id,
            "prefix_labels_masked": True,
        }
        for row in rows
    ]


def _select_exact_old_path(
    paths: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_rows: int,
    target_tokens: int,
    tokenizer: Any,
    config: M6SFTConfig,
    excluded_tasks: set[str],
) -> tuple[str, list[dict[str, Any]]]:
    candidates = []
    for path_id, rows in paths.items():
        if len(rows) != target_rows or str(rows[0]["task_id"]) in excluded_tasks:
            continue
        tokens = _path_tokens(rows, tokenizer, config)
        if tokens == target_tokens:
            candidates.append((sha256_json({"seed": SEED, "path_id": path_id}), path_id, [dict(row) for row in rows]))
    _require(candidates, "Phase10 probe has no exact action-row/token-matched rehearsal path")
    _, path_id, rows = min(candidates)
    return path_id, rows


def control_feasibility(
    paths: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_rows: int,
    target_tokens: int,
    tokenizer: Any,
    config: M6SFTConfig,
    excluded_tasks: set[str],
) -> dict[str, Any]:
    signatures: dict[tuple[int, int], int] = defaultdict(int)
    matching_path_ids = []
    for path_id, rows in paths.items():
        if str(rows[0]["task_id"]) in excluded_tasks:
            continue
        signature = (len(rows), _path_tokens(rows, tokenizer, config))
        signatures[signature] += 1
        if signature == (target_rows, target_tokens):
            matching_path_ids.append(path_id)
    return {
        "target": {"action_row_count": target_rows, "labeled_token_count": target_tokens},
        "eligible_old_sft_path_count": sum(signatures.values()),
        "exact_matching_old_sft_path_count": len(matching_path_ids),
        "exact_matching_path_id_hashes": sorted(sha256_json({"path_id": path_id}) for path_id in matching_path_ids),
        "observed_signature_counts": [
            {"action_row_count": rows, "labeled_token_count": tokens, "path_count": count}
            for (rows, tokens), count in sorted(signatures.items())
        ],
        "control_feasible": bool(matching_path_ids),
        "approximate_fallback_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--old-sft-jsonl", type=Path, required=True)
    parser.add_argument("--student-tokenizer", type=Path, required=True)
    parser.add_argument("--feasibility-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase10 single-update probe input exists")
    audit = _load_hashed(args.audit_report)
    _require(audit.get("passed") is True, "Phase10 engineering smoke did not pass")
    manifest = _load_hashed(args.manifest)
    specs = {row["task_id"]: row for row in manifest["tasks"]}
    student_groups = _load_groups(args.student_root.expanduser().resolve())
    teacher_groups = _load_groups(args.teacher_root.expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.student_tokenizer.expanduser().resolve()), trust_remote_code=True
    )
    config = M6SFTConfig.from_protocol(load_protocol()["payload"], seed=SEED)
    audit_rows = {
        (row["identity"], row["task_id"], int(row["rollout_index"])): row
        for row in audit["trajectories"]
    }

    def selected_smoke_rows(identity: str, groups: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        candidates = []
        for task_id, group in groups.items():
            spec = specs[task_id]
            for trajectory in group["trajectories"]:
                audit_row = audit_rows[(identity, task_id, int(trajectory["rollout_index"]))]
                if not (
                    audit_row["first_execution_strict"] is True
                    and audit_row["fresh_replay"]["strict_success"] is True
                ):
                    continue
                rows = suffix_rows(
                    trajectory=trajectory,
                    prefix_turn_count=int(spec["prefix_turn_count"]),
                    source="new" if identity == "teacher" else "current_student",
                    state_id=str(spec["correction_public_state_sha256"]),
                )
                candidates.append((task_id, int(trajectory["rollout_index"]), rows))
        _require(candidates, f"Phase10 probe has no replay-strict {identity} path")
        return min(candidates, key=lambda item: (item[0], item[1]))[2]

    teacher_new = selected_smoke_rows("teacher", teacher_groups)
    current_student = selected_smoke_rows("student", student_groups)
    teacher_new_tokens = _path_tokens(teacher_new, tokenizer, config)
    current_tokens = _path_tokens(current_student, tokenizer, config)
    old_paths = _old_paths(args.old_sft_jsonl.expanduser().resolve())
    excluded = {str(teacher_new[0]["task_id"]), str(current_student[0]["task_id"])}
    feasibility = control_feasibility(
        old_paths,
        target_rows=len(teacher_new),
        target_tokens=teacher_new_tokens,
        tokenizer=tokenizer,
        config=config,
        excluded_tasks=excluded,
    )
    feasibility_payload = {
        "schema_version": "m6_phase10_control_feasibility_report_v1",
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "smoke_audit_content_sha256": audit["content_sha256"],
        "state_manifest_content_sha256": manifest["content_sha256"],
        "old_sft_jsonl_sha256": sha256_file(args.old_sft_jsonl.expanduser().resolve()),
        "teacher_new_path_signature": {
            "action_row_count": len(teacher_new),
            "labeled_token_count": teacher_new_tokens,
        },
        "current_student_strict_path_signature": {
            "action_row_count": len(current_student),
            "labeled_token_count": current_tokens,
        },
        "rehearsal_control": feasibility,
        "decision": "build_matched_single_update_inputs" if feasibility["control_feasible"] else "stop_before_single_update",
    }
    feasibility_payload["content_sha256"] = _self_hash(feasibility_payload)
    feasibility_output = args.feasibility_output.expanduser().resolve()
    _require(not feasibility_output.exists(), "Phase10 control feasibility report exists")
    atomic_write_json(feasibility_output, feasibility_payload)
    _require(feasibility["control_feasible"], "Phase10 probe has no exact action-row/token-matched rehearsal path")
    rehearsal_path_id, rehearsal_source = _select_exact_old_path(
        old_paths,
        target_rows=len(teacher_new),
        target_tokens=teacher_new_tokens,
        tokenizer=tokenizer,
        config=config,
        excluded_tasks=excluded,
    )
    rehearsal_new = _decorate_old(rehearsal_source, source="new")
    excluded.add(str(rehearsal_new[0]["task_id"]))
    retention_candidates = [
        (sha256_json({"seed": SEED + 1, "path_id": path_id}), path_id, rows)
        for path_id, rows in old_paths.items()
        if str(rows[0]["task_id"]) not in excluded
    ]
    _require(retention_candidates, "Phase10 probe has no old-SFT retention path")
    _, retention_path_id, retention_source = min(retention_candidates)
    old_sft = _decorate_old(retention_source, source="old_sft")

    arms = {
        "teacher": {"new": teacher_new, "old_sft": old_sft, "current_student": current_student},
        "rehearsal": {"new": rehearsal_new, "old_sft": old_sft, "current_student": current_student},
    }
    source_audit = {}
    for arm, sources in arms.items():
        source_audit[arm] = {}
        for source, rows in sources.items():
            tokenized = [tokenize_sft_row(row, tokenizer, config) for row in rows]
            source_audit[arm][source] = {
                "task_count": len({row["task_id"] for row in rows}),
                "state_count": len({row["state_id"] for row in rows}),
                "path_count": len({row["path_id"] for row in rows}),
                "action_row_count": len(rows),
                "labeled_token_count": sum(example.completion_label_tokens for example in tokenized),
                "maximum_forward_tokens": max(example.forward_tokens for example in tokenized),
            }
    _require(
        all(
            source_audit["teacher"]["new"][field]
            == source_audit["rehearsal"]["new"][field]
            for field in ("task_count", "state_count", "path_count", "action_row_count", "labeled_token_count")
        ),
        "Phase10 probe new-source compute match drift",
    )
    payload = {
        "schema_version": "m6_phase10_single_update_probe_inputs_v1",
        "complete": True,
        "development_only": True,
        "formal_checkpoint_reusable": False,
        "training_performed": False,
        "optimizer_steps": 0,
        "seed": SEED,
        "source_weights": SOURCE_WEIGHTS,
        "loss_hierarchy": "mean_source_weighted_task_state_path_action_token_v1",
        "learning_rate": config.learning_rate,
        "smoke_audit_content_sha256": audit["content_sha256"],
        "state_manifest_content_sha256": manifest["content_sha256"],
        "old_sft_jsonl": str(args.old_sft_jsonl.expanduser().resolve()),
        "old_sft_jsonl_sha256": sha256_file(args.old_sft_jsonl.expanduser().resolve()),
        "student_tokenizer": str(args.student_tokenizer.expanduser().resolve()),
        "teacher_new_path_id": teacher_new[0]["path_id"],
        "rehearsal_new_path_id": rehearsal_path_id,
        "retention_path_id": retention_path_id,
        "current_student_path_id": current_student[0]["path_id"],
        "teacher_new_labeled_tokens": teacher_new_tokens,
        "current_student_labeled_tokens": current_tokens,
        "compute_match_tolerance": {
            "new_source_task_count_delta": 0,
            "new_source_action_row_count_delta": 0,
            "new_source_labeled_token_delta": 0,
            "prompt_token_delta_is_diagnostic_only": True,
        },
        "source_audit": source_audit,
        "arms": arms,
    }
    payload["content_sha256"] = _self_hash(payload)
    atomic_write_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "content_sha256": payload["content_sha256"],
        "source_audit": source_audit,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
