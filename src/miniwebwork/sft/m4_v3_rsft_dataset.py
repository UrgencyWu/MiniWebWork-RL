"""Build the v3 RSFT corpus from full-roster, cap-stopped rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..m4_protocol import ONLINE_GROUP_SIZE, ONLINE_PASS_ACTION_TOKEN_CAP, ONLINE_PASSES, M4_TRAIN_TASK_COUNT
from ..m4_v3_protocol import V3_STUDY_ID, V3_TARGET_SUPERVISED_COMPLETION_TOKENS, V3_MAX_SEQUENCE_LENGTH
from ..m4_training import select_rsft_rollouts
from .m4_rsft_dataset import _record_from_dict


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_artifact(path: Path, seed: int) -> tuple[dict[str, Any], list[Any]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != "3.3" or not payload.get("complete"):
        raise ValueError("v3 RSFT requires complete schema-3.3 rollout artifacts")
    if payload.get("study_id") != V3_STUDY_ID or payload.get("split") != "train":
        raise ValueError("v3 RSFT accepts only v3 train artifacts")
    if payload.get("study_seed") != seed or payload.get("K") != ONLINE_GROUP_SIZE:
        raise ValueError("v3 RSFT artifact seed or K mismatch")
    if (payload.get("temperature"), payload.get("top_p"), payload.get("top_k")) != (1.0, 1.0, 0):
        raise ValueError("v3 RSFT requires raw sampling distribution")
    if payload.get("max_collected_action_tokens") != ONLINE_PASS_ACTION_TOKEN_CAP:
        raise ValueError("v3 RSFT pass cap mismatch")
    if payload.get("max_tasks") is not None or payload.get("available_task_count") != M4_TRAIN_TASK_COUNT:
        raise ValueError("v3 RSFT must expose the complete 240-task universe without a fixed task cap")
    completed = payload.get("completed_task_count")
    groups = payload.get("groups")
    if not isinstance(completed, int) or not 0 < completed <= M4_TRAIN_TASK_COUNT:
        raise ValueError("v3 RSFT artifact has invalid completed task count")
    if not isinstance(groups, list) or len(groups) != completed:
        raise ValueError("v3 RSFT group count does not match completed tasks")
    if not payload.get("stopped_for_action_token_budget") and completed != M4_TRAIN_TASK_COUNT:
        raise ValueError("v3 RSFT pass must stop at the action-token cap or exhaust the full roster")
    action_tokens = payload.get("collected_action_tokens")
    if not isinstance(action_tokens, int) or not 0 <= action_tokens <= ONLINE_PASS_ACTION_TOKEN_CAP:
        raise ValueError("v3 RSFT action-token accounting is invalid")
    records = [_record_from_dict(row) for row in payload.get("records", [])]
    if not records:
        raise ValueError("v3 RSFT artifact has no records")
    if len(records) != completed * ONLINE_GROUP_SIZE:
        raise ValueError("v3 RSFT artifact does not contain exactly K records per group")
    task_counts: dict[str, int] = {}
    for record in records:
        task_counts[record.task_id] = task_counts.get(record.task_id, 0) + 1
    if any(value != ONLINE_GROUP_SIZE for value in task_counts.values()):
        raise ValueError("v3 RSFT requires complete K=4 groups")
    for index, group in enumerate(groups):
        group_records = records[index * ONLINE_GROUP_SIZE : (index + 1) * ONLINE_GROUP_SIZE]
        if group.get("task_id") != group_records[0].task_id:
            raise ValueError("v3 RSFT group task identity disagrees with records")
        encoded = json.dumps(
            [record.to_dict() for record in group_records],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_hash = hashlib.sha256(encoded).hexdigest()
        if group.get("group_sha256") != expected_hash:
            raise ValueError("v3 RSFT group hash mismatch")
    return payload, records


def _rows(records: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selections = select_rsft_rollouts(records)
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for item in selections:
        audit.append(item["selection"])
        record = item["record"]
        if record is None:
            continue
        for step in record.steps:
            labels = list(step.generated_token_ids)
            if not step.prompt_token_ids or not labels:
                raise ValueError("v3 RSFT cannot train on an empty prompt or completion label")
            if len(step.prompt_token_ids) + len(labels) > V3_MAX_SEQUENCE_LENGTH:
                raise ValueError("v3 RSFT selected trajectory exceeds max sequence length")
            rows.append({
                "sample_id": f"{record.episode_id}:turn:{step.turn}",
                "dataset_id": "m4_rsft_tokenized_v3",
                "split": "train",
                "task_id": record.task_id,
                "task_type": record.task_type,
                "source": "m4_v3_verified_full_roster_rollout",
                "episode_id": record.episode_id,
                "rollout_index": record.rollout_index,
                "turn_index": step.turn,
                "input_ids": list(step.prompt_token_ids) + labels,
                "attention_mask": [1] * (len(step.prompt_token_ids) + len(labels)),
                "labels": [-100] * len(step.prompt_token_ids) + labels,
            })
    rows.sort(key=lambda row: (row["task_id"], row["episode_id"], row["turn_index"]))
    return rows, audit


def _pack(rows: list[dict[str, Any]], target: int) -> tuple[list[dict[str, Any]], int]:
    if not rows:
        raise ValueError("v3 RSFT has no verified rows to pack")
    packed: list[dict[str, Any]] = []
    total = 0
    occurrence = 0
    while total < target:
        progressed = False
        for row in rows:
            count = sum(value != -100 for value in row["labels"])
            if total + count > target:
                continue
            clone = dict(row)
            clone["sample_id"] = f"{row['sample_id']}:pack:{occurrence}"
            clone["pack_occurrence"] = occurrence
            packed.append(clone)
            occurrence += 1
            total += count
            progressed = True
            if total == target:
                break
        if not progressed:
            break
    if total < target:
        shortfall = target - total
        minimum = min(sum(value != -100 for value in row["labels"]) for row in rows)
        if shortfall >= minimum:
            raise ValueError(f"v3 RSFT deterministic packing stopped {shortfall} tokens short")
    return packed, total


def build_m4_v3_rsft_dataset(artifact_paths: list[Path], output_dir: Path, *, seed: int) -> dict[str, Any]:
    if len(artifact_paths) != ONLINE_PASSES:
        raise ValueError("v3 RSFT requires exactly two ordered pass artifacts")
    loaded = [_load_artifact(path, seed) for path in artifact_paths]
    artifacts = [item[0] for item in loaded]
    if [item.get("collection_pass_index") for item in artifacts] != [1, 2]:
        raise ValueError("v3 RSFT pass artifacts must be ordered 1,2")
    for key in ("task_source_sha256", "task_order_seed", "task_order_sha256", "full_task_order_sha256", "adapter_sha256"):
        if len({item.get(key) for item in artifacts}) != 1:
            raise ValueError(f"v3 RSFT pass artifacts disagree on {key}")
    all_records = [record for _, records in loaded for record in records]
    rows, selection_audit = _rows(all_records)
    packed, realized = _pack(rows, V3_TARGET_SUPERVISED_COMPLETION_TOKENS)
    label_counts = [sum(value != -100 for value in row["labels"]) for row in rows]
    minimum_nonzero_labels = min(label_counts) if label_counts else None
    if not isinstance(minimum_nonzero_labels, int) or minimum_nonzero_labels <= 0:
        raise ValueError("v3 RSFT selected rows must all have positive completion labels")
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in packed)
    selected_tasks = sorted({row["task_id"] for row in rows})
    pass_tasks = [sorted({record.task_id for record in records}) for _, records in loaded]
    manifest = {
        "schema_version": "m4_rsft_tokenized_v3",
        "dataset_id": "m4_rsft_tokenized_v3",
        "algorithm": "rsft",
        "split": "train",
        "source_passes": ONLINE_PASSES,
        "source_artifacts": [str(Path(path).resolve()) for path in artifact_paths],
        "source_artifact_sha256": [_sha256(Path(path)) for path in artifact_paths],
        "source_adapter_sha256": artifacts[0]["adapter_sha256"],
        "source_study_seed": seed,
        "source_pass_indices": [1, 2],
        "source_task_universe_count": M4_TRAIN_TASK_COUNT,
        "source_task_count_per_pass": [len(value) for value in pass_tasks],
        "source_task_coverage_union_count": len(set(pass_tasks[0]).union(pass_tasks[1])),
        "source_task_coverage_union_fraction": len(set(pass_tasks[0]).union(pass_tasks[1])) / M4_TRAIN_TASK_COUNT,
        "source_task_overlap_count": len(set(pass_tasks[0]).intersection(pass_tasks[1])),
        "source_task_order_seed": artifacts[0]["task_order_seed"],
        "source_task_order_sha256": artifacts[0]["task_order_sha256"],
        "source_full_task_order_sha256": artifacts[0]["full_task_order_sha256"],
        "source_task_source_sha256": artifacts[0]["task_source_sha256"],
        "source_pass_action_token_cap": ONLINE_PASS_ACTION_TOKEN_CAP,
        "source_collected_action_tokens_per_pass": [item["collected_action_tokens"] for item in artifacts],
        "source_total_collected_action_tokens": sum(item["collected_action_tokens"] for item in artifacts),
        "source_group_sha256": [
            [group["group_sha256"] for group in item["groups"]]
            for item in artifacts
        ],
        "selected_task_count": len(selected_tasks),
        "verified_source_row_count": len(rows),
        "sample_count": len(packed),
        "target_supervised_completion_tokens": V3_TARGET_SUPERVISED_COMPLETION_TOKENS,
        "realized_supervised_completion_tokens": realized,
        "shortfall_supervised_completion_tokens": V3_TARGET_SUPERVISED_COMPLETION_TOKENS - realized,
        "min_nonzero_completion_label_tokens": minimum_nonzero_labels,
        "max_sequence_length": V3_MAX_SEQUENCE_LENGTH,
        "zero_completion_label_sample_fraction": 0.0,
        "records_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "selection_audit": selection_audit,
        "packing": "deterministic_sorted_verified_rows_repeated_without_partial_examples",
    }
    output_dir = Path(output_dir).expanduser().resolve()
    _atomic_write(output_dir / "train.jsonl", text)
    _atomic_write(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest
