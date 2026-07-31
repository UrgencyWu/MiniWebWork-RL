"""Build a train-only, token-evidence RSFT corpus from strict M4 rollouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

from ..m4_protocol import ONLINE_PASSES
from ..m4_training import select_rsft_rollouts
from ..rollout import RolloutRecord, RolloutStep

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ID = "m4_rsft_tokenized_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sft" / "m4_rsft_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _record_from_dict(value: dict[str, Any]) -> RolloutRecord:
    step_fields = {field.name for field in fields(RolloutStep)}
    record_fields = {field.name for field in fields(RolloutRecord)}
    steps = [
        RolloutStep(**{key: item[key] for key in step_fields if key in item})
        for item in value.get("steps", [])
    ]
    payload = {
        key: value[key]
        for key in record_fields
        if key in value and key != "steps"
    }
    payload["steps"] = steps
    record = RolloutRecord(**payload)
    record.validate()
    return record


def _load_artifact(path: Path, *, expected_seed: int | None) -> tuple[dict[str, Any], list[RolloutRecord]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RSFT rollout artifact not found: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "3.3" or not artifact.get("complete"):
        raise ValueError(f"RSFT requires a complete schema-3.3 artifact: {path}")
    if artifact.get("study_id") != "m4_rlvr_v1":
        raise ValueError(f"RSFT artifact is not an M4 rollout: {path}")
    if artifact.get("split") != "train":
        raise PermissionError("RSFT source artifacts must use the M4 train split")
    if artifact.get("K") != 4:
        raise ValueError("RSFT requires K=4 candidates per train task")
    if (artifact.get("temperature"), artifact.get("top_p"), artifact.get("top_k")) != (1.0, 1.0, 0):
        raise ValueError("RSFT requires the M4 raw sampling distribution")
    if artifact.get("generation_runtime") != {"use_cache": False, "strict_on_policy": True}:
        raise ValueError("RSFT requires strict no-cache M4 collection artifacts")
    if expected_seed is not None and artifact.get("study_seed") != expected_seed:
        raise ValueError(
            f"RSFT artifact study seed mismatch: expected {expected_seed}, got {artifact.get('study_seed')}"
        )
    records = [_record_from_dict(record) for record in artifact.get("records", [])]
    if not records:
        raise ValueError(f"RSFT artifact contains no rollout records: {path}")
    task_counts: dict[str, int] = {}
    for record in records:
        task_counts[record.task_id] = task_counts.get(record.task_id, 0) + 1
    if any(count != 4 for count in task_counts.values()):
        raise ValueError("RSFT requires exactly four candidates for every task in each pass")
    return artifact, records


def _jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def build_m4_rsft_dataset(
    artifact_paths: Iterable[Path],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Select verified Best-of-N traces from exactly two complete train passes.

    The resulting rows are already tokenized by the exact policy tokenizer used
    at collection time.  Each label covers only generated action tokens; prompt
    IDs remain context and never receive an SFT loss.
    """
    paths = [Path(path).expanduser().resolve() for path in artifact_paths]
    if len(paths) != ONLINE_PASSES:
        raise ValueError(f"RSFT requires exactly {ONLINE_PASSES} ordered train-pass artifacts")
    loaded = [_load_artifact(path, expected_seed=seed) for path in paths]
    artifacts = [item[0] for item in loaded]
    pass_indices = {artifact.get("collection_pass_index") for artifact in artifacts}
    if pass_indices != set(range(1, ONLINE_PASSES + 1)):
        raise ValueError(f"RSFT artifacts must contain ordered pass indices 1..{ONLINE_PASSES}")
    adapter_hashes = {artifact.get("adapter_sha256") for artifact in artifacts}
    if len(adapter_hashes) != 1 or None in adapter_hashes:
        raise ValueError("RSFT pass artifacts must share one non-empty initial adapter hash")
    task_sets = [{record.task_id for record in records} for _, records in loaded]
    if task_sets[0] != task_sets[1]:
        raise ValueError("RSFT passes must cover the identical train task roster")

    selections = select_rsft_rollouts(
        [record for _, records in loaded for record in records]
    )
    rows: list[dict[str, Any]] = []
    selection_audit: list[dict[str, Any]] = []
    for item in selections:
        selection = item["selection"]
        record = item["record"]
        selection_audit.append(selection)
        if record is None:
            continue
        for step in record.steps:
            if not step.prompt_token_ids or not step.generated_token_ids:
                raise ValueError(
                    f"RSFT selection {record.episode_id} has incomplete token evidence at turn {step.turn}"
                )
            rows.append(
                {
                    "sample_id": f"{record.episode_id}:turn:{step.turn}",
                    "dataset_id": DATASET_ID,
                    "split": "train",
                    "task_id": record.task_id,
                    "task_type": record.task_type,
                    "source": "m4_verified_best_of_n_rollout",
                    "episode_id": record.episode_id,
                    "rollout_index": record.rollout_index,
                    "turn_index": step.turn,
                    "input_ids": list(step.prompt_token_ids) + list(step.generated_token_ids),
                    "attention_mask": [1] * (len(step.prompt_token_ids) + len(step.generated_token_ids)),
                    "labels": [-100] * len(step.prompt_token_ids) + list(step.generated_token_ids),
                }
            )
    if not rows:
        raise ValueError("RSFT had no verifier-successful trajectories; no oracle fallback is allowed")

    output_dir = Path(output_dir).expanduser().resolve()
    rows_text = _jsonl(rows)
    manifest = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "algorithm": "rsft",
        "split": "train",
        "source_passes": ONLINE_PASSES,
        "source_artifacts": [str(path) for path in paths],
        "source_artifact_sha256": [_sha256(path) for path in paths],
        "source_adapter_sha256": next(iter(adapter_hashes)),
        "source_study_seed": artifacts[0].get("study_seed"),
        "source_collection_seeds": [artifact.get("seed") for artifact in artifacts],
        "source_pass_indices": [artifact.get("collection_pass_index") for artifact in artifacts],
        "source_task_count": len(task_sets[0]),
        "selected_task_count": sum(item["selected_episode_id"] is not None for item in selection_audit),
        "unselected_task_count": sum(item["selected_episode_id"] is None for item in selection_audit),
        "sample_count": len(rows),
        "records_sha256": hashlib.sha256(rows_text.encode("utf-8")).hexdigest(),
        "selection_audit": selection_audit,
        "selection_boundary": "only verified successful train trajectories; failures are not labels",
    }
    _atomic_write(output_dir / "train.jsonl", rows_text)
    _atomic_write(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest
