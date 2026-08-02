"""Versioned M4 v3 protocol, provenance and resumable collection gates."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from .data_generation.m4_rlvr import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TASK_ROOT,
    DEFAULT_SEED_DIR,
    DATASET_ID,
    assert_m4_split_purpose,
    validate_m4_rlvr_dataset,
)
from .m4_protocol import (
    M4RunConfig,
    M4_PROMPT_CONTRACT,
    M4_MAX_MODEL_TURNS,
    M4_MAX_NEW_TOKENS,
    ONLINE_GROUP_SIZE,
    ONLINE_PASSES,
    ONLINE_PASS_ACTION_TOKEN_CAP,
    COLLECTED_ACTION_TOKEN_CAP,
    M4_TRAIN_TASK_COUNT,
    m4_task_source_sha256,
    m4_task_roster_sha256,
    m4_adapter_directory_sha256,
)
from .model_agent import prompt_builder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_PROTOCOL_VERSION = "m4_rlvr_study_v3"
V3_STUDY_ID = "m4_rlvr_v3"
V3_STUDY_MANIFEST_SCHEMA = "m4_study_manifest_v3"
V3_STUDY_MANIFEST_PATH = PROJECT_ROOT / "data" / "m4_study_manifest_v3.json"
V3_RESUME_SCHEMA = "m4_v3_collection_resume_v1"
V3_TARGET_SUPERVISED_COMPLETION_TOKENS = 250_000
V3_MAX_SEQUENCE_LENGTH = 6_144


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_v3_study_manifest() -> dict[str, Any]:
    path = V3_STUDY_MANIFEST_PATH.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != V3_STUDY_MANIFEST_SCHEMA:
        raise ValueError("unexpected v3 study manifest schema")
    if payload.get("study_id") != V3_STUDY_ID:
        raise ValueError("unexpected v3 study id")
    if payload.get("prompt_contract") != M4_PROMPT_CONTRACT:
        raise ValueError("v3 prompt contract drift")
    adapter = payload.get("canonical_initial_adapter", {})
    if not isinstance(adapter.get("relative_path"), str):
        raise ValueError("v3 manifest lacks canonical adapter path")
    if not isinstance(adapter.get("directory_sha256"), str) or len(adapter["directory_sha256"]) != 64:
        raise ValueError("v3 manifest lacks canonical adapter hash")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "payload": payload,
    }


def assert_v3_initial_adapter(adapter: Path) -> dict[str, str]:
    study = load_v3_study_manifest()
    declared = study["payload"]["canonical_initial_adapter"]
    expected = (PROJECT_ROOT / declared["relative_path"]).resolve()
    actual = Path(adapter).expanduser().resolve()
    if actual != expected:
        raise ValueError(f"v3 initial adapter path mismatch: {actual} != {expected}")
    actual_hash = m4_adapter_directory_sha256(actual)
    if actual_hash != declared["directory_sha256"]:
        raise ValueError("v3 canonical initial adapter hash mismatch")
    return {
        "path": str(actual),
        "relative_path": declared["relative_path"],
        "sha256": actual_hash,
        "study_manifest_sha256": study["sha256"],
    }


def v3_task_order(task_root: Path, seed: int) -> tuple[str, ...]:
    public = Path(task_root).expanduser().resolve() / "train" / "train_public.jsonl"
    rows = [json.loads(line) for line in public.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row.get("task_id") for row in rows]
    if len(ids) != M4_TRAIN_TASK_COUNT or len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
        raise ValueError("v3 requires the complete unique 240-task train roster")
    random.Random(seed).shuffle(ids)
    return tuple(ids)


def build_v3_run_manifest(
    config: M4RunConfig,
    *,
    task_root: Path = DEFAULT_TASK_ROOT,
    seed_dir: Path = DEFAULT_SEED_DIR,
) -> dict[str, Any]:
    task_root = Path(task_root).expanduser().resolve()
    seed_dir = Path(seed_dir).expanduser().resolve()
    # Reuse the v2 dataset gate, but bind its result to an independent v3
    # protocol identity and v3 study manifest.
    split = config.validate(task_root=task_root)
    validation = validate_m4_rlvr_dataset(task_root, seed_dir=seed_dir)
    if not validation.get("valid"):
        raise ValueError(f"v3 dataset validation failed: {validation.get('errors')}")
    task_dir = task_root / config.split
    task_files = sorted(task_dir.glob("*.jsonl"))
    study = load_v3_study_manifest()
    return {
        "schema_version": V3_PROTOCOL_VERSION,
        "study_id": V3_STUDY_ID,
        "study_dataset_id": DATASET_ID,
        "git_sha": _git_sha(),
        "algorithm": config.algorithm.__dict__,
        "config": config.__dict__,
        "resolved_split": config.split,
        "task_dir": str(task_dir),
        "seed_dir": str(seed_dir),
        "study_manifest": {
            "path": study["path"],
            "sha256": study["sha256"],
            "payload": study["payload"],
        },
        "prompt_contract": M4_PROMPT_CONTRACT,
        "split_manifest": split,
        "hashes": {
            "dataset_manifest_sha256": _sha256(task_root / "dataset_manifest.json"),
            "split_manifest_sha256": _sha256(task_dir / "m4_split_manifest.json"),
            "seed_manifest_sha256": _sha256(seed_dir / "manifest.json"),
            "task_source_sha256": m4_task_source_sha256(task_dir, config.split),
            "prompt_system_sha256": prompt_builder.prompt_sha256(M4_PROMPT_CONTRACT),
            **{path.name: _sha256(path) for path in task_files},
        },
        "budget_contract": {
            "online_and_rsft": {
                "ordered_passes": ONLINE_PASSES,
                "per_pass_generated_action_token_cap": ONLINE_PASS_ACTION_TOKEN_CAP,
                "total_generated_action_token_cap": COLLECTED_ACTION_TOKEN_CAP,
                "rollout_group_size": ONLINE_GROUP_SIZE,
            },
            "sft_and_rsft_supervision": {
                "target_effective_completion_label_tokens": V3_TARGET_SUPERVISED_COMPLETION_TOKENS,
                "max_sequence_length": V3_MAX_SEQUENCE_LENGTH,
                "max_zero_completion_label_fraction": 0.0,
            },
        },
    }


def write_v3_manifest(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to overwrite immutable v3 manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def resume_identity(manifest: dict[str, Any], *, pass_index: int, adapter_sha256: str, task_order_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": V3_RESUME_SCHEMA,
        "study_id": V3_STUDY_ID,
        "git_sha": manifest["git_sha"],
        "prompt_contract": manifest["prompt_contract"],
        "dataset_manifest_sha256": manifest["hashes"]["dataset_manifest_sha256"],
        "task_source_sha256": manifest["hashes"]["task_source_sha256"],
        "pass_index": pass_index,
        "study_seed": manifest["config"]["seed"],
        "adapter_sha256": adapter_sha256,
        "task_order_seed": manifest["config"]["seed"],
        "task_order_sha256": task_order_sha256,
        "group_size": ONLINE_GROUP_SIZE,
        "action_token_cap": ONLINE_PASS_ACTION_TOKEN_CAP,
    }


def assert_resume_identity(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if existing.get(key) != value:
            raise ValueError(f"v3 resume identity mismatch for {key}: {existing.get(key)!r} != {value!r}")

