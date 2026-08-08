"""Deterministic functional-file manifest for the frozen base model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..m4_long_horizon_protocol import PROJECT_ROOT, STUDY_ID
from .contracts import atomic_write_json, sha256_file, sha256_json

BASE_MODEL_MANIFEST_SCHEMA = "m4_long_horizon_base_model_manifest_v1"
BASE_MODEL_PATH = Path("/data/share/model/Qwen3.5-4B")
BASE_MODEL_MANIFEST_PATH = PROJECT_ROOT / "data" / "m4_long_horizon_base_model_manifest_v1.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _model_files(root: Path) -> tuple[Path, ...]:
    resolved = Path(root).expanduser().resolve()
    _require(resolved.is_dir(), f"base model directory is missing: {resolved}")
    files = tuple(sorted(path for path in resolved.rglob("*") if path.is_file()))
    _require(bool(files), "base model directory contains no files")
    for path in files:
        relative = path.relative_to(resolved)
        _require(not path.is_symlink(), f"base model manifest forbids symlinks: {relative}")
        _require(".." not in relative.parts, "base model file escapes model root")
    return files


def build_base_model_manifest(
    *,
    base_model: Path = BASE_MODEL_PATH,
    destination: Path = BASE_MODEL_MANIFEST_PATH,
) -> dict[str, Any]:
    """Hash every functional file and write one deterministic canonical manifest."""

    root = Path(base_model).expanduser().resolve()
    files = _model_files(root)
    entries = [
        {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    payload = {
        "schema_version": BASE_MODEL_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "base_model_path": str(root),
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
        "functional_file_set_sha256": sha256_json(entries),
        "complete": True,
    }
    atomic_write_json(destination, payload)
    return {
        "path": str(Path(destination).expanduser().resolve()),
        "sha256": sha256_file(destination),
        "payload": payload,
    }


def validate_base_model_manifest(
    path: Path = BASE_MODEL_MANIFEST_PATH,
    *,
    expected_base_model: Path = BASE_MODEL_PATH,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Fail closed on manifest drift and optionally rehash every model file."""

    manifest_path = Path(path).expanduser().resolve()
    _require(manifest_path.is_file(), "base model manifest is missing")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "base model manifest is not an object")
    _require(payload.get("schema_version") == BASE_MODEL_MANIFEST_SCHEMA, "base model manifest schema drift")
    _require(payload.get("study_id") == STUDY_ID, "base model manifest study drift")
    root = Path(expected_base_model).expanduser().resolve()
    _require(payload.get("base_model_path") == str(root), "base model manifest path drift")
    entries = payload.get("files")
    _require(isinstance(entries, list) and bool(entries), "base model manifest file list is empty")
    relative_paths = [entry.get("relative_path") for entry in entries if isinstance(entry, Mapping)]
    _require(len(relative_paths) == len(entries), "base model manifest contains a malformed entry")
    _require(relative_paths == sorted(relative_paths), "base model manifest files are not sorted")
    _require(len(set(relative_paths)) == len(relative_paths), "base model manifest contains duplicate files")
    _require(payload.get("file_count") == len(entries), "base model manifest file-count drift")
    _require(
        payload.get("total_bytes") == sum(entry.get("size_bytes", -1) for entry in entries),
        "base model manifest byte-count drift",
    )
    _require(
        payload.get("functional_file_set_sha256") == sha256_json(entries),
        "base model functional-file-set hash drift",
    )
    _require(payload.get("complete") is True, "base model manifest is incomplete")
    if verify_files:
        actual_files = _model_files(root)
        _require(
            [str(path.relative_to(root)) for path in actual_files] == relative_paths,
            "base model file roster drift",
        )
        for entry, actual in zip(entries, actual_files):
            _require(entry.get("size_bytes") == actual.stat().st_size, f"base model size drift: {actual.name}")
            _require(entry.get("sha256") == sha256_file(actual), f"base model hash drift: {actual.name}")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "payload": dict(payload),
    }
