from __future__ import annotations

import json

import pytest

from miniwebwork.long_horizon_rl.model_manifest import (
    build_base_model_manifest,
    validate_base_model_manifest,
)


def test_model_manifest_binds_exact_sorted_file_roster_and_content(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    built = build_base_model_manifest(base_model=model, destination=manifest)
    checked = validate_base_model_manifest(
        manifest,
        expected_base_model=model,
        verify_files=True,
    )
    assert checked["sha256"] == built["sha256"]
    assert checked["payload"]["file_count"] == 2
    assert [entry["relative_path"] for entry in checked["payload"]["files"]] == [
        "config.json",
        "weights.safetensors",
    ]


def test_model_manifest_rejects_content_and_roster_drift(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    build_base_model_manifest(base_model=model, destination=manifest)
    (model / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size drift|hash drift"):
        validate_base_model_manifest(manifest, expected_base_model=model)
    (model / "weights.safetensors").write_bytes(b"weights")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="roster drift"):
        validate_base_model_manifest(manifest, expected_base_model=model)


def test_model_manifest_rejects_self_content_tampering_without_model_rehash(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    build_base_model_manifest(base_model=model, destination=manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "f" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="functional-file-set hash drift"):
        validate_base_model_manifest(
            manifest,
            expected_base_model=model,
            verify_files=False,
        )
