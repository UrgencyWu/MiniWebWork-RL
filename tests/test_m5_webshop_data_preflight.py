from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.m5_webshop_data_preflight import (
    _download_one,
    _observed_runtime_files,
    _source_url,
    _valid_existing,
)


def test_source_url_quotes_each_locked_path_component():
    assert _source_url("https://example.test/root/", "lucene_index/a b") == (
        "https://example.test/root/lucene_index/a%20b"
    )


def test_valid_existing_requires_both_size_and_digest(tmp_path: Path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"locked")
    item = {"size": 6, "sha256": hashlib.sha256(b"locked").hexdigest()}
    assert _valid_existing(path, item)
    changed_size = dict(item, size=7)
    assert not _valid_existing(path, changed_size)
    changed_sha = dict(item, sha256="0" * 64)
    assert not _valid_existing(path, changed_sha)


def test_valid_existing_does_not_accept_a_directory(tmp_path: Path):
    assert not _valid_existing(tmp_path, {"size": 0, "sha256": "0" * 64})


def test_complete_verified_partial_is_promoted_without_network(tmp_path: Path):
    payload = b"resumable"
    destination = tmp_path / "nested" / "file.bin"
    destination.parent.mkdir()
    partial = destination.with_name(".file.bin.part")
    partial.write_bytes(payload)
    item = {
        "path": "nested/file.bin",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    report = _download_one(base_url="https://must-not-be-opened.invalid", destination_root=tmp_path, item=item)
    assert report["status"] == "resumed_verified"
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_runtime_roster_observes_nested_unlocked_files(tmp_path: Path):
    (tmp_path / "lucene_index").mkdir()
    (tmp_path / "goals.json").write_text("[]", encoding="utf-8")
    (tmp_path / "lucene_index" / "segments_2").write_bytes(b"unlocked")
    assert _observed_runtime_files(tmp_path) == {"goals.json", "lucene_index/segments_2"}
