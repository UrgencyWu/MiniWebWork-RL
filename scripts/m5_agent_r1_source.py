#!/usr/bin/env python3
"""Install or verify the pinned Agent-R1 source from its locked GitHub archive."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m5_webshop_protocol import content_tree_audit, load_protocol, sha256_file  # noqa: E402

MANIFEST_NAME = ".m5_source_manifest.json"
DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024
USER_AGENT = "MiniWebWork-RL-M5-Agent-R1-source/1.0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_members(members: Iterable[tarfile.TarInfo], expected_count: int) -> tuple[list[tarfile.TarInfo], str]:
    roster = list(members)
    _require(len(roster) == expected_count, "Agent-R1 archive member-count drift")
    roots: set[str] = set()
    for member in roster:
        path = PurePosixPath(member.name)
        _require(not path.is_absolute() and ".." not in path.parts, "unsafe Agent-R1 archive path")
        _require(bool(path.parts), "empty Agent-R1 archive path")
        roots.add(path.parts[0])
        _require(member.isdir() or member.isfile(), "Agent-R1 archive contains a non-file member")
    _require(len(roots) == 1, "Agent-R1 archive has multiple roots")
    return roster, next(iter(roots))


def _extract_members(archive: tarfile.TarFile, members: list[tarfile.TarInfo], destination: Path) -> None:
    for member in members:
        target = destination.joinpath(*PurePosixPath(member.name).parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(0o755)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        _require(source is not None, f"Agent-R1 archive file cannot be read: {member.name}")
        with source, target.open("xb") as handle:
            shutil.copyfileobj(source, handle, length=DOWNLOAD_CHUNK_BYTES)
            handle.flush()
            os.fsync(handle.fileno())
        target.chmod(0o755 if member.mode & 0o111 else 0o644)


def _expected_source_contract() -> dict[str, Any]:
    return dict(load_protocol()["payload"]["upstream_sources"]["agent_r1_code"])


def _validate_manifest(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    _require(path.is_file(), "Agent-R1 archive manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "Agent-R1 archive manifest is malformed")
    expected = dict(manifest)
    observed_hash = expected.pop("content_sha256", None)
    _require(observed_hash == sha256_json(expected), "Agent-R1 archive manifest self-hash drift")
    _require(manifest.get("schema_version") == "m5_agent_r1_archive_source_v1", "Agent-R1 manifest schema drift")
    for field in ("revision", "archive_url", "archive_sha256", "archive_size", "archive_member_count"):
        _require(manifest.get(field) == contract[field], f"Agent-R1 manifest {field} drift")
    source_tree = content_tree_audit(root, "", excluded_prefixes=(MANIFEST_NAME, ".git"))
    _require(source_tree == manifest.get("source_content_tree"), "Agent-R1 extracted source-tree drift")
    _require(source_tree["sha256"] == contract["source_content_tree_sha256"], "Agent-R1 source-tree lock drift")
    _require(
        source_tree["file_count"] == contract["source_content_tree_file_count"],
        "Agent-R1 source file-count drift",
    )
    _require(source_tree["total_bytes"] == contract["source_content_tree_bytes"], "Agent-R1 source byte-count drift")
    webshop_tree = content_tree_audit(root, "recipes/webshop")
    _require(webshop_tree == manifest.get("webshop_content_tree"), "Agent-R1 extracted WebShop tree drift")
    _require(webshop_tree["sha256"] == contract["webshop_content_tree_sha256"], "Agent-R1 WebShop tree lock drift")
    return manifest


def install_archive(destination: Path) -> dict[str, Any]:
    target = Path(destination).expanduser().resolve()
    contract = _expected_source_contract()
    if (target / MANIFEST_NAME).is_file():
        return _validate_manifest(target, contract)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".m5-agent-r1-", dir=target.parent) as temporary_name:
        temporary = Path(temporary_name)
        archive_path = temporary / "source.tar.gz"
        last_error: BaseException | None = None
        for attempt in range(1, 5):
            try:
                archive_path.unlink(missing_ok=True)
                request = urllib.request.Request(contract["archive_url"], headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("xb") as handle:
                    _require(
                        int(getattr(response, "status", response.getcode())) == 200,
                        "Agent-R1 archive HTTP drift",
                    )
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                _require(archive_path.stat().st_size == contract["archive_size"], "Agent-R1 archive size drift")
                _require(sha256_file(archive_path) == contract["archive_sha256"], "Agent-R1 archive SHA256 drift")
                break
            except (OSError, urllib.error.URLError, ValueError) as exc:
                last_error = exc
                if attempt == 4:
                    raise RuntimeError("Agent-R1 archive download failed after four attempts") from last_error
                time.sleep(2**attempt)
        extraction_root = temporary / "extracted"
        extraction_root.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            members, archive_root_name = _validate_members(archive.getmembers(), contract["archive_member_count"])
            _extract_members(archive, members, extraction_root)
        extracted = extraction_root / archive_root_name
        source_tree = content_tree_audit(extracted, "", excluded_prefixes=(MANIFEST_NAME, ".git"))
        _require(source_tree["sha256"] == contract["source_content_tree_sha256"], "Agent-R1 source-tree lock drift")
        _require(
            source_tree["file_count"] == contract["source_content_tree_file_count"],
            "Agent-R1 source file-count drift",
        )
        _require(
            source_tree["total_bytes"] == contract["source_content_tree_bytes"],
            "Agent-R1 source byte-count drift",
        )
        webshop_tree = content_tree_audit(extracted, "recipes/webshop")
        _require(webshop_tree["sha256"] == contract["webshop_content_tree_sha256"], "Agent-R1 WebShop tree lock drift")
        manifest = {
            "schema_version": "m5_agent_r1_archive_source_v1",
            "revision": contract["revision"],
            "archive_url": contract["archive_url"],
            "archive_sha256": contract["archive_sha256"],
            "archive_size": contract["archive_size"],
            "archive_member_count": contract["archive_member_count"],
            "source_content_tree": source_tree,
            "webshop_content_tree": webshop_tree,
        }
        manifest["content_sha256"] = sha256_json(manifest)
        atomic_write_json(extracted / MANIFEST_NAME, manifest)
        if target.exists():
            quarantine_root = target.parent / f"{target.name}_quarantine"
            quarantine_root.mkdir(parents=True, exist_ok=True)
            quarantine = quarantine_root / f"{target.name}.partial-{time.time_ns()}"
            os.replace(target, quarantine)
            _fsync_directory(quarantine_root)
            print(f"quarantined_partial_source={quarantine}", flush=True)
        os.replace(extracted, target)
        _fsync_directory(target.parent)
    return _validate_manifest(target, contract)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install_archive(args.destination), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
