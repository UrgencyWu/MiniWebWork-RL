#!/usr/bin/env python3
"""Acquire and audit the pinned M5 WebShop runtime without training a model."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m5_webshop_protocol import (  # noqa: E402
    UPSTREAM_LOCK_PATH,
    audit_goals,
    load_protocol,
    load_split_exclusions,
    load_upstream_lock,
    sha256_file,
)

DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
USER_AGENT = "MiniWebWork-RL-M5-pinned-downloader/1.0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _quarantine(path: Path, reason: str, *, quarantine_root: Path | None = None) -> Path:
    parent = Path(quarantine_root).resolve() if quarantine_root is not None else path.parent
    parent.mkdir(parents=True, exist_ok=True)
    quarantine = parent / f"{path.name}.invalid-{reason}-{time.time_ns()}"
    os.replace(path, quarantine)
    _fsync_directory(parent)
    return quarantine


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _source_url(base_url: str, relative_path: str) -> str:
    quoted = "/".join(urllib.parse.quote(part, safe="") for part in Path(relative_path).parts)
    return f"{base_url.rstrip('/')}/{quoted}"


def _valid_existing(path: Path, item: Mapping[str, Any]) -> bool:
    return path.is_file() and path.stat().st_size == item["size"] and sha256_file(path) == item["sha256"]


def _observed_runtime_files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def _download_one(*, base_url: str, destination_root: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    relative = str(item["path"])
    destination = destination_root / relative
    quarantine_root = destination_root.parent / f"{destination_root.name}_quarantine"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _valid_existing(destination, item):
        return {"path": relative, "status": "reused_verified", "size": item["size"], "sha256": item["sha256"]}
    if destination.exists():
        _quarantine(destination, "final", quarantine_root=quarantine_root)

    partial = destination.with_name(f".{destination.name}.part")
    if partial.exists() and partial.stat().st_size > item["size"]:
        _quarantine(partial, "oversize", quarantine_root=quarantine_root)
    if partial.exists() and partial.stat().st_size == item["size"]:
        if sha256_file(partial) == item["sha256"]:
            os.replace(partial, destination)
            _fsync_directory(destination.parent)
            return {
                "path": relative,
                "status": "resumed_verified",
                "size": item["size"],
                "sha256": item["sha256"],
            }
        _quarantine(partial, "digest", quarantine_root=quarantine_root)
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(_source_url(base_url, relative), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = int(getattr(response, "status", response.getcode()))
        append = offset > 0 and status == 206
        if offset > 0 and status not in (200, 206):
            raise RuntimeError(f"unexpected HTTP status {status} while resuming {relative}")
        mode = "ab" if append else "wb"
        if not append:
            offset = 0
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    if partial.stat().st_size != item["size"]:
        observed_size = partial.stat().st_size
        _quarantine(partial, "size", quarantine_root=quarantine_root)
        raise ValueError(f"downloaded size drift: {relative}: {observed_size} != {item['size']}")
    observed_sha = sha256_file(partial)
    if observed_sha != item["sha256"]:
        _quarantine(partial, "digest", quarantine_root=quarantine_root)
        raise ValueError(f"downloaded SHA256 drift: {relative}")
    os.replace(partial, destination)
    _fsync_directory(destination.parent)
    return {"path": relative, "status": "downloaded_verified", "size": item["size"], "sha256": observed_sha}


def download_runtime(destination_root: Path, *, maximum_attempts: int = 4) -> dict[str, Any]:
    lock = load_upstream_lock()["payload"]
    root = Path(destination_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for item in lock["runtime_files"]:
        last_error: BaseException | None = None
        for attempt in range(1, maximum_attempts + 1):
            try:
                result = _download_one(
                    base_url=lock["source"]["download_base_url"],
                    destination_root=root,
                    item=item,
                )
                result["attempt"] = attempt
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
                break
            except (OSError, urllib.error.URLError, RuntimeError, ValueError) as exc:
                last_error = exc
                print(f"retry path={item['path']} attempt={attempt} error={type(exc).__name__}: {exc}", flush=True)
                if attempt < maximum_attempts:
                    time.sleep(min(30, 2**attempt))
        else:
            raise RuntimeError(f"download failed after {maximum_attempts} attempts: {item['path']}") from last_error
    return {
        "schema_version": "m5_webshop_download_v1",
        "destination_root": str(root),
        "file_count": len(results),
        "total_bytes": sum(item["size"] for item in results),
        "files": results,
    }


def verify_runtime(runtime_root: Path) -> dict[str, Any]:
    root = Path(runtime_root).expanduser().resolve()
    lock = load_upstream_lock()
    expected_roster = {str(item["path"]) for item in lock["payload"]["runtime_files"]}
    _require(_observed_runtime_files(root) == expected_roster, "WebShop runtime contains missing or unlocked files")
    files = []
    for expected in lock["payload"]["runtime_files"]:
        path = root / expected["path"]
        _require(path.is_file(), f"locked runtime file is missing: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        _require(size == expected["size"], f"locked runtime size drift: {expected['path']}")
        _require(digest == expected["sha256"], f"locked runtime SHA256 drift: {expected['path']}")
        files.append({"path": expected["path"], "size": size, "sha256": digest})

    goal_audit = audit_goals(root / "goals.json")
    database_uri = f"file:{root / 'products.sqlite'}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        product_count = int(connection.execute("SELECT COUNT(*) FROM products").fetchone()[0])
        meta_count_row = connection.execute("SELECT value FROM meta WHERE key = 'num_products'").fetchone()
    _require(product_count == 1181430, "WebShop SQLite product count drift")
    _require(meta_count_row is not None and int(meta_count_row[0]) == product_count, "WebShop SQLite metadata drift")

    protocol = load_protocol()
    exclusions = load_split_exclusions()
    report = {
        "schema_version": "m5_webshop_runtime_audit_v1",
        "study_id": protocol["payload"]["study_id"],
        "passed": True,
        "runtime_root": str(root),
        "protocol_sha256": protocol["sha256"],
        "upstream_lock_sha256": lock["sha256"],
        "split_exclusions_sha256": exclusions["sha256"],
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "product_count": product_count,
        "goal_audit": goal_audit,
        "files": files,
    }
    report["content_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("download", "audit"),
        help="download the exact lock, or verify every byte/count and emit a self-hashed audit",
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-attempts", type=int, default=4)
    args = parser.parse_args()
    _require(args.maximum_attempts >= 1, "maximum attempts must be positive")

    if args.command == "download":
        payload = download_runtime(args.runtime_root, maximum_attempts=args.maximum_attempts)
    else:
        payload = verify_runtime(args.runtime_root)
    if args.output:
        atomic_write_json(args.output, payload)
        print(str(args.output.expanduser().resolve()))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
