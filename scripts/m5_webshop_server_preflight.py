#!/usr/bin/env python3
"""Audit the isolated WebShop server runtime and its public HTTP contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m5_webshop_protocol import load_protocol, sha256_file  # noqa: E402
from miniwebwork.webshop_rl import prompt  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _tracked_tree_sha256(root: Path, prefix: str) -> str:
    """Hash the pinned Git tree, excluding runtime bytecode and caches."""

    listing = _run(["git", "ls-tree", "-r", "--full-tree", "HEAD", prefix], cwd=root)
    _require(bool(listing), f"tracked upstream tree is empty: {prefix}")
    return hashlib.sha256((listing + "\n").encode("utf-8")).hexdigest()


def environment_audit(*, upstream_root: Path, output: Path) -> dict[str, Any]:
    root = Path(upstream_root).expanduser().resolve()
    protocol = load_protocol()
    expected_revision = protocol["payload"]["upstream_sources"]["agent_r1_code"]["revision"]
    _require((root / ".git").exists(), "Agent-R1 server source is not a Git checkout")
    revision = _run(["git", "rev-parse", "HEAD"], cwd=root)
    _require(revision == expected_revision, "Agent-R1 server revision drift")
    _require(not _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root), "Agent-R1 tracked source is dirty")
    _require(sys.version_info[:2] == (3, 12), "M5 WebShop server requires Python 3.12")
    java_version = subprocess.run(["java", "-version"], check=True, capture_output=True, text=True).stderr.strip()
    _require('version "21' in java_version, "M5 WebShop server requires Java 21")
    java_home = Path(os.environ.get("JAVA_HOME", "")).expanduser().resolve()
    jvm_path = Path(os.environ.get("JVM_PATH", "")).expanduser().resolve()
    _require((java_home / "bin" / "java").is_file(), "M5 WebShop JAVA_HOME is invalid")
    _require(jvm_path.is_file(), "M5 WebShop JVM_PATH is invalid")
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "pandas",
            "pyarrow",
            "fastapi",
            "gunicorn",
            "uvicorn",
            "pyserini",
            "pyjnius",
            "httpx",
            "rank-bm25",
        )
    }
    expected_packages = {
        "pandas": "3.0.3",
        "pyarrow": "25.0.0",
        "fastapi": "0.139.2",
        "gunicorn": "26.0.0",
        "uvicorn": "0.51.0",
        "pyserini": "2.3.0",
        "pyjnius": "1.7.0",
        "httpx": "0.28.1",
        "rank-bm25": "0.2.2",
    }
    _require(packages == expected_packages, "M5 WebShop server package lock drift")
    _require(importlib.util.find_spec("thefuzz") is None, "optional thefuzz would change the frozen reward semantics")
    _require(importlib.util.find_spec("spacy") is None, "optional spaCy would change the frozen reward semantics")
    source_root = root / "recipes" / "webshop"
    _require(source_root.is_dir(), "Agent-R1 WebShop recipe is missing")
    report = {
        "schema_version": "m5_webshop_server_environment_audit_v1",
        "study_id": protocol["payload"]["study_id"],
        "passed": True,
        "protocol_sha256": protocol["sha256"],
        "upstream_root": str(root),
        "upstream_revision": revision,
        "upstream_webshop_tree_sha256": _tracked_tree_sha256(root, "recipes/webshop"),
        "python": sys.version,
        "java_version": java_version.splitlines()[0],
        "java_home": str(java_home),
        "jvm_path": str(jvm_path),
        "packages": packages,
        "optional_reward_dependencies_absent": ["spacy", "thefuzz"],
        "pip_freeze": sorted(_run([sys.executable, "-m", "pip", "freeze"]).splitlines()),
        "requirements_sha256": sha256_file(PROJECT_ROOT / "requirements.m5-webshop-server.txt"),
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(output, report)
    return report


def health_audit(*, base_url: str, output: Path, expected_workers: int) -> dict[str, Any]:
    protocol = load_protocol()
    _require(expected_workers in {2, 4, 8}, "unexpected WebShop service worker count")
    url = base_url.rstrip("/") + "/health"
    worker_health: dict[int, dict[str, Any]] = {}
    for _ in range(expected_workers * 16):
        with urllib.request.urlopen(url, timeout=120) as response:
            health = json.loads(response.read().decode("utf-8"))
        _require(health.get("status") == "ok", "WebShop service health status failed")
        _require(health.get("dataset_mode") == "full", "WebShop service is not in full mode")
        _require(health.get("num_products") == 1181430, "WebShop service product count drift")
        _require(health.get("num_goals") == 12087, "WebShop service goal count drift")
        _require(health.get("search_top_k") == 50, "WebShop service search depth drift")
        pid = health.get("pid")
        _require(isinstance(pid, int) and pid > 0, "WebShop service health lacks a worker PID")
        worker_health[pid] = health
        if len(worker_health) == expected_workers:
            break
    _require(len(worker_health) == expected_workers, "not every WebShop worker passed warm health initialization")
    environment = WebShopHTTPEnvironment(base_url=base_url, split="train", timeout_seconds=120)
    try:
        observation = environment.reset("webshop_goal_01000")
        rendered = json.dumps(prompt.build_messages(observation), ensure_ascii=False)
        _require("target_asin" not in rendered and '"asin"' not in rendered, "WebShop prompt leaked target metadata")
        _require(observation.available_actions == ("search[<your query>]",), "WebShop reset action contract drift")
    finally:
        environment.close()
    report = {
        "schema_version": "m5_webshop_service_health_audit_v1",
        "study_id": protocol["payload"]["study_id"],
        "passed": True,
        "protocol_sha256": protocol["sha256"],
        "base_url": base_url.rstrip("/"),
        "expected_workers": expected_workers,
        "worker_pids": sorted(worker_health),
        "health": worker_health[sorted(worker_health)[0]],
        "adapter_smoke_task_id": "webshop_goal_01000",
        "target_metadata_absent_from_prompt": True,
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    environment_parser = subparsers.add_parser("environment")
    environment_parser.add_argument("--upstream-root", type=Path, required=True)
    environment_parser.add_argument("--output", type=Path, required=True)
    health_parser = subparsers.add_parser("health")
    health_parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    health_parser.add_argument("--output", type=Path, required=True)
    health_parser.add_argument("--expected-workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "environment":
        report = environment_audit(upstream_root=args.upstream_root, output=args.output)
    else:
        report = health_audit(base_url=args.base_url, output=args.output, expected_workers=args.expected_workers)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
