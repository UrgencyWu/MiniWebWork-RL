#!/usr/bin/env python3
"""Audit the isolated WebShop server runtime and its public HTTP contract."""

from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m5_webshop_protocol import content_tree_audit, load_protocol, sha256_file  # noqa: E402
from miniwebwork.webshop_rl import prompt  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def environment_audit(
    *,
    upstream_root: Path,
    output: Path,
    reference_audit_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(upstream_root).expanduser().resolve()
    protocol = load_protocol()
    runtime_contract = protocol["payload"]["server_runtime"]
    source_contract = protocol["payload"]["upstream_sources"]["agent_r1_code"]
    expected_revision = source_contract["revision"]
    archive_sha256 = None
    if (root / ".git").exists():
        source_mode = "git"
        revision = _run(["git", "rev-parse", "HEAD"], cwd=root)
        _require(revision == expected_revision, "Agent-R1 server revision drift")
        _require(
            not _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root),
            "Agent-R1 tracked source is dirty",
        )
    else:
        source_mode = "github_codeload_archive"
        manifest_path = root / ".m5_source_manifest.json"
        _require(manifest_path.is_file(), "Agent-R1 source is neither a Git checkout nor a locked archive")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(isinstance(manifest, dict), "Agent-R1 archive manifest is malformed")
        manifest_without_hash = dict(manifest)
        manifest_hash = manifest_without_hash.pop("content_sha256", None)
        _require(manifest_hash == sha256_json(manifest_without_hash), "Agent-R1 archive manifest self-hash drift")
        _require(manifest.get("schema_version") == "m5_agent_r1_archive_source_v1", "Agent-R1 manifest schema drift")
        for field in ("revision", "archive_url", "archive_sha256", "archive_size", "archive_member_count"):
            _require(manifest.get(field) == source_contract[field], f"Agent-R1 archive {field} drift")
        revision = str(manifest["revision"])
        archive_sha256 = str(manifest["archive_sha256"])
    source_tree = content_tree_audit(root, "", excluded_prefixes=(".git", ".m5_source_manifest.json"))
    _require(
        source_tree["sha256"] == source_contract["source_content_tree_sha256"],
        "Agent-R1 complete source-tree drift",
    )
    _require(
        source_tree["file_count"] == source_contract["source_content_tree_file_count"],
        "Agent-R1 complete source file-count drift",
    )
    _require(
        source_tree["total_bytes"] == source_contract["source_content_tree_bytes"],
        "Agent-R1 complete source byte-count drift",
    )
    webshop_tree = content_tree_audit(root, "recipes/webshop")
    _require(
        webshop_tree["sha256"] == source_contract["webshop_content_tree_sha256"],
        "Agent-R1 WebShop content-tree drift",
    )
    if source_mode == "github_codeload_archive":
        _require(manifest.get("source_content_tree") == source_tree, "Agent-R1 manifest source-tree drift")
        _require(manifest.get("webshop_content_tree") == webshop_tree, "Agent-R1 manifest content-tree drift")
    _require(".".join(map(str, sys.version_info[:3])) == runtime_contract["python"], "M5 WebShop server Python drift")
    java_version = subprocess.run(["java", "-version"], check=True, capture_output=True, text=True).stderr.strip()
    _require(f'version "{runtime_contract["java"]}' in java_version, "M5 WebShop server Java drift")
    java_home = Path(os.environ.get("JAVA_HOME", "")).expanduser().resolve()
    jvm_path = Path(os.environ.get("JVM_PATH", "")).expanduser().resolve()
    _require((java_home / "bin" / "java").is_file(), "M5 WebShop JAVA_HOME is invalid")
    _require(jvm_path.is_file(), "M5 WebShop JVM_PATH is invalid")
    expected_packages = runtime_contract["critical_packages"]
    packages = {name: importlib.metadata.version(name) for name in expected_packages}
    _require(packages == expected_packages, "M5 WebShop server package lock drift")
    for name in runtime_contract["reward_changing_optional_packages_forbidden"]:
        _require(importlib.util.find_spec(name) is None, f"optional {name} would change frozen reward semantics")
    source_root = root / "recipes" / "webshop"
    _require(source_root.is_dir(), "Agent-R1 WebShop recipe is missing")
    report = {
        "schema_version": "m5_webshop_server_environment_audit_v1",
        "study_id": protocol["payload"]["study_id"],
        "passed": True,
        "protocol_sha256": protocol["sha256"],
        "git_sha": protocol["git_sha"],
        "upstream_root": str(root),
        "upstream_revision": revision,
        "upstream_source_mode": source_mode,
        "upstream_archive_sha256": archive_sha256,
        "upstream_source_content_tree": source_tree,
        "upstream_webshop_content_tree": webshop_tree,
        "python": sys.version,
        "java_version": java_version.splitlines()[0],
        "java_home": str(java_home),
        "jvm_path": str(jvm_path),
        "packages": packages,
        "optional_reward_dependencies_absent": runtime_contract["reward_changing_optional_packages_forbidden"],
        "pip_freeze": sorted(_run([sys.executable, "-m", "pip", "freeze"]).splitlines()),
        "requirements_sha256": sha256_file(PROJECT_ROOT / "requirements.m5-webshop-server.txt"),
    }
    report["content_sha256"] = sha256_json(report)
    if reference_audit_path is not None:
        reference_path = Path(reference_audit_path).expanduser().resolve()
        _require(reference_path.is_file(), "M5 WebShop reference environment audit is missing")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        _require(isinstance(reference, dict), "M5 WebShop reference environment audit is malformed")
        expected = dict(reference)
        observed_hash = expected.pop("content_sha256", None)
        _require(observed_hash == sha256_json(expected), "M5 WebShop reference environment self-hash drift")
        _require(reference.get("passed") is True, "M5 WebShop reference environment did not pass")
        _require(
            report["content_sha256"] == reference.get("content_sha256"),
            "M5 WebShop environment differs from the frozen setup audit",
        )
    atomic_write_json(output, report)
    return report


def health_audit(*, base_url: str, output: Path, expected_workers: int) -> dict[str, Any]:
    protocol = load_protocol()
    _require(expected_workers in {8, 16}, "unexpected WebShop service worker count")
    url = base_url.rstrip("/") + "/health"
    worker_health: dict[int, dict[str, Any]] = {}
    probe_requests = 0

    def fetch_health(_: int) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"Connection": "close"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        _require(isinstance(payload, dict), "WebShop service health response is malformed")
        return payload

    # Concurrent, connection-closing waves reliably exercise separate Gunicorn
    # workers.  The process-local ASGI wrapper must make this safe even while a
    # worker lazily initializes SQLite and Lucene state.
    with ThreadPoolExecutor(max_workers=expected_workers * 2) as executor:
        for _ in range(16):
            batch = list(executor.map(fetch_health, range(expected_workers * 2)))
            probe_requests += len(batch)
            for health in batch:
                _require(health.get("status") == "ok", "WebShop service health status failed")
                _require(health.get("dataset_mode") == "full", "WebShop service is not in full mode")
                _require(health.get("num_products") == 1181430, "WebShop service product count drift")
                _require(health.get("num_goals") == 12087, "WebShop service goal count drift")
                _require(health.get("search_top_k") == 50, "WebShop service search depth drift")
                pid = health.get("pid")
                _require(isinstance(pid, int) and pid > 0, "WebShop service health lacks a worker PID")
                worker_health[pid] = health
            if len(worker_health) >= expected_workers:
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
        "git_sha": protocol["git_sha"],
        "base_url": base_url.rstrip("/"),
        "expected_workers": expected_workers,
        "worker_pids": sorted(worker_health),
        "request_concurrency_mode": protocol["payload"]["server_runtime"]["request_concurrency"]["mode"],
        "probe_mode": "concurrent_connection_close_waves_v1",
        "probe_requests": probe_requests,
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
    environment_parser.add_argument("--reference-audit", type=Path)
    health_parser = subparsers.add_parser("health")
    health_parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    health_parser.add_argument("--output", type=Path, required=True)
    health_parser.add_argument("--expected-workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "environment":
        report = environment_audit(
            upstream_root=args.upstream_root,
            output=args.output,
            reference_audit_path=args.reference_audit,
        )
    else:
        report = health_audit(base_url=args.base_url, output=args.output, expected_workers=args.expected_workers)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
