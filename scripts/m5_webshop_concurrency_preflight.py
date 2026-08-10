#!/usr/bin/env python3
"""Stress the real WebShop reset/search path at frozen lane counts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m5_webshop_protocol import (  # noqa: E402
    deterministic_candidate_order,
    eligible_goal_indices,
    load_protocol,
    sha256_file,
    task_id_for_goal_index,
)
from miniwebwork.webshop_rl.actions import WebShopCommand  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _percentile(values: list[float], quantile: float) -> float:
    _require(bool(values), "latency sample is empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def run_stress(
    *,
    base_url: str,
    workers: int,
    lanes: int,
    episodes_per_lane: int,
    health_audit_path: Path,
    output: Path,
) -> dict[str, Any]:
    protocol = load_protocol()
    service = protocol["payload"]["slurm"]["shared_environment_service"]
    online = protocol["payload"]["online"]
    _require(workers in service["worker_candidates"], "stress worker count is outside the frozen candidates")
    _require(lanes in online["candidate_parallel_lanes"], "stress lane count is outside the frozen candidates")
    _require(episodes_per_lane >= 1, "stress episodes per lane must be positive")

    health_path = Path(health_audit_path).expanduser().resolve()
    health = json.loads(health_path.read_text(encoding="utf-8"))
    health_without_hash = dict(health)
    health_hash = health_without_hash.pop("content_sha256", None)
    _require(health_hash == sha256_json(health_without_hash), "stress health audit self-hash drift")
    _require(health.get("passed") is True, "stress requires a passing health audit")
    _require(health.get("git_sha") == protocol["git_sha"], "stress health Git lineage drift")
    _require(health.get("protocol_sha256") == protocol["sha256"], "stress health protocol drift")
    _require(health.get("expected_workers") == workers, "stress health worker count drift")
    _require(
        health.get("request_concurrency_mode") == "process_serialized_asgi_v1",
        "stress health request-concurrency mode drift",
    )
    normalized_url = base_url.rstrip("/")
    _require(health.get("base_url") == normalized_url, "stress health URL drift")

    required_tasks = lanes * episodes_per_lane
    order = deterministic_candidate_order(
        candidates=eligible_goal_indices("train"),
        seed=20260810,
        namespace=f"m5-service-stress-{lanes}-{episodes_per_lane}",
    )[:required_tasks]
    _require(len(order) == required_tasks, "insufficient frozen train tasks for stress")
    assignments = [order[lane_index::lanes] for lane_index in range(lanes)]

    def run_lane(lane_index: int) -> dict[str, Any]:
        latencies: list[float] = []
        failures: list[dict[str, Any]] = []
        environment = WebShopHTTPEnvironment(
            base_url=normalized_url,
            split="train",
            timeout_seconds=120,
        )
        try:
            for goal_index in assignments[lane_index]:
                task_id = task_id_for_goal_index(goal_index)
                try:
                    started = time.perf_counter()
                    observation = environment.reset(task_id)
                    latencies.append(time.perf_counter() - started)
                    _require(
                        "search[<your query>]" in observation.available_actions,
                        "stress reset lacks public search action",
                    )
                    started = time.perf_counter()
                    result = environment.step(WebShopCommand("search[product]"))
                    latencies.append(time.perf_counter() - started)
                    _require(result.info.get("action_result", {}).get("success") is True, "stress search failed")
                    _require(not result.terminated, "stress search terminated unexpectedly")
                except Exception as exc:  # diagnostic artifact must survive any HTTP/runtime failure
                    status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                    failures.append(
                        {
                            "lane_index": lane_index,
                            "task_id": task_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                            "http_status_code": status_code,
                        }
                    )
        finally:
            environment.close()
        return {"latencies": latencies, "failures": failures}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=lanes) as executor:
        lane_reports = list(executor.map(run_lane, range(lanes)))
    wall_seconds = time.perf_counter() - started
    latencies = [value for lane in lane_reports for value in lane["latencies"]]
    failures = [failure for lane in lane_reports for failure in lane["failures"]]
    http_5xx_count = sum(
        isinstance(failure.get("http_status_code"), int) and 500 <= failure["http_status_code"] <= 599
        for failure in failures
    )
    planned_requests = required_tasks * 2
    completed_requests = len(latencies)
    passed = not failures and completed_requests == planned_requests
    report = {
        "schema_version": "m5_webshop_concurrency_preflight_v1",
        "study_id": protocol["payload"]["study_id"],
        "passed": passed,
        "formal_training": False,
        "protocol_sha256": protocol["sha256"],
        "git_sha": protocol["git_sha"],
        "base_url": normalized_url,
        "workers": workers,
        "lanes": lanes,
        "episodes_per_lane": episodes_per_lane,
        "task_order_sha256": sha256_json(order),
        "health_audit_sha256": sha256_file(health_path),
        "planned_requests": planned_requests,
        "completed_requests": completed_requests,
        "failure_count": len(failures),
        "http_5xx_count": http_5xx_count,
        "http_5xx_fraction": http_5xx_count / planned_requests,
        "failures": failures[:100],
        "wall_seconds": wall_seconds,
        "requests_per_second": completed_requests / wall_seconds,
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50) if latencies else None,
            "p95": _percentile(latencies, 0.95) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--episodes-per-lane", type=int, default=4)
    parser.add_argument("--health-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_stress(
        base_url=args.base_url,
        workers=args.workers,
        lanes=args.lanes,
        episodes_per_lane=args.episodes_per_lane,
        health_audit_path=args.health_audit,
        output=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
