"""Durable Slurm invocation identity and final resource-cost accounting."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import atomic_write_json, sha256_file, sha256_json

FORMAL_INVOCATION_SCHEMA = "m4_long_horizon_formal_invocation_v1"
SLURM_JOB_ID = re.compile(r"^[0-9]+$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _content_sha(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("invocation_content_sha256", None)
    return sha256_json(value)


def _visible_gpu_count() -> int:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return len([item for item in value.split(",") if item.strip() and item.strip() != "-1"])


def start_formal_invocation(
    *,
    root: Path,
    phase: str,
    method: str,
    seed: int,
    git_sha: str,
    expected_cpus: int,
    expected_gpus: int,
    log_stem: str,
    telemetry_path: str | None,
) -> Path:
    """Record the allocation before material formal work starts."""

    job_id = os.environ.get("SLURM_JOB_ID", "")
    _require(SLURM_JOB_ID.fullmatch(job_id) is not None, "formal execution requires a numeric Slurm job id")
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    _require(allocated_cpus == expected_cpus, "formal invocation CPU allocation drift")
    _require(_visible_gpu_count() == expected_gpus, "formal invocation visible-GPU allocation drift")
    if expected_gpus:
        _require(isinstance(telemetry_path, str) and telemetry_path, "formal GPU invocation requires telemetry")
        telemetry = Path(telemetry_path)
        _require(not telemetry.is_absolute() and ".." not in telemetry.parts, "formal telemetry path escapes repository")
    else:
        _require(telemetry_path is None, "CPU-only formal invocation cannot declare GPU telemetry")
    destination = Path(root).resolve() / "invocations" / f"job-{job_id}.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        validate_formal_invocation(existing)
        for field, expected in {
            "slurm_job_id": int(job_id), "phase": phase, "method": method,
            "seed": seed, "git_sha": git_sha, "allocated_cpus": expected_cpus,
            "allocated_gpus": expected_gpus, "gpu_telemetry_path": telemetry_path,
        }.items():
            _require(existing.get(field) == expected, f"formal invocation resume drift: {field}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FORMAL_INVOCATION_SCHEMA,
        "slurm_job_id": int(job_id),
        "phase": phase,
        "method": method,
        "seed": seed,
        "git_sha": git_sha,
        "allocated_cpus": allocated_cpus,
        "allocated_gpus": expected_gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST", ""),
        "stdout_path": f"logs/{log_stem}_{job_id}.out",
        "stderr_path": f"logs/{log_stem}_{job_id}.err",
        "gpu_telemetry_path": telemetry_path,
        "started_at_ns": time.time_ns(),
        "runner_status": "RUNNING",
        "finished_at_ns": None,
    }
    payload["invocation_content_sha256"] = _content_sha(payload)
    atomic_write_json(destination, payload)
    return destination


def finish_formal_invocation(path: Path, *, status: str) -> dict[str, Any]:
    _require(status in {"WORKLOAD_COMPLETE", "ERROR"}, "invalid formal invocation completion status")
    destination = Path(path).resolve()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    validate_formal_invocation(payload)
    _require(payload["runner_status"] == "RUNNING", "formal invocation was already closed")
    payload["runner_status"] = status
    payload["finished_at_ns"] = time.time_ns()
    payload["invocation_content_sha256"] = _content_sha(payload)
    atomic_write_json(destination, payload)
    return validate_formal_invocation(payload)


def validate_formal_invocation(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == FORMAL_INVOCATION_SCHEMA, "formal invocation schema drift")
    _require(isinstance(value.get("slurm_job_id"), int) and value["slurm_job_id"] > 0, "formal invocation job id drift")
    _require(value.get("runner_status") in {"RUNNING", "WORKLOAD_COMPLETE", "ERROR"}, "formal invocation status drift")
    _require(value.get("invocation_content_sha256") == _content_sha(value), "formal invocation self-hash drift")
    _require(value.get("allocated_cpus", 0) > 0 and value.get("allocated_gpus", -1) in {0, 1}, "formal invocation resources drift")
    telemetry_path = value.get("gpu_telemetry_path")
    if value["allocated_gpus"] == 1:
        _require(isinstance(telemetry_path, str) and telemetry_path, "formal GPU invocation telemetry drift")
        telemetry = Path(telemetry_path)
        _require(not telemetry.is_absolute() and ".." not in telemetry.parts, "formal invocation telemetry path drift")
    else:
        _require(telemetry_path is None, "formal CPU invocation unexpectedly declares GPU telemetry")
    return value


def load_formal_invocations(roots: Iterable[Path]) -> list[dict[str, Any]]:
    invocations = []
    for root in roots:
        root = Path(root).resolve()
        paths = sorted((root / "invocations").glob("job-*.json"))
        _require(bool(paths), f"formal invocation evidence is missing: {root}")
        for path in paths:
            payload = validate_formal_invocation(json.loads(path.read_text(encoding="utf-8")))
            payload["invocation_path"] = str(path)
            payload["invocation_file_sha256"] = sha256_file(path)
            payload["artifact_root"] = str(root)
            invocations.append(payload)
    job_ids = [item["slurm_job_id"] for item in invocations]
    _require(len(job_ids) == len(set(job_ids)), "formal Slurm job id is reused across model roots")
    return invocations


def _allocated_gpu_count(alloc_tres: str) -> int:
    match = re.search(r"(?:^|,)gres/gpu=([0-9]+)(?:,|$)", alloc_tres)
    return int(match.group(1)) if match else 0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "cannot summarize empty GPU telemetry")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarize_gpu_telemetry(path: Path) -> dict[str, Any]:
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            _require(len(row) == 8, f"formal GPU telemetry row width drift at line {line_number}")
            try:
                values = {
                    "uuid": row[2].strip(),
                    "gpu_utilization": float(row[3]),
                    "memory_utilization": float(row[4]),
                    "memory_used_mib": float(row[5]),
                    "memory_total_mib": float(row[6]),
                    "power_watts": float(row[7]),
                }
            except ValueError as exc:
                raise ValueError(f"formal GPU telemetry numeric drift at line {line_number}") from exc
            _require(values["uuid"].startswith("GPU-"), "formal GPU telemetry UUID drift")
            _require(0 <= values["gpu_utilization"] <= 100, "formal GPU utilization is out of range")
            _require(0 <= values["memory_used_mib"] <= values["memory_total_mib"], "formal GPU memory is out of range")
            rows.append(values)
    _require(bool(rows), f"formal GPU telemetry is empty: {path}")
    uuids = {row["uuid"] for row in rows}
    totals = {row["memory_total_mib"] for row in rows}
    _require(len(uuids) == 1 and len(totals) == 1, "formal GPU telemetry spans multiple devices")
    gpu = [row["gpu_utilization"] for row in rows]
    memory = [row["memory_used_mib"] for row in rows]
    power = [row["power_watts"] for row in rows]
    total = next(iter(totals))
    return {
        "sample_count": len(rows),
        "gpu_uuid": next(iter(uuids)),
        "gpu_utilization_fraction_mean": sum(gpu) / len(gpu) / 100,
        "gpu_utilization_fraction_p50": _percentile(gpu, 0.5) / 100,
        "gpu_utilization_fraction_p95": _percentile(gpu, 0.95) / 100,
        "peak_memory_mib": max(memory),
        "device_memory_mib": total,
        "minimum_vram_headroom_fraction": 1 - max(memory) / total,
        "mean_power_watts": sum(power) / len(power),
        "maximum_power_watts": max(power),
    }


def collect_slurm_accounting(
    roots: Iterable[Path],
    *,
    repo_root: Path,
    sacct_path: Path = Path("/opt/slurm/slurm.25.05/bin/sacct"),
) -> dict[str, Any]:
    """Join durable invocations to final Slurm allocation records and logs."""

    artifact_roots = tuple(Path(root).resolve() for root in roots)
    invocations = load_formal_invocations(artifact_roots)
    records = []
    for invocation in invocations:
        job_id = invocation["slurm_job_id"]
        result = subprocess.run(
            [str(sacct_path), "-X", "-j", str(job_id), "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,AllocTRES", "-n", "-P"],
            cwd=Path(repo_root), check=True, text=True, capture_output=True,
        )
        rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
        row = next((item for item in rows if item[0] == str(job_id)), None)
        _require(row is not None and len(row) >= 7, f"missing formal sacct record: {job_id}")
        elapsed_seconds = int(row[3])
        allocated_cpus = int(row[4])
        allocated_gpus = _allocated_gpu_count(row[6])
        _require(allocated_cpus == invocation["allocated_cpus"], f"formal sacct CPU drift: {job_id}")
        _require(allocated_gpus == invocation["allocated_gpus"], f"formal sacct GPU drift: {job_id}")
        _require(not row[1].startswith(("PENDING", "RUNNING")), f"formal job accounting is not final: {job_id}")
        logs = {}
        for label in ("stdout", "stderr"):
            path = Path(repo_root) / invocation[f"{label}_path"]
            _require(path.is_file(), f"formal {label} log is missing: {job_id}")
            logs[f"{label}_path"] = str(path.relative_to(repo_root))
            logs[f"{label}_sha256"] = sha256_file(path)
        telemetry = None
        if allocated_gpus:
            telemetry_path = Path(repo_root) / invocation["gpu_telemetry_path"]
            _require(telemetry_path.is_file(), f"formal GPU telemetry is missing: {job_id}")
            telemetry = {
                "path": str(telemetry_path.relative_to(repo_root)),
                "sha256": sha256_file(telemetry_path),
                **_summarize_gpu_telemetry(telemetry_path),
            }
        records.append({
            "job_id": job_id,
            "artifact_root": invocation["artifact_root"],
            "phase": invocation["phase"],
            "method": invocation["method"],
            "seed": invocation["seed"],
            "state": row[1],
            "exit_code": row[2],
            "elapsed_seconds": elapsed_seconds,
            "allocated_cpus": allocated_cpus,
            "allocated_gpus": allocated_gpus,
            "requested_memory": row[5],
            "allocated_tres": row[6],
            "gpu_hours": elapsed_seconds * allocated_gpus / 3600,
            "cpu_hours": elapsed_seconds * allocated_cpus / 3600,
            "runner_status": invocation["runner_status"],
            "invocation_file_sha256": invocation["invocation_file_sha256"],
            "gpu_telemetry": telemetry,
            **logs,
        })
    for root in artifact_roots:
        successful = [
            record for record in records
            if record["artifact_root"] == str(root)
            and record["state"].startswith("COMPLETED")
            and record["exit_code"] == "0:0"
            and record["runner_status"] == "WORKLOAD_COMPLETE"
        ]
        _require(bool(successful), f"formal artifact root lacks a completed Slurm invocation: {root}")
    return {
        "job_count": len(records),
        "gpu_hours": sum(item["gpu_hours"] for item in records),
        "cpu_hours": sum(item["cpu_hours"] for item in records),
        "jobs": records,
    }
