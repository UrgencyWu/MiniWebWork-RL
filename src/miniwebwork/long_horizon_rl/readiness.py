"""Fail-closed final readiness audit for focused long-horizon Agent RL."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..m4_long_horizon_protocol import (
    PROJECT_ROOT,
    PROMPT_CONTRACT,
    PROMPT_PATH,
    STUDY_ID,
    load_study_manifest,
)
from .adapter_view import validate_vllm_adapter_view
from .contracts import (
    RunIdentity,
    atomic_write_json,
    directory_sha256,
    sha256_file,
    sha256_json,
    validate_committed_group,
)
from .iteration import IterationLedger, IterationStore
from .journal import AppendOnlyAttemptJournal
from .model_manifest import validate_base_model_manifest
from .runtime_contract import load_online_runtime_contract

READINESS_SCHEMA = "m4_long_horizon_formal_training_readiness_v1"
READINESS_ROOT = PROJECT_ROOT / "outputs" / STUDY_ID / "readiness"
SFT_CORPUS_ROOT = PROJECT_ROOT / "data" / "sft" / "m4_long_horizon_verified_v2"
SFT_SELECTION_PATH = PROJECT_ROOT / "data" / "m4_long_horizon_sft_preflight_selection_v1.json"


@dataclass(frozen=True)
class JobEvidence:
    job_id: int
    purpose: str
    expected_state: str
    expected_exit_code: str
    stdout_path: str
    stderr_path: str


JOB_EVIDENCE = (
    JobEvidence(1261, "verified_sft_corpus", "COMPLETED", "0:0", "logs/m4_lh_sft_corpus_1261.out", "logs/m4_lh_sft_corpus_1261.err"),
    JobEvidence(1264, "sft_gpu_preflight", "COMPLETED", "0:0", "logs/m4_lh_sft_preflight_1264.out", "logs/m4_lh_sft_preflight_1264.err"),
    JobEvidence(1295, "selected_32_lane_grpo_soak", "COMPLETED", "0:0", "/home/wushaohua/logs/m4_lh_online_preflight_1295.out", "/home/wushaohua/logs/m4_lh_online_preflight_1295.err"),
    JobEvidence(1296, "selected_32_lane_stepaware_collection_diagnostic", "FAILED", "1:0", "/home/wushaohua/logs/m4_lh_online_preflight_1296.out", "/home/wushaohua/logs/m4_lh_online_preflight_1296.err"),
    JobEvidence(1302, "selected_32_lane_stepaware_e2e", "COMPLETED", "0:0", "logs/m4_lh_online_preflight_1302.out", "logs/m4_lh_online_preflight_1302.err"),
    JobEvidence(1311, "64_lane_saturation_challenger", "COMPLETED", "0:0", "logs/m4_lh_online_preflight_1311.out", "logs/m4_lh_online_preflight_1311.err"),
    JobEvidence(1314, "runtime_v8_clean_cpu_regression", "COMPLETED", "0:0", "/home/wushaohua/logs/m4_v8_clean_1314.out", "/home/wushaohua/logs/m4_v8_clean_1314.err"),
    JobEvidence(1315, "runtime_v8_selected_e2e", "COMPLETED", "0:0", "logs/m4_lh_online_preflight_1315.out", "logs/m4_lh_online_preflight_1315.err"),
    JobEvidence(1316, "runtime_v8_selected_soak", "COMPLETED", "0:0", "logs/m4_lh_online_preflight_1316.out", "logs/m4_lh_online_preflight_1316.err"),
    JobEvidence(1317, "real_collection_interruption", "CANCELLED", "0:0", "logs/m4_lh_online_preflight_1317.out", "logs/m4_lh_online_preflight_1317.err"),
    JobEvidence(1318, "real_learner_interruption", "CANCELLED", "0:0", "logs/m4_lh_online_preflight_1318.out", "logs/m4_lh_online_preflight_1318.err"),
    JobEvidence(1319, "real_resume_commit_wake", "COMPLETED", "0:0", "logs/m4_lh_online_preflight_1319.out", "logs/m4_lh_online_preflight_1319.err"),
)

SOAK_RUNS = (
    "online_runtime_v5_batch32_grpo_s20260801_w32_t8_mb8_a1",
    "online_runtime_v5_batch32_stepaware_s20260801_w32_t8_mb8_a1",
    "online_runtime_v6_stepaware_s20260801_w32_t8_mb8_a1",
    "online_runtime_v8_selected32_stepaware_s20260801_w32_t8_mb8_a1",
    "online_runtime_v8_selected32_soak_stepaware_s20260801_w32_t16_a1",
)
SELECTED_RUN = "online_runtime_v8_selected32_stepaware_s20260801_w32_t8_mb8_a1"
SOAK_RUN = "online_runtime_v8_selected32_soak_stepaware_s20260801_w32_t16_a1"
RECOVERY_RUN = "online_runtime_v8_recovery_stepaware_s20260801_w32_t8_mb8_a1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _run(command: Sequence[str], *, cwd: Path) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=True, text=True, stdout=subprocess.PIPE
    ).stdout


def _relative_path(value: str | Path, root: Path) -> str:
    return str(Path(value).resolve().relative_to(root.resolve()))


def tracked_file_manifest(repo_root: Path) -> dict[str, Any]:
    raw = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    paths = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    files = []
    for relative in paths:
        path = repo_root / relative
        _require(path.is_file(), f"tracked file is missing: {relative}")
        files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "file_count": len(files),
        "files": files,
        "tracked_file_set_sha256": sha256_json(files),
    }


def parse_sacct_record(text: str, job: JobEvidence) -> dict[str, Any]:
    records = [line.split("|") for line in text.splitlines() if line.strip()]
    record = next((parts for parts in records if parts[0] == str(job.job_id)), None)
    _require(record is not None and len(record) >= 6, f"missing sacct record for Job {job.job_id}")
    state = record[1]
    exit_code = record[2]
    _require(state.startswith(job.expected_state), f"Job {job.job_id} state drift: {state}")
    _require(exit_code == job.expected_exit_code, f"Job {job.job_id} exit drift: {exit_code}")
    return {
        "job_id": job.job_id,
        "purpose": job.purpose,
        "state": state,
        "exit_code": exit_code,
        "elapsed": record[3],
        "allocated_cpus": int(record[4]),
        "requested_memory": record[5],
    }


def collect_job_evidence(repo_root: Path, sacct_path: Path) -> list[dict[str, Any]]:
    records = []
    for job in JOB_EVIDENCE:
        text = _run(
            [str(sacct_path), "-j", str(job.job_id), "--format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,ReqMem", "-n", "-P"],
            cwd=repo_root,
        )
        record = parse_sacct_record(text, job)
        for label, configured in (("stdout", job.stdout_path), ("stderr", job.stderr_path)):
            path = Path(configured)
            if not path.is_absolute():
                path = repo_root / path
            _require(path.is_file(), f"Job {job.job_id} {label} log missing")
            record[f"{label}_path"] = configured
            record[f"{label}_sha256"] = sha256_file(path)
        records.append(record)
    return records


def collect_one_job_evidence(
    repo_root: Path,
    sacct_path: Path,
    job: JobEvidence,
) -> dict[str, Any]:
    text = _run(
        [str(sacct_path), "-j", str(job.job_id), "--format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,ReqMem", "-n", "-P"],
        cwd=repo_root,
    )
    record = parse_sacct_record(text, job)
    for label, configured in (("stdout", job.stdout_path), ("stderr", job.stderr_path)):
        path = Path(configured)
        if not path.is_absolute():
            path = repo_root / path
        _require(path.is_file(), f"Job {job.job_id} {label} log missing")
        record[f"{label}_path"] = configured
        record[f"{label}_sha256"] = sha256_file(path)
    return record


def _event_map(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    events = [_json(path) for path in sorted((root / "phase_events").glob("*.json"))]
    for index, event in enumerate(events):
        _require(event.get("sequence") == index, "phase sequence drift")
        expected = sha256_json({key: value for key, value in event.items() if key != "event_sha256"})
        _require(event.get("event_sha256") == expected, "phase event hash drift")
    latest = {event["event_type"]: event for event in events}
    return events, latest


def soak_evidence(preflight_root: Path) -> dict[str, Any]:
    runs = []
    for name in SOAK_RUNS:
        root = preflight_root / name
        _, events = _event_map(root)
        start = events["collection_started"]["timestamp_ns"]
        frozen = events["collection_frozen"]
        phase_seconds = (frozen["timestamp_ns"] - start) / 1_000_000_000
        runs.append(
            {
                "run": name,
                "phase_seconds": phase_seconds,
                "measured_generation_seconds": frozen["payload"]["elapsed_seconds"],
            }
        )
    total = sum(run["phase_seconds"] for run in runs)
    _require(total >= 1800.0, "selected 32-lane collection phase soak is below 30 minutes")
    return {
        "runs": runs,
        "total_phase_seconds": total,
        "total_phase_minutes": total / 60,
        "total_measured_generation_seconds": sum(run["measured_generation_seconds"] for run in runs),
        "gate_minimum_phase_seconds": 1800.0,
        "passed": True,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot take percentile of empty samples")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def telemetry_window(root: Path, start_event: str, end_event: str, job_id: int) -> dict[str, Any]:
    _, events = _event_map(root)
    start = events[start_event]["timestamp_ns"] / 1_000_000_000
    end = events[end_event]["timestamp_ns"] / 1_000_000_000
    rows = []
    with (root / f"gpu_telemetry_job_{job_id}.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            if start <= timestamp <= end:
                rows.append(
                    {
                        "gpu": float(row[3]),
                        "memory": float(row[5]),
                        "total": float(row[6]),
                        "power": float(row[7]),
                    }
                )
    _require(bool(rows), f"no telemetry samples for {start_event}..{end_event}")
    gpu = [row["gpu"] for row in rows]
    peak_memory = max(row["memory"] for row in rows)
    return {
        "samples": len(rows),
        "elapsed_seconds": end - start,
        "gpu_utilization_p50": _percentile(gpu, 0.5) / 100,
        "gpu_utilization_mean": sum(gpu) / len(gpu) / 100,
        "gpu_utilization_p95": _percentile(gpu, 0.95) / 100,
        "peak_memory_mib": peak_memory,
        "minimum_vram_headroom_fraction": 1 - peak_memory / rows[0]["total"],
        "mean_power_watts": sum(row["power"] for row in rows) / len(rows),
    }


def selected_e2e_evidence(preflight_root: Path) -> dict[str, Any]:
    root = preflight_root / SELECTED_RUN
    report = _json(root / "preflight_report.json")
    _require(report.get("result") == "PASS" and report.get("complete") is True, "selected v8 E2E failed")
    _require(report.get("formal_training") is False, "selected E2E leaked into formal mode")
    performance = report["collection_performance"]
    generation = telemetry_window(root, "collection_started", "collection_frozen", 1315)
    learner = telemetry_window(root, "learner_started", "learner_committed", 1315)
    post_wake = telemetry_window(root, "vllm_wake_complete", "engine_shutdown", 1315)
    _require(performance["trajectories_per_hour"] >= 114.0, "rollout throughput gate failed")
    _require(generation["gpu_utilization_p95"] >= 0.5, "generation P95 gate failed")
    _require(learner["gpu_utilization_p50"] >= 0.8, "learner P50 gate failed")
    _require(min(generation["minimum_vram_headroom_fraction"], learner["minimum_vram_headroom_fraction"], post_wake["minimum_vram_headroom_fraction"]) >= 0.15, "VRAM headroom gate failed")
    parity = report["learner"]["initial_replay_parity"]
    _require(parity["passed"] is True, "selected E2E parity failed")
    _require(report["same_gpu_phase_switch"]["passed"] is True, "same-GPU wake gate failed")
    return {
        "result": report["result"],
        "report_sha256": sha256_file(root / "preflight_report.json"),
        "collection_performance": performance,
        "generation_telemetry": generation,
        "learner_telemetry": learner,
        "post_wake_telemetry": post_wake,
        "parity": parity,
        "learner": {key: report["learner"][key] for key in (
            "zero_advantage_group_count", "optimizer_updates", "effective_optimizer_action_tokens",
            "effective_optimizer_action_token_fraction", "parameter_change_norm",
            "output_adapter_sha256", "output_rollout_adapter_sha256", "output_optimizer_sha256",
        )},
        "same_gpu_phase_switch": report["same_gpu_phase_switch"],
        "passed": True,
    }


def recovery_evidence(preflight_root: Path) -> dict[str, Any]:
    root = preflight_root / RECOVERY_RUN
    report = _json(root / "preflight_report.json")
    _require(report.get("result") == "PASS" and report.get("complete") is True, "recovery final report failed")
    _require(report["collection"]["resumed_frozen_collection"] is True, "frozen collection was not resumed")
    identity = RunIdentity.from_mapping(_json(root / "collections" / "iteration-0000" / "run_identity.json"))
    attempt_journal = AppendOnlyAttemptJournal(
        root / "collections" / "iteration-0000" / "attempt_journal.jsonl",
        identity,
    )
    journal_events = attempt_journal.events
    _require(len(journal_events) == 1150, "recovery journal event count drift")
    groups = [
        validate_committed_group(_json(path), identity)
        for path in sorted((root / "collections" / "iteration-0000" / "groups").glob("*.json"))
    ]
    _require(len(groups) == 8, "recovery group count drift")
    collection = report["collection"]["collection_manifest"]
    invalidated_cost = collection["all_generated_action_tokens"] - collection["committed_group_action_tokens"]
    _require(invalidated_cost == 4051, "interrupted collection token cost drift")
    invalidated_attempts = list((root / "collections" / "iteration-0000" / "invalidated_attempts").glob("*/*"))
    _require(len(invalidated_attempts) == 8, "interrupted attempt archive count drift")
    invalidated_stages = sorted(path.name for path in (root / "state" / "invalidated_stages").iterdir())
    _require(invalidated_stages == ["iteration-0000-attempt-0000"], "interrupted learner stage archive drift")
    phases, _ = _event_map(root)
    _require(len(phases) == 16, "recovery phase count drift")
    ledger = IterationLedger(root / "state" / "iteration_ledger.jsonl").events
    ledger_types = [event["event_type"] for event in ledger]
    _require(ledger_types == [
        "run_initialized", "update_stage_started", "interrupted_stage_archived",
        "update_stage_started", "iteration_directory_committed", "run_state_advanced",
    ], "recovery ledger sequence drift")
    state = IterationStore(root / "state").load_state()
    _require(state["current_policy_version"] == "policy_0001", "recovery policy did not advance")
    _require(state["global_generated_action_tokens"] == collection["all_generated_action_tokens"], "recovery token ledger drift")
    iteration = report["iteration_manifest"]
    adapter = root / "state" / iteration["output_adapter"]["relative_path"]
    view = root / "state" / iteration["output_rollout_adapter"]["relative_path"]
    optimizer = root / "state" / iteration["output_optimizer"]["relative_path"]
    _require(directory_sha256(adapter) == iteration["output_adapter"]["sha256"], "recovery adapter hash drift")
    _require(directory_sha256(view) == iteration["output_rollout_adapter"]["sha256"], "recovery rollout-view hash drift")
    _require(sha256_file(optimizer) == iteration["output_optimizer"]["sha256"], "recovery optimizer hash drift")
    view_audit = validate_vllm_adapter_view(source_adapter=adapter, view_directory=view, base_model=Path("/data/share/model/Qwen3.5-4B"))
    _require(report["same_gpu_phase_switch"]["passed"] is True, "recovery same-GPU wake failed")
    return {
        "report_sha256": sha256_file(root / "preflight_report.json"),
        "collection_sha256": collection["collection_sha256"],
        "all_generated_action_tokens": collection["all_generated_action_tokens"],
        "committed_group_action_tokens": collection["committed_group_action_tokens"],
        "invalidated_action_token_cost": invalidated_cost,
        "invalidated_attempt_count": len(invalidated_attempts),
        "invalidated_stages": invalidated_stages,
        "strict_group_count": len(groups),
        "attempt_journal_event_count": len(journal_events),
        "attempt_journal_tail_sha256": journal_events[-1]["event_sha256"],
        "phase_event_count": len(phases),
        "phase_event_types": [event["event_type"] for event in phases],
        "ledger_event_types": ledger_types,
        "final_policy_version": state["current_policy_version"],
        "final_global_generated_action_tokens": state["global_generated_action_tokens"],
        "learner_zero_update_diagnostic": report["learner"]["optimizer_updates"] == 0,
        "adapter_sha256": directory_sha256(adapter),
        "rollout_adapter_sha256": directory_sha256(view),
        "optimizer_sha256": sha256_file(optimizer),
        "adapter_semantic_sha256": view_audit["semantic_tensor_sha256"],
        "same_gpu_phase_switch_passed": True,
        "passed": True,
    }


def build_readiness_manifest(
    *,
    repo_root: Path = PROJECT_ROOT,
    expected_git_sha: str,
    clean_regression_job: JobEvidence,
    sacct_path: Path = Path("/opt/slurm/slurm.25.05/bin/sacct"),
) -> dict[str, Any]:
    root = repo_root.resolve()
    git_sha = _run(["git", "rev-parse", "HEAD"], cwd=root).strip()
    _require(git_sha == expected_git_sha, "final readiness Git SHA drift")
    _require(not _run(["git", "status", "--porcelain"], cwd=root).strip(), "final readiness worktree is dirty")
    study = load_study_manifest()
    runtime = load_online_runtime_contract()
    base_model = validate_base_model_manifest(verify_files=True)
    token_audit = _json(SFT_CORPUS_ROOT / "token_audit.json")
    selection = _json(SFT_SELECTION_PATH)
    _require(token_audit["passed"] is True and token_audit["max_length"] == 6144, "SFT token audit failed")
    for split in ("train", "dev"):
        audit = token_audit["splits"][split]
        _require(audit["zero_completion_label_sample_count"] == 0, f"SFT {split} zero-label drift")
        _require(audit["duplicate_sample_count"] == 0 and audit["truncated_sample_count"] == 0, f"SFT {split} integrity drift")
    _require(selection["result"] == "PASS" and selection["formal_training"] is False, "SFT selection drift")
    _require(selection["selection"]["selected_microbatch"] == 8, "SFT microbatch selection drift")
    _require(study["payload"]["formal_submission_allowed"] is False, "study formal switch unexpectedly open")
    _require(runtime["payload"]["formal_submission_allowed"] is False, "runtime formal switch unexpectedly open")
    formal_root = root / study["payload"]["output_contract"]["formal_root"]
    _require(not formal_root.exists() or not any(formal_root.rglob("*")), "formal outputs already exist")
    preflight_root = root / study["payload"]["output_contract"]["preflight_root"]
    tracked = tracked_file_manifest(root)
    jobs = collect_job_evidence(root, sacct_path)
    clean_regression = collect_one_job_evidence(root, sacct_path, clean_regression_job)
    _require(clean_regression["state"] == "COMPLETED", "final clean regression did not complete")
    _require(clean_regression["allocated_cpus"] == 2, "final clean regression CPU request drift")
    jobs.append(clean_regression)
    selected = selected_e2e_evidence(preflight_root)
    soak = soak_evidence(preflight_root)
    recovery = recovery_evidence(preflight_root)
    gates = {
        "git_clean_and_frozen": True,
        "tracked_files_hashed": True,
        "base_model_rehashed": True,
        "study_dataset_prompt_frozen": True,
        "verified_sft_and_zero_label_gate": True,
        "sft_gpu_selection_passed": True,
        "runtime_v8_clean_cpu_regression_passed": True,
        "selected_32_lane_e2e_passed": True,
        "32_to_64_saturation_selection_passed": True,
        "selected_collection_phase_soak_30_minutes_passed": True,
        "real_slurm_collection_and_learner_resume_passed": True,
        "formal_outputs_absent": True,
        "formal_submission_switch_remains_closed": True,
    }
    manifest = {
        "schema_version": READINESS_SCHEMA,
        "study_id": STUDY_ID,
        "decision": "READY",
        "ready_for_formal_training_submission": True,
        "formal_submission_allowed": False,
        "formal_training_started": False,
        "readiness_only_does_not_submit_jobs": True,
        "unmet_gates": [],
        "git": {
            "sha": git_sha,
            "worktree_clean": True,
            **tracked,
        },
        "contracts": {
            "study_manifest_path": _relative_path(study["path"], root),
            "study_manifest_sha256": study["sha256"],
            "runtime_manifest_path": _relative_path(runtime["path"], root),
            "runtime_manifest_sha256": runtime["sha256"],
            "prompt_contract": PROMPT_CONTRACT,
            "prompt_sha256": sha256_file(PROMPT_PATH),
            "base_model_manifest_sha256": base_model["sha256"],
            "base_model_functional_file_set_sha256": base_model["payload"]["functional_file_set_sha256"],
            "dataset_manifest_sha256": study["payload"]["dataset_contract"]["dataset_manifest_sha256"],
            "seed_manifest_sha256": study["payload"]["dataset_contract"]["seed_manifest_sha256"],
        },
        "sft": {
            "corpus_root": str(SFT_CORPUS_ROOT.relative_to(root)),
            "corpus_manifest_sha256": token_audit["corpus_manifest_sha256"],
            "token_audit_file_sha256": sha256_file(SFT_CORPUS_ROOT / "token_audit.json"),
            "train": token_audit["splits"]["train"],
            "dev": token_audit["splits"]["dev"],
            "selection_manifest_sha256": sha256_file(SFT_SELECTION_PATH),
            "selected_microbatch": selection["selection"]["selected_microbatch"],
            "gradient_accumulation_steps": selection["selection"]["gradient_accumulation_steps"],
            "disposable_adapter_sha256": selection["adapter_audit"]["directory_sha256"],
        },
        "runtime_selection": {
            "selected_browser_workers": runtime["payload"]["rollout_contract"]["selected_browser_workers"],
            "selected_concurrent_k4_groups": runtime["payload"]["rollout_contract"]["selected_concurrent_k4_groups"],
            "saturation_gates": runtime["payload"]["telemetry_contract"]["gates"],
            "selected_e2e": selected,
            "soak": soak,
        },
        "recovery": recovery,
        "slurm_jobs": jobs,
        "software": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "vllm": importlib.metadata.version("vllm"),
        },
        "gates": gates,
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    return manifest


def write_readiness_manifest(manifest: Mapping[str, Any], output_dir: Path = READINESS_ROOT) -> Path:
    destination = Path(output_dir).resolve() / "readiness_manifest.json"
    atomic_write_json(destination, dict(manifest))
    return destination
