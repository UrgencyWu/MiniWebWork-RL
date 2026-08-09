"""Computed, fail-closed readiness-v2 audit for formal training submission."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..m4_long_horizon_protocol import PROJECT_ROOT, STUDY_ID, assert_dataset_binding, load_study_manifest
from .contracts import atomic_write_json, sha256_file, sha256_json
from .formal_contract import (
    EXPECTED_ENTRYPOINTS,
    FORMAL_READINESS_SCHEMA,
    load_formal_authorization,
)
from .model_manifest import validate_base_model_manifest
from .readiness import (
    JobEvidence,
    READINESS_ROOT,
    SFT_CORPUS_ROOT,
    SFT_SELECTION_PATH,
    collect_job_evidence,
    collect_one_job_evidence,
    recovery_evidence,
    selected_e2e_evidence,
    soak_evidence,
    tracked_file_manifest,
)
from .runtime_contract import load_online_runtime_contract
from .sft_selection import load_sft_preflight_selection
from .sft_trainer import validate_sft_training_inputs

READINESS_V2_NAME = "readiness_manifest_v2.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON payload must be an object: {path}")
    return value


def _run(command: Sequence[str], *, cwd: Path) -> str:
    return subprocess.run(list(command), cwd=cwd, check=True, text=True, stdout=subprocess.PIPE).stdout


def _is_tracked(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=root,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _slurm_directives(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"#SBATCH --([^= ]+)(?:=| +)(.+)$", line.strip())
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def formal_suite_evidence(root: Path, authorization: Mapping[str, Any]) -> dict[str, Any]:
    entries = authorization["entrypoint_contract"]
    _require(entries == EXPECTED_ENTRYPOINTS, "readiness formal entrypoint contract drift")
    files = []
    for role, relative in sorted(entries.items()):
        path = root / relative
        _require(path.is_file(), f"formal entrypoint is missing: {relative}")
        _require(_is_tracked(root, relative), f"formal entrypoint is not tracked: {relative}")
        files.append({"role": role, "path": relative, "sha256": sha256_file(path)})
    expected_resources = {
        "shared_sft_slurm": {"time": "24:00:00", "cpus-per-task": "4", "mem": "32G", "gres": "gpu:1"},
        "online_slurm": {"time": "24:00:00", "cpus-per-task": "8", "mem": "48G", "gres": "gpu:1"},
        "evaluation_slurm": {"time": "24:00:00", "cpus-per-task": "8", "mem": "48G", "gres": "gpu:1"},
        "analysis_slurm": {"time": "24:00:00", "cpus-per-task": "2", "mem": "8G"},
        "clean_regression_slurm": {"time": "24:00:00", "cpus-per-task": "2", "mem": "8G"},
        "readiness_slurm": {"time": "24:00:00", "cpus-per-task": "2", "mem": "8G"},
    }
    slurm = {}
    for role, expected in expected_resources.items():
        directives = _slurm_directives(root / entries[role])
        observed = {field: directives.get(field) for field in expected}
        _require(observed == expected, f"formal Slurm resource drift: {role}")
        slurm[role] = observed
    for role in ("shared_sft_slurm", "online_slurm", "evaluation_slurm"):
        source = (root / entries[role]).read_text(encoding="utf-8")
        _require(
            'nvidia-smi -i "$CUDA_VISIBLE_DEVICES"' in source
            and "--loop=5" in source
            and "telemetry_samples=" in source,
            f"formal single-device GPU telemetry drift: {role}",
        )
    required_source_markers = {
        "shared_sft": ["prepare_sft_attempt", "archive_incomplete_attempt", "microbatch_size=8", "start_formal_invocation"],
        "online": ["run_online_formal", "ACTION_TOKEN_CAP", "process_restart_recovery", "start_formal_invocation"],
        "evaluation": ["expected_trajectory_count", "480", "training_updates_allowed", "start_formal_invocation"],
        "analysis": ["trajectory_count\": 3360", "failure_categories", "paired_seed_task", "collect_slurm_accounting", "training_dynamics"],
    }
    for role, markers in required_source_markers.items():
        source = (root / entries[role]).read_text(encoding="utf-8")
        if role in {"shared_sft", "online", "evaluation", "analysis"}:
            implementation = root / "src" / "miniwebwork" / "long_horizon_rl" / f"formal_{'analysis' if role == 'analysis' else 'eval' if role == 'evaluation' else 'sft' if role == 'shared_sft' else 'online'}.py"
            source += implementation.read_text(encoding="utf-8")
        _require(all(marker in source for marker in markers), f"formal feature inventory drift: {role}")
    return {
        "entrypoints": files,
        "entrypoint_set_sha256": sha256_json(files),
        "slurm_resources": slurm,
        "shared_sft_recovery_implemented": True,
        "online_multi_iteration_250k_recovery_implemented": True,
        "frozen_test_120xK4_implemented": True,
        "seven_model_statistics_and_lineage_implemented": True,
        "passed": True,
    }


def saturation_evidence(preflight_root: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    candidates = []
    for report_path in preflight_root.rglob("preflight_report.json"):
        config_path = report_path.parent / "run_config.json"
        if not config_path.is_file():
            continue
        report, config = _json(report_path), _json(config_path)
        performance = report.get("collection_performance") or {}
        if (
            report.get("result") == "PASS"
            and report.get("complete") is True
            and config.get("browser_workers") == 64
            and performance.get("trajectory_count") == 64
            and performance.get("group_count") == 16
        ):
            candidates.append((report_path, report, config))
    _require(len(candidates) == 1, f"expected exactly one reviewed 64-lane saturation artifact, found {len(candidates)}")
    report_path, report, config = candidates[0]
    gates = runtime["telemetry_contract"]["gates"]
    selected = float(gates["saturation_selected_trajectories_per_hour"])
    challenger = float(report["collection_performance"]["trajectories_per_hour"])
    _require(abs(challenger - float(gates["saturation_challenger_trajectories_per_hour"])) < 1e-9, "64-lane throughput/runtime evidence drift")
    ratio = challenger / selected
    _require(ratio <= float(gates["saturation_challenger_throughput_improvement_maximum"]), "64-lane challenger invalidated the 32-lane selection")
    return {
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
        "report_sha256": sha256_file(report_path),
        "selected_browser_workers": gates["saturation_selected_browser_workers"],
        "challenger_browser_workers": config["browser_workers"],
        "selected_trajectories_per_hour": selected,
        "challenger_trajectories_per_hour": challenger,
        "challenger_to_selected_ratio": ratio,
        "maximum_allowed_ratio": gates["saturation_challenger_throughput_improvement_maximum"],
        "passed": True,
    }


def _clean_regression_evidence(
    root: Path,
    sacct_path: Path,
    job: JobEvidence,
) -> dict[str, Any]:
    evidence = collect_one_job_evidence(root, sacct_path, job)
    _require(evidence["state"] == "COMPLETED" and evidence["exit_code"] == "0:0", "final clean regression failed")
    _require(evidence["allocated_cpus"] == 2, "final clean regression CPU request drift")
    stdout = Path(job.stdout_path)
    if not stdout.is_absolute():
        stdout = root / stdout
    text = stdout.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"([0-9]+) passed(?:, ([0-9]+) deselected)?", text)
    _require(match is not None and int(match.group(1)) >= 430, "final clean regression did not report the full passing suite")
    _require(" failed" not in text, "final clean regression log contains failures")
    evidence["pytest_passed"] = int(match.group(1))
    evidence["pytest_deselected"] = int(match.group(2) or 0)
    return evidence


def build_readiness_v2(
    *,
    expected_git_sha: str,
    clean_regression_job: JobEvidence,
    repo_root: Path = PROJECT_ROOT,
    sacct_path: Path = Path("/opt/slurm/slurm.25.05/bin/sacct"),
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    git_sha = _run(["git", "rev-parse", "HEAD"], cwd=root).strip()
    _require(git_sha == expected_git_sha, "readiness-v2 Git SHA drift")
    tracked_status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root).strip()
    _require(not tracked_status, "readiness-v2 tracked worktree is dirty")
    authorization = load_formal_authorization()
    study = load_study_manifest()
    runtime = load_online_runtime_contract()
    dataset = assert_dataset_binding(study["payload"])
    base_model = validate_base_model_manifest(verify_files=True)
    selection = load_sft_preflight_selection()
    sft_input = validate_sft_training_inputs(
        SFT_CORPUS_ROOT,
        Path(runtime["payload"]["model_contract"]["base_model_path"]),
    )
    _require(selection["payload"]["selection"]["selected_microbatch"] == 8, "readiness-v2 SFT selection drift")
    _require(selection["payload"]["selection"]["gradient_accumulation_steps"] == 2, "readiness-v2 SFT accumulation drift")
    formal_root = root / authorization["payload"]["output_contract"]["formal_root"]
    formal_outputs_absent = not formal_root.exists() or not any(formal_root.rglob("*"))
    _require(formal_outputs_absent, "formal outputs exist before readiness-v2")
    preflight_root = root / study["payload"]["output_contract"]["preflight_root"]
    selected = selected_e2e_evidence(preflight_root)
    soak = soak_evidence(preflight_root)
    recovery = recovery_evidence(preflight_root)
    saturation = saturation_evidence(preflight_root, runtime["payload"])
    all_jobs = collect_job_evidence(root, sacct_path)
    clean = _clean_regression_evidence(root, sacct_path, clean_regression_job)
    suite = formal_suite_evidence(root, authorization["payload"])
    hard = authorization["payload"]["gate_resolution"]["hard_gates"]
    selected_headroom = min(
        selected["generation_telemetry"]["minimum_vram_headroom_fraction"],
        selected["learner_telemetry"]["minimum_vram_headroom_fraction"],
        selected["post_wake_telemetry"]["minimum_vram_headroom_fraction"],
    )
    computed_gates = {
        "git_clean_and_frozen": git_sha == expected_git_sha and not tracked_status,
        "immutable_study_runtime_hashes_match_authorization": (
            study["sha256"] == authorization["payload"]["immutable_preflight_contracts"]["study_manifest_sha256"]
            and runtime["sha256"] == authorization["payload"]["immutable_preflight_contracts"]["runtime_manifest_sha256"]
        ),
        "dataset_prompt_base_model_bound": bool(dataset) and bool(base_model),
        "verified_unique_sft_zero_label_and_no_truncation": all(
            sft_input["splits"][split]["duplicate_sample_count"] == 0
            and sft_input["splits"][split]["zero_completion_label_sample_count"] == 0
            and sft_input["splits"][split]["truncated_sample_count"] == 0
            for split in ("train", "dev")
        ),
        "sft_microbatch8_accumulation2_selected": selection["payload"]["selection"]["selected_microbatch"] == 8 and selection["payload"]["selection"]["gradient_accumulation_steps"] == 2,
        "formal_entrypoints_recovery_and_resources_validated": suite["passed"],
        "rollout_throughput_hard_gate": selected["collection_performance"]["trajectories_per_hour"] >= hard["rollout_trajectories_per_hour_minimum"],
        "generation_p95_hard_gate": selected["generation_telemetry"]["gpu_utilization_p95"] >= hard["generation_gpu_utilization_p95_minimum"],
        "learner_p50_hard_gate": selected["learner_telemetry"]["gpu_utilization_p50"] >= hard["learner_gpu_utilization_p50_minimum"],
        "vram_headroom_hard_gate": selected_headroom >= hard["vram_headroom_minimum"],
        "same_gpu_commit_wake_generation_gate": selected["same_gpu_phase_switch"]["passed"] is True,
        "32_lane_saturation_gate": saturation["passed"],
        "30_minute_collection_soak_gate": soak["total_phase_seconds"] >= 1800.0,
        "real_collection_and_learner_interruption_recovery_gate": recovery["passed"],
        "final_clean_full_cpu_regression_gate": clean["state"] == "COMPLETED" and clean["pytest_passed"] >= 430,
        "formal_outputs_absent_before_submission": formal_outputs_absent,
    }
    unmet = sorted(name for name, passed in computed_gates.items() if passed is not True)
    manifest = {
        "schema_version": FORMAL_READINESS_SCHEMA,
        "study_id": STUDY_ID,
        "decision": "READY" if not unmet else "NOT_READY",
        "ready_for_formal_training_submission": not unmet,
        "formal_submission_allowed": authorization["payload"]["formal_submission_allowed"] and not unmet,
        "formal_training_started": False,
        "unmet_gates": unmet,
        "git": {"sha": git_sha, "worktree_clean": not tracked_status, **tracked_file_manifest(root)},
        "contracts": {
            "authorization_path": str(Path(authorization["path"]).relative_to(root)),
            "authorization_sha256": authorization["sha256"],
            "study_manifest_sha256": study["sha256"],
            "runtime_manifest_sha256": runtime["sha256"],
            "base_model_manifest_sha256": base_model["sha256"],
            "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
            "seed_manifest_sha256": dataset["seed_manifest_sha256"],
        },
        "sft": {
            "input_binding": sft_input,
            "selection_sha256": selection["sha256"],
            "selected_microbatch": selection["payload"]["selection"]["selected_microbatch"],
            "gradient_accumulation_steps": selection["payload"]["selection"]["gradient_accumulation_steps"],
        },
        "formal_suite": suite,
        "computed_evidence": {
            "selected_32_lane_e2e": selected,
            "saturation": saturation,
            "soak": soak,
            "recovery": recovery,
            "hard_gate_contract": hard,
            "diagnostics_not_used_as_hard_gates": {
                "generation_gpu_utilization_p50": selected["generation_telemetry"]["gpu_utilization_p50"],
                "optimizer_action_token_fraction": selected["learner"]["effective_optimizer_action_token_fraction"],
                "optimizer_action_token_fraction_target": authorization["payload"]["gate_resolution"]["diagnostics"]["optimizer_action_token_fraction"]["target"],
            },
        },
        "slurm_jobs": all_jobs + [clean],
        "gates": computed_gates,
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    return manifest


def write_readiness_v2(manifest: Mapping[str, Any], output_dir: Path = READINESS_ROOT) -> Path:
    destination = Path(output_dir).resolve() / READINESS_V2_NAME
    atomic_write_json(destination, dict(manifest))
    return destination
