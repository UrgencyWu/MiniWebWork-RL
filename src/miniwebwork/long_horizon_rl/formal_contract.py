"""Fail-closed authorization and path contract for the formal focused study.

The immutable study-v2 and runtime-v8 manifests remain historical preflight
records.  Formal execution requires this separate, narrowly scoped
authorization *and* a readiness-v2 artifact bound to the exact Git revision.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..m4_long_horizon_protocol import (
    ACTION_TOKEN_CAP,
    GROUP_SIZE,
    ONLINE_METHODS,
    ONLINE_SEEDS,
    PROJECT_ROOT,
    SFT_SEED,
    STUDY_ID,
    STUDY_MANIFEST_PATH,
    load_study_manifest,
)
from .contracts import sha256_file, sha256_json
from .runtime_contract import RUNTIME_CONTRACT_PATH, load_online_runtime_contract

FORMAL_AUTHORIZATION_SCHEMA = "m4_long_horizon_formal_authorization_v1"
FORMAL_AUTHORIZATION_PATH = (
    PROJECT_ROOT / "data" / "m4_long_horizon_formal_authorization_v1.json"
)
FORMAL_READINESS_SCHEMA = "m4_long_horizon_formal_training_readiness_v2"
EXPECTED_STUDY_SHA256 = "50ef68b4506f4c04750db631f57d47c9d45d964f1bbb0ef6e09a5bbb4c534bc7"
EXPECTED_RUNTIME_SHA256 = "06670a7caef86158dc849ac8a79d092267544b023c7b2fe45142089a48cdf6c4"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_ENTRYPOINTS = {
    "shared_sft": "scripts/m4_long_horizon_formal_sft.py",
    "online": "scripts/m4_long_horizon_formal_online.py",
    "evaluation": "scripts/m4_long_horizon_formal_eval.py",
    "analysis": "scripts/m4_long_horizon_formal_analyze.py",
    "readiness": "scripts/m4_long_horizon_readiness_v2.py",
    "shared_sft_slurm": "scripts/run_m4_long_horizon_formal_sft_job.sh",
    "online_slurm": "scripts/run_m4_long_horizon_formal_online_job.sh",
    "evaluation_slurm": "scripts/run_m4_long_horizon_formal_eval_job.sh",
    "analysis_slurm": "scripts/run_m4_long_horizon_formal_analysis_job.sh",
    "clean_regression_slurm": "scripts/run_m4_long_horizon_formal_clean_regression_job.sh",
    "readiness_slurm": "scripts/run_m4_long_horizon_readiness_v2_job.sh",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _project_path(relative_path: str) -> Path:
    _require(isinstance(relative_path, str) and relative_path, "empty formal path")
    candidate = (PROJECT_ROOT / relative_path).resolve()
    project = PROJECT_ROOT.resolve()
    _require(
        os.path.commonpath((str(candidate), str(project))) == str(project),
        f"formal path escapes project root: {relative_path}",
    )
    return candidate


def validate_formal_authorization(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == FORMAL_AUTHORIZATION_SCHEMA, "formal authorization schema drift")
    _require(payload.get("study_id") == STUDY_ID, "formal authorization study drift")
    _require(payload.get("lifecycle_state") == "formal_ready", "formal lifecycle is not ready")
    _require(payload.get("formal_submission_allowed") is True, "formal submission is not authorized")
    _require(
        payload.get("authorization_scope")
        == "shared_sft_then_two_online_methods_three_seeds_then_frozen_test_only",
        "formal authorization scope drift",
    )

    immutable = payload.get("immutable_preflight_contracts", {})
    _require(immutable.get("study_manifest_path") == str(STUDY_MANIFEST_PATH.relative_to(PROJECT_ROOT)), "formal study path drift")
    _require(immutable.get("study_manifest_sha256") == EXPECTED_STUDY_SHA256, "formal study hash drift")
    _require(immutable.get("study_lifecycle_state") == "preflight", "historical study lifecycle drift")
    _require(immutable.get("study_formal_submission_allowed") is False, "historical study switch drift")
    _require(immutable.get("runtime_manifest_path") == str(RUNTIME_CONTRACT_PATH.relative_to(PROJECT_ROOT)), "formal runtime path drift")
    _require(immutable.get("runtime_manifest_sha256") == EXPECTED_RUNTIME_SHA256, "formal runtime hash drift")
    _require(immutable.get("runtime_lifecycle_state") == "preflight", "historical runtime lifecycle drift")
    _require(immutable.get("runtime_formal_submission_allowed") is False, "historical runtime switch drift")
    _require(sha256_file(STUDY_MANIFEST_PATH) == EXPECTED_STUDY_SHA256, "immutable study file changed")
    _require(sha256_file(RUNTIME_CONTRACT_PATH) == EXPECTED_RUNTIME_SHA256, "immutable runtime file changed")
    study = load_study_manifest()["payload"]
    runtime = load_online_runtime_contract()["payload"]
    _require(study["lifecycle_state"] == "preflight" and study["formal_submission_allowed"] is False, "base study is not immutable preflight")
    _require(runtime["lifecycle_state"] == "preflight" and runtime["formal_submission_allowed"] is False, "base runtime is not immutable preflight")

    hard = payload.get("gate_resolution", {}).get("hard_gates", {})
    runtime_gates = runtime["telemetry_contract"]["gates"]
    expected_hard = {
        "rollout_trajectories_per_hour_minimum": runtime_gates["rollout_trajectories_per_hour_minimum"],
        "generation_gpu_utilization_p95_minimum": runtime_gates["generation_gpu_utilization_p95_minimum"],
        "learner_gpu_utilization_p50_minimum": runtime_gates["learner_gpu_utilization_p50_minimum"],
        "vram_headroom_minimum": runtime_gates["vram_headroom_minimum"],
        "selected_browser_workers": runtime_gates["saturation_selected_browser_workers"],
        "saturation_challenger_browser_workers": runtime_gates["saturation_challenger_browser_workers"],
        "saturation_challenger_throughput_improvement_maximum": runtime_gates["saturation_challenger_throughput_improvement_maximum"],
        "real_slurm_resume_smoke_required": True,
        "post_wake_generation_smoke_required": True,
    }
    _require(hard == expected_hard, "formal hard-gate resolution drift")
    diagnostics = payload.get("gate_resolution", {}).get("diagnostics", {})
    _require(diagnostics.get("generation_gpu_utilization_p50", {}).get("hard_gate") is False, "generation P50 was silently made hard")
    _require(diagnostics.get("generation_gpu_utilization_p50", {}).get("historical_target") == 0.6, "generation P50 history drift")
    _require(diagnostics.get("optimizer_action_token_fraction", {}).get("target") == runtime_gates["optimizer_action_token_fraction_target"], "optimizer-token target drift")
    _require(diagnostics.get("optimizer_action_token_fraction", {}).get("hard_gate") is False, "optimizer-token diagnostic was silently made hard")

    matrix = payload.get("formal_matrix", {})
    _require(matrix.get("shared_sft") == {"method": "verified_sft", "seed": SFT_SEED, "model_count": 1}, "formal shared-SFT matrix drift")
    _require(tuple(matrix.get("online_methods", ())) == ONLINE_METHODS, "formal online methods drift")
    _require(tuple(matrix.get("online_seeds", ())) == ONLINE_SEEDS, "formal online seeds drift")
    _require(matrix.get("online_model_count") == 6 and matrix.get("total_model_count") == 7, "formal model count drift")
    _require(matrix.get("generated_action_token_cap_per_online_run") == ACTION_TOKEN_CAP, "formal token cap drift")
    _require(matrix.get("group_size") == GROUP_SIZE, "formal K drift")
    _require(matrix.get("frozen_test_tasks") == 120 and matrix.get("frozen_test_rollouts_per_task") == 4, "formal frozen-test matrix drift")

    outputs = payload.get("output_contract", {})
    formal_root = _project_path(outputs.get("formal_root", ""))
    _require(formal_root == (PROJECT_ROOT / "outputs" / STUDY_ID / "formal").resolve(), "formal root drift")
    for key in ("shared_sft_root", "evaluation_root", "analysis_root"):
        child = _project_path(outputs.get(key, ""))
        _require(os.path.commonpath((str(formal_root), str(child))) == str(formal_root), f"{key} escapes formal root")
    _require(outputs.get("online_root_template") == f"outputs/{STUDY_ID}/formal/online/{{method}}/seed_{{seed}}", "formal online root template drift")
    _require(outputs.get("readiness_manifest") == f"outputs/{STUDY_ID}/readiness/readiness_manifest_v2.json", "formal readiness path drift")
    _require(payload.get("entrypoint_contract") == EXPECTED_ENTRYPOINTS, "formal entrypoint inventory drift")
    resources = payload.get("resource_contract", {})
    _require(resources == {
        "maximum_concurrent_gpu_jobs": 4,
        "gpus_per_job": 1,
        "wall_time": "24:00:00",
        "shared_sft": {"cpus": 4, "memory_gb": 32},
        "online": {"cpus": 8, "memory_gb": 48},
        "evaluation": {"cpus": 8, "memory_gb": 48},
        "analysis": {"cpus": 2, "memory_gb": 8, "gpus": 0},
        "clean_regression": {"cpus": 2, "memory_gb": 8, "gpus": 0},
        "readiness": {"cpus": 2, "memory_gb": 8, "gpus": 0},
    }, "formal resource contract drift")
    ordering = payload.get("ordering_contract", {})
    for field in (
        "readiness_v2_before_any_formal_submission",
        "shared_sft_before_online",
        "shared_sft_artifact_audit_before_online",
        "all_six_online_artifact_audits_before_frozen_test",
        "all_seven_frozen_test_artifact_audits_before_analysis",
    ):
        _require(ordering.get(field) is True, f"formal ordering gate disabled: {field}")
    _require(ordering.get("maximum_first_wave_online_jobs") == 4, "formal first-wave concurrency drift")


def load_formal_authorization(path: Path = FORMAL_AUTHORIZATION_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_formal_authorization(payload)
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}


def formal_output_path(path: Path, authorization: Mapping[str, Any] | None = None) -> Path:
    payload = dict(authorization) if authorization is not None else load_formal_authorization()["payload"]
    validate_formal_authorization(payload)
    formal_root = _project_path(payload["output_contract"]["formal_root"])
    requested = Path(path).expanduser()
    actual = requested.resolve() if requested.is_absolute() else (PROJECT_ROOT / requested).resolve()
    _require(os.path.commonpath((str(formal_root), str(actual))) == str(formal_root), f"not a formal output: {actual}")
    return actual


def current_git_sha(repo_root: Path = PROJECT_ROOT) -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    _require(GIT_SHA_RE.fullmatch(sha) is not None, "invalid Git SHA")
    return sha


def assert_clean_tracked_worktree(repo_root: Path = PROJECT_ROOT) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    _require(not status, "formal execution requires a clean tracked worktree")


def validate_readiness_for_submission(
    path: Path | None = None,
    *,
    expected_git_sha: str | None = None,
) -> dict[str, Any]:
    authorization = load_formal_authorization()
    readiness_path = path or _project_path(authorization["payload"]["output_contract"]["readiness_manifest"])
    readiness_path = Path(readiness_path).expanduser().resolve()
    _require(readiness_path.is_file(), f"formal readiness-v2 artifact is missing: {readiness_path}")
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == FORMAL_READINESS_SCHEMA, "formal readiness schema drift")
    _require(payload.get("study_id") == STUDY_ID, "formal readiness study drift")
    _require(payload.get("decision") == "READY", "formal readiness decision is not READY")
    _require(payload.get("ready_for_formal_training_submission") is True, "formal readiness gate is closed")
    _require(payload.get("formal_submission_allowed") is True, "formal readiness does not carry authorization")
    _require(payload.get("unmet_gates") == [], "formal readiness has unmet gates")
    recorded_content_sha = payload.get("manifest_content_sha256")
    content = dict(payload)
    content.pop("manifest_content_sha256", None)
    _require(recorded_content_sha == sha256_json(content), "formal readiness self-hash drift")
    git_sha = current_git_sha()
    _require(payload.get("git", {}).get("sha") == git_sha, "formal readiness Git binding drift")
    if expected_git_sha is not None:
        _require(git_sha == expected_git_sha, "formal expected Git SHA drift")
    _require(payload.get("git", {}).get("worktree_clean") is True, "formal readiness did not audit a clean tree")
    contracts = payload.get("contracts", {})
    _require(contracts.get("authorization_sha256") == authorization["sha256"], "formal readiness authorization binding drift")
    _require(contracts.get("study_manifest_sha256") == EXPECTED_STUDY_SHA256, "formal readiness study binding drift")
    _require(contracts.get("runtime_manifest_sha256") == EXPECTED_RUNTIME_SHA256, "formal readiness runtime binding drift")
    required = payload.get("gates", {})
    _require(required and all(value is True for value in required.values()), "formal readiness contains a failed gate")
    return {"path": str(readiness_path), "sha256": sha256_file(readiness_path), "payload": payload}
