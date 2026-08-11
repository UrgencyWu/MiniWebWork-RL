#!/usr/bin/env python3
"""Build pre-authorization M5 readiness from the final clean-SHA evidence."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_file, sha256_json
from miniwebwork.m5_webshop_protocol import load_protocol
from miniwebwork.webshop_rl.formal_training import READINESS_SCHEMA, load_formal_plan, self_hash
from miniwebwork.webshop_rl.online_training import validate_learner_report


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _self_hashed(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = value.get("content_sha256")
    expected = dict(value)
    expected.pop("content_sha256", None)
    _require(value.get("schema_version") == schema, f"M5 readiness input schema drift: {path}")
    _require(observed == sha256_json(expected), f"M5 readiness input self-hash drift: {path}")
    return value


def build_readiness(args: argparse.Namespace) -> dict[str, Any]:
    git_sha = _git_sha()
    _require(git_sha == args.expected_git_sha, "M5 readiness Git SHA drift")
    _require(_clean(), "M5 readiness requires a clean tracked worktree")
    protocol = load_protocol()
    _require(protocol["git_sha"] == git_sha, "M5 readiness protocol Git drift")
    plan = load_formal_plan(args.formal_plan)
    evidence = plan["payload"]["preflight_evidence"]
    preflight_path = PROJECT_ROOT / evidence["report_path"]
    preflight = _self_hashed(preflight_path, "m5_webshop_online_preflight_report_v1")
    collection_path = preflight_path.parent / "collection_report.json"
    collection = _self_hashed(collection_path, "m5_webshop_online_collection_report_v1")
    learner_reports = {
        method: validate_learner_report(preflight_path.parent / "learners" / method / "learner_report.json")
        for method in plan["payload"]["matrix"]["methods"]
    }
    sft_adapter = PROJECT_ROOT / plan["payload"]["matrix"]["initial_adapter"]
    regression_stdout = Path(args.regression_stdout).expanduser().resolve()
    regression_stderr = Path(args.regression_stderr).expanduser().resolve()
    stdout_text = regression_stdout.read_text(encoding="utf-8", errors="replace")
    stderr_text = regression_stderr.read_text(encoding="utf-8", errors="replace")
    entrypoints = [
        PROJECT_ROOT / plan["payload"]["output_contract"]["formal_runner"],
        PROJECT_ROOT / plan["payload"]["output_contract"]["slurm_entrypoint"],
        PROJECT_ROOT / "scripts" / "m5_webshop_formal_readiness.py",
        PROJECT_ROOT / "scripts" / "m5_webshop_authorize_formal.py",
        PROJECT_ROOT / "src" / "miniwebwork" / "webshop_rl" / "formal_training.py",
        PROJECT_ROOT / "src" / "miniwebwork" / "webshop_rl" / "online_training.py",
    ]
    checks = {
        "clean_expected_git": _clean() and git_sha == args.expected_git_sha,
        "protocol_matches_passed_preflight": protocol["sha256"] == evidence["protocol_sha256"],
        "slurm_preflight_completed": args.preflight_job_id == evidence["slurm_job_id"] and args.preflight_state == "COMPLETED" and args.preflight_exit_code == "0:0",
        "preflight_report_passed": preflight.get("passed") is True and preflight.get("formal_training") is False and preflight.get("content_sha256") == evidence["report_content_sha256"],
        "preflight_producer_is_explicit": preflight.get("git_sha") == evidence["producer_git_sha"] and preflight.get("git_sha") != git_sha,
        "collection_passed": collection.get("passed") is True and all(collection.get("checks", {}).values()),
        "collection_cross_hash": sha256_file(collection_path) == preflight.get("collection_report_sha256"),
        "both_learners_complete": all(report.get("complete") is True and report.get("optimizer_updates", 0) >= 2 for report in learner_reports.values()),
        "both_learners_updated_parameters": all(report["output_adapter_sha256"] != report["input_adapter_sha256"] for report in learner_reports.values()),
        "finite_learner_metrics": all(
            all(
                isinstance(report[field], (int, float)) and math.isfinite(float(report[field]))
                for field in ("mean_loss", "maximum_absolute_loss", "mean_gradient_norm", "maximum_gradient_norm")
            )
            for report in learner_reports.values()
        ),
        "credit_and_cost_recorded": collection["metrics"]["informative_micro_turn_fraction"] >= 0.02 and collection["metrics"]["shared_noninitial_group_fraction"] >= 0.05 and collection["ledger"]["generated_action_tokens"] > 0,
        "gpu_telemetry_passed": all(preflight.get("telemetry_checks", {}).values()),
        "sft_adapter_unchanged": directory_sha256(sft_adapter) == evidence["sft_adapter_sha256"] == preflight["sft_compatibility"]["adapter_sha256"],
        "formal_matrix_is_six_runs": len(plan["payload"]["matrix"]["methods"]) * len(plan["payload"]["matrix"]["seeds"]) == 6,
        "formal_entrypoints_present": all(path.is_file() for path in entrypoints),
        "clean_sha_regression_completed": args.regression_state == "COMPLETED" and args.regression_exit_code == "0:0",
        "clean_sha_regression_log_bound": f"git_sha={git_sha}" in stdout_text and "passed" in stdout_text and "FAILED" not in stdout_text and "FAILED" not in stderr_text,
        "authorization_not_embedded_in_plan": plan["payload"]["formal_submission_allowed"] is False,
        "test_split_remains_closed": "not read by online training" in plan["payload"]["data_isolation"]["test"],
    }
    unmet = [name for name, passed in checks.items() if not passed]
    _require(not unmet, f"M5 readiness gates failed: {unmet}")
    report = {
        "schema_version": READINESS_SCHEMA,
        "study_id": plan["payload"]["study_id"],
        "decision": "READY_PENDING_USER_AUTHORIZATION",
        "ready": True,
        "formal_submission_allowed": False,
        "git_sha": git_sha,
        "protocol_sha256": protocol["sha256"],
        "formal_plan_path": str(Path(plan["path"]).relative_to(PROJECT_ROOT)),
        "formal_plan_sha256": plan["sha256"],
        "preflight": {
            "job_id": args.preflight_job_id,
            "state": args.preflight_state,
            "exit_code": args.preflight_exit_code,
            "producer_git_sha": preflight["git_sha"],
            "report_path": str(preflight_path),
            "report_sha256": sha256_file(preflight_path),
            "report_content_sha256": preflight["content_sha256"],
            "collection_report_sha256": sha256_file(collection_path),
        },
        "clean_regression": {
            "job_id": args.regression_job_id,
            "state": args.regression_state,
            "exit_code": args.regression_exit_code,
            "stdout_path": str(regression_stdout),
            "stdout_sha256": sha256_file(regression_stdout),
            "stderr_path": str(regression_stderr),
            "stderr_sha256": sha256_file(regression_stderr),
        },
        "learner_evidence": {
            method: {
                "optimizer_updates": report["optimizer_updates"],
                "effective_optimizer_action_token_fraction": report["effective_optimizer_action_token_fraction"],
                "mean_loss": report["mean_loss"],
                "maximum_gradient_norm": report["maximum_gradient_norm"],
                "output_adapter_sha256": report["output_adapter_sha256"],
            }
            for method, report in learner_reports.items()
        },
        "collection_metrics": collection["metrics"],
        "resource_contract": plan["payload"]["resource_contract"],
        "matrix": plan["payload"]["matrix"],
        "entrypoint_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in entrypoints},
        "checks": checks,
        "unmet_gates": unmet,
        "next_action": "obtain explicit user authorization, generate the bound authorization artifact, then submit the six online jobs",
    }
    report["content_sha256"] = self_hash(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--formal-plan", type=Path, default=PROJECT_ROOT / "data" / "m5_webshop_formal_plan_v1.json")
    parser.add_argument("--preflight-job-id", type=int, default=2139)
    parser.add_argument("--preflight-state", default="COMPLETED")
    parser.add_argument("--preflight-exit-code", default="0:0")
    parser.add_argument("--regression-job-id", type=int, required=True)
    parser.add_argument("--regression-state", required=True)
    parser.add_argument("--regression-exit-code", required=True)
    parser.add_argument("--regression-stdout", required=True)
    parser.add_argument("--regression-stderr", required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "readiness" / "readiness_manifest_v1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_readiness(args)
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
