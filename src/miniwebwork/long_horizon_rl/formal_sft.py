"""Recoverable formal shared-SFT lifecycle and artifact validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..m4_long_horizon_protocol import (
    PROJECT_ROOT,
    SFT_EFFECTIVE_BATCH_SIZE,
    SFT_SEED,
    SFT_STOPPING_RULE,
    STUDY_ID,
)
from .adapter_view import build_vllm_adapter_view, validate_vllm_adapter_view
from .contracts import atomic_write_json, directory_sha256, sha256_file, sha256_json
from .formal_contract import (
    assert_clean_tracked_worktree,
    current_git_sha,
    formal_output_path,
    load_formal_authorization,
    validate_readiness_for_submission,
)
from .sft_selection import load_sft_preflight_selection
from .sft_trainer import validate_sft_training_inputs

FORMAL_SFT_SCHEMA = "m4_long_horizon_formal_sft_v1"
FORMAL_SFT_RUN_CONFIG_SCHEMA = "m4_long_horizon_formal_sft_run_config_v1"
FORMAL_SFT_ROOT = PROJECT_ROOT / "outputs" / STUDY_ID / "formal" / "shared_sft" / f"seed_{SFT_SEED}"
FORMAL_SFT_MANIFEST_NAME = "formal_sft_manifest.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON payload must be an object: {path}")
    return payload


def _relative(path: Path, root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    return str(resolved.relative_to(root.resolve()))


def _manifest_content_sha(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("manifest_content_sha256", None)
    return sha256_json(content)


def build_formal_sft_run_config(
    *,
    output_root: Path,
    expected_git_sha: str,
    readiness_path: Path | None,
    base_model: Path,
    data_dir: Path,
    dataloader_workers: int,
) -> dict[str, Any]:
    root = formal_output_path(output_root)
    expected_root = FORMAL_SFT_ROOT.resolve()
    _require(root == expected_root, f"shared SFT must use the frozen root: {expected_root}")
    _require(dataloader_workers in range(0, 4), "formal SFT workers must be 0-3")
    assert_clean_tracked_worktree()
    git_sha = current_git_sha()
    _require(git_sha == expected_git_sha, "formal SFT Git SHA drift")
    authorization = load_formal_authorization()
    readiness = validate_readiness_for_submission(readiness_path, expected_git_sha=git_sha)
    selection = load_sft_preflight_selection()
    _require(selection["payload"]["selection"]["selected_microbatch"] == 8, "formal SFT microbatch is not selected 8")
    _require(selection["payload"]["selection"]["gradient_accumulation_steps"] == 2, "formal SFT accumulation is not selected 2")
    config = {
        "schema_version": FORMAL_SFT_RUN_CONFIG_SCHEMA,
        "study_id": STUDY_ID,
        "formal_training": True,
        "method": "verified_sft",
        "seed": SFT_SEED,
        "git_sha": git_sha,
        "authorization_sha256": authorization["sha256"],
        "readiness_sha256": readiness["sha256"],
        "base_model": str(Path(base_model).expanduser().resolve()),
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "microbatch_size": 8,
        "effective_batch_size": SFT_EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation_steps": 2,
        "dataloader_workers": dataloader_workers,
        "maximum_epochs": SFT_STOPPING_RULE["maximum_epochs"],
        "minimum_epochs": SFT_STOPPING_RULE["minimum_epochs"],
        "recovery": "archive_incomplete_attempt_then_deterministically_restart_from_frozen_base",
        "output_root": str(root),
    }
    config_path = root / "run_config.json"
    if config_path.is_file():
        _require(_load_json(config_path) == config, "formal SFT run-config identity drift")
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_path, config)
    return config


def _attempt_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid formal SFT attempt directory: {path.name}") from exc


def prepare_sft_attempt(root: Path) -> tuple[Path, bool]:
    """Return an active attempt, archiving only an incomplete previous one.

    The boolean is true when a completed training report is being reused for
    finalization after a process interruption.
    """

    root = Path(root).resolve()
    attempts_root = root / "attempts"
    invalid_root = root / "invalidated_attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    active = sorted(path for path in attempts_root.iterdir() if path.is_dir())
    _require(len(active) <= 1, "multiple active formal SFT attempts")
    if active:
        attempt = active[0]
        report_path = attempt / "training" / "training_report.json"
        if report_path.is_file() and _load_json(report_path).get("complete") is True:
            return attempt, True
        invalid_root.mkdir(parents=True, exist_ok=True)
        archived = invalid_root / attempt.name
        _require(not archived.exists(), f"formal SFT invalidated-attempt collision: {archived}")
        os.rename(attempt, archived)
        directory_fd = os.open(invalid_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    historical = [
        _attempt_number(path)
        for parent in (attempts_root, invalid_root)
        if parent.exists()
        for path in parent.iterdir()
        if path.is_dir() and path.name.startswith("attempt-")
    ]
    number = max(historical, default=0) + 1
    attempt = attempts_root / f"attempt-{number:04d}"
    attempt.mkdir()
    atomic_write_json(
        attempt / "attempt.json",
        {
            "schema_version": "m4_long_horizon_formal_sft_attempt_v1",
            "attempt_index": number,
            "complete": False,
            "restarts_from_frozen_base": True,
        },
    )
    return attempt, False


def finalize_formal_sft(
    *,
    root: Path,
    config: Mapping[str, Any],
    attempt: Path,
    input_binding: Mapping[str, Any],
    loaded_audit: Mapping[str, Any],
    completed_attempt_reused: bool,
) -> dict[str, Any]:
    root = Path(root).resolve()
    training_report_path = attempt / "training" / "training_report.json"
    _require(training_report_path.is_file(), "formal SFT training report is missing")
    training = _load_json(training_report_path)
    _require(training.get("complete") is True, "formal SFT training is incomplete")
    history = training.get("history", [])
    _require(SFT_STOPPING_RULE["minimum_epochs"] <= len(history) <= SFT_STOPPING_RULE["maximum_epochs"], "formal SFT epoch count drift")
    _require(training.get("microbatch_size") == 8, "formal SFT microbatch drift")
    _require(training.get("effective_batch_size") == 16, "formal SFT effective-batch drift")
    _require(training.get("gradient_accumulation") == 2, "formal SFT gradient accumulation drift")
    _require(training.get("stop_reason") in {"maximum_epochs", "dev_plateau"}, "formal SFT stopping rule drift")
    train_audit = training.get("train_audit", {})
    dev_audit = training.get("dev_audit", {})
    _require(train_audit == loaded_audit.get("train"), "formal SFT loaded train audit drift")
    _require(dev_audit == loaded_audit.get("dev"), "formal SFT loaded dev audit drift")
    for row in history:
        _require(row.get("train_seen_unique_samples") == train_audit.get("sample_count"), "formal SFT epoch did not consume the unique train roster")
        _require(row.get("train_completion_label_tokens") == train_audit.get("completion_label_tokens"), "formal SFT epoch token ledger drift")
    best_epoch = training.get("best_epoch")
    _require(isinstance(best_epoch, int) and 1 <= best_epoch <= len(history), "formal SFT best epoch drift")
    adapter = Path(training["final_adapter"]).resolve()
    _require(adapter == (attempt / "training" / "final_adapter").resolve(), "formal SFT final adapter path drift")
    view = root / "final_rollout_adapter"
    view_audit = build_vllm_adapter_view(
        source_adapter=adapter,
        destination=view,
        base_model=Path(config["base_model"]),
    )
    manifest = {
        "schema_version": FORMAL_SFT_SCHEMA,
        "study_id": STUDY_ID,
        "formal_training": True,
        "complete": True,
        "method": "verified_sft",
        "seed": SFT_SEED,
        "git_sha": config["git_sha"],
        "authorization_sha256": config["authorization_sha256"],
        "readiness_sha256": config["readiness_sha256"],
        "run_config_sha256": sha256_file(root / "run_config.json"),
        "attempt_index": _attempt_number(attempt),
        "recovery": {
            "invalidated_attempt_count": len(list((root / "invalidated_attempts").glob("attempt-*"))) if (root / "invalidated_attempts").exists() else 0,
            "completed_attempt_reused": completed_attempt_reused,
        },
        "input_binding": dict(input_binding),
        "loaded_audit": dict(loaded_audit),
        "training": {
            "report": _relative(training_report_path, root),
            "report_sha256": sha256_file(training_report_path),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_dev_nll": training["best_dev_nll"],
            "optimizer_updates": training["optimizer_updates"],
            "stop_reason": training["stop_reason"],
            "microbatch_size": training["microbatch_size"],
            "effective_batch_size": training["effective_batch_size"],
            "gradient_accumulation_steps": training["gradient_accumulation"],
        },
        "canonical_adapter": {
            "relative_path": _relative(adapter, root),
            "directory_sha256": directory_sha256(adapter),
        },
        "rollout_adapter": {
            "relative_path": _relative(view, root),
            "directory_sha256": view_audit["view_directory_sha256"],
            "semantic_tensor_sha256": view_audit["semantic_tensor_sha256"],
            "view_manifest_sha256": sha256_file(Path(view_audit["manifest_path"])),
        },
    }
    manifest["manifest_content_sha256"] = _manifest_content_sha(manifest)
    attempt_record = _load_json(attempt / "attempt.json")
    attempt_record["complete"] = True
    attempt_record["training_report_sha256"] = sha256_file(training_report_path)
    atomic_write_json(attempt / "attempt.json", attempt_record)
    # Publish the top-level completion marker last.  A 24-hour interruption
    # before this atomic write remains safely recoverable from the completed
    # attempt; after it, every artifact required by downstream validation is
    # already durable.
    atomic_write_json(root / FORMAL_SFT_MANIFEST_NAME, manifest)
    return validate_formal_sft_manifest(root)


def validate_formal_sft_manifest(root: Path = FORMAL_SFT_ROOT) -> dict[str, Any]:
    root = formal_output_path(root)
    _require(root == FORMAL_SFT_ROOT.resolve(), "formal SFT root drift")
    path = root / FORMAL_SFT_MANIFEST_NAME
    _require(path.is_file(), "formal SFT manifest is missing")
    payload = _load_json(path)
    _require(payload.get("schema_version") == FORMAL_SFT_SCHEMA, "formal SFT schema drift")
    _require(payload.get("study_id") == STUDY_ID and payload.get("method") == "verified_sft", "formal SFT identity drift")
    _require(payload.get("formal_training") is True and payload.get("complete") is True, "formal SFT is incomplete")
    _require(payload.get("seed") == SFT_SEED, "formal SFT seed drift")
    _require(payload.get("manifest_content_sha256") == _manifest_content_sha(payload), "formal SFT self-hash drift")
    config_path = root / "run_config.json"
    config = _load_json(config_path)
    _require(config.get("schema_version") == FORMAL_SFT_RUN_CONFIG_SCHEMA, "formal SFT run-config schema drift")
    _require(Path(config.get("output_root", "")).resolve() == root, "formal SFT run-config root drift")
    _require(config.get("microbatch_size") == 8, "formal SFT run-config microbatch drift")
    _require(config.get("effective_batch_size") == SFT_EFFECTIVE_BATCH_SIZE, "formal SFT run-config effective-batch drift")
    _require(config.get("gradient_accumulation_steps") == 2, "formal SFT run-config accumulation drift")
    _require(payload.get("run_config_sha256") == sha256_file(config_path), "formal SFT run-config hash drift")
    for field in ("git_sha", "authorization_sha256", "readiness_sha256"):
        _require(payload.get(field) == config.get(field), f"formal SFT {field} drift")
    training = payload.get("training", {})
    report_path = root / training.get("report", "")
    _require(report_path.is_file() and sha256_file(report_path) == training.get("report_sha256"), "formal SFT training-report drift")
    report = _load_json(report_path)
    _require(report.get("complete") is True, "formal SFT training report is incomplete")
    _require(report.get("stop_reason") in {"maximum_epochs", "dev_plateau"}, "formal SFT report stopping-rule drift")
    _require(report.get("microbatch_size") == 8 and report.get("effective_batch_size") == SFT_EFFECTIVE_BATCH_SIZE, "formal SFT report batch drift")
    _require(report.get("gradient_accumulation") == 2, "formal SFT report accumulation drift")
    history = report.get("history", [])
    _require(SFT_STOPPING_RULE["minimum_epochs"] <= len(history) <= SFT_STOPPING_RULE["maximum_epochs"], "formal SFT report epoch drift")
    current_input = validate_sft_training_inputs(Path(config["data_dir"]), Path(config["base_model"]))
    _require(payload.get("input_binding") == current_input, "formal SFT current input binding drift")
    _require(payload.get("loaded_audit", {}).get("train") == report.get("train_audit"), "formal SFT train audit lineage drift")
    _require(payload.get("loaded_audit", {}).get("dev") == report.get("dev_audit"), "formal SFT dev audit lineage drift")
    for row in history:
        _require(row.get("train_seen_unique_samples") == report["train_audit"]["sample_count"], "formal SFT epoch roster drift")
        _require(row.get("train_completion_label_tokens") == report["train_audit"]["completion_label_tokens"], "formal SFT epoch label-token drift")
    _require(training.get("epochs_completed") == len(history), "formal SFT manifest/report epoch drift")
    _require(training.get("optimizer_updates") == report.get("optimizer_updates"), "formal SFT manifest/report optimizer drift")
    adapter = root / payload.get("canonical_adapter", {}).get("relative_path", "")
    view = root / payload.get("rollout_adapter", {}).get("relative_path", "")
    _require(directory_sha256(adapter) == payload["canonical_adapter"]["directory_sha256"], "formal SFT adapter hash drift")
    view_audit = validate_vllm_adapter_view(source_adapter=adapter, view_directory=view, base_model=Path(config["base_model"]))
    _require(view_audit["view_directory_sha256"] == payload["rollout_adapter"]["directory_sha256"], "formal SFT rollout-view hash drift")
    _require(view_audit["semantic_tensor_sha256"] == payload["rollout_adapter"]["semantic_tensor_sha256"], "formal SFT adapter semantic drift")
    attempt = root / "attempts" / f"attempt-{payload.get('attempt_index', -1):04d}"
    attempt_record = _load_json(attempt / "attempt.json")
    _require(attempt_record.get("complete") is True, "formal SFT attempt record is incomplete")
    _require(attempt_record.get("training_report_sha256") == sha256_file(report_path), "formal SFT attempt/report drift")
    return {"path": str(path), "sha256": sha256_file(path), "payload": payload}
