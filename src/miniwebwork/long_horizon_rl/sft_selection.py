"""Fail-closed loader for the frozen SFT GPU preflight selection."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from ..m4_long_horizon_protocol import (
    PROJECT_ROOT,
    SFT_EFFECTIVE_BATCH_SIZE,
    SFT_LORA_CONFIG,
    SFT_MICROBATCH_CANDIDATES,
    SFT_PREFLIGHT_MAXIMUM_OPTIMIZER_UPDATES,
    SFT_VRAM_HEADROOM_MINIMUM,
    STUDY_ID,
)
from .contracts import SHA256_PATTERN, sha256_file

SFT_SELECTION_SCHEMA = "m4_long_horizon_sft_preflight_selection_v1"
SFT_SELECTION_PATH = PROJECT_ROOT / "data" / "m4_long_horizon_sft_preflight_selection_v1.json"
GIT_OBJECT_ID_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, field: str) -> None:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"invalid {field}",
    )


def validate_sft_preflight_selection(payload: Mapping[str, Any]) -> None:
    """Validate evidence and the formal SFT microbatch choice without trusting it."""

    _require(payload.get("schema_version") == SFT_SELECTION_SCHEMA, "SFT selection schema drift")
    _require(payload.get("study_id") == STUDY_ID, "SFT selection study drift")
    _require(payload.get("stage") == "sft_gpu_preflight", "SFT selection stage drift")
    _require(payload.get("result") == "PASS", "SFT preflight did not pass")
    _require(payload.get("formal_training") is False, "preflight mislabeled as formal training")
    _require(payload.get("disposable_adapter") is True, "preflight adapter must remain disposable")

    slurm = payload.get("slurm", {})
    _require(slurm.get("job_id") == 1264, "unreviewed SFT preflight JobID")
    _require(slurm.get("state") == "COMPLETED", "SFT preflight Slurm state drift")
    _require(slurm.get("exit_code") == "0:0", "SFT preflight exit code drift")
    _require(slurm.get("gpus") == 1 and slurm.get("cpus") == 4, "SFT preflight resource drift")
    _require(slurm.get("memory_gb") == 32, "SFT preflight memory drift")
    _require(slurm.get("wall_time_limit") == "24:00:00", "SFT preflight wall-time drift")

    lineage = payload.get("lineage", {})
    _require(
        isinstance(lineage.get("git_sha"), str)
        and GIT_OBJECT_ID_PATTERN.fullmatch(lineage["git_sha"]) is not None,
        "invalid lineage.git_sha",
    )
    for field in (
        "sft_manifest_sha256",
        "sft_token_audit_sha256",
        "sft_train_sha256",
        "sft_valid_sha256",
    ):
        _require_sha256(lineage.get(field), f"lineage.{field}")
    _require(lineage.get("tracked_worktree_clean") is True, "SFT preflight worktree was not clean")
    _require(lineage.get("formal_outputs_created") is False, "SFT preflight created formal output")
    _require(
        lineage.get("artifact_root")
        == "outputs/m4_long_horizon_credit_v1/preflight/sft_1264",
        "SFT preflight artifact root drift",
    )
    _require(
        lineage.get("sft_corpus") == "data/sft/m4_long_horizon_verified_v2",
        "SFT corpus selection drift",
    )

    selection = payload.get("selection", {})
    _require(
        selection.get("candidates") == list(SFT_MICROBATCH_CANDIDATES),
        "SFT microbatch candidate drift",
    )
    _require(selection.get("selected_microbatch") == 8, "frozen SFT microbatch drift")
    _require(
        selection.get("effective_batch_size") == SFT_EFFECTIVE_BATCH_SIZE,
        "SFT effective batch drift",
    )
    _require(selection.get("gradient_accumulation_steps") == 2, "SFT accumulation drift")
    minimum_headroom = selection.get("minimum_vram_headroom_fraction")
    selected_headroom = selection.get("selected_reserved_vram_headroom_fraction")
    _require(
        math.isclose(minimum_headroom, SFT_VRAM_HEADROOM_MINIMUM, abs_tol=1e-12),
        "SFT headroom threshold drift",
    )
    _require(
        isinstance(selected_headroom, (int, float))
        and selected_headroom >= SFT_VRAM_HEADROOM_MINIMUM,
        "selected SFT microbatch lacks required VRAM headroom",
    )
    measurements = selection.get("measurements", [])
    _require(
        [item.get("microbatch") for item in measurements] == list(SFT_MICROBATCH_CANDIDATES),
        "SFT benchmark measurements are incomplete",
    )
    _require(
        all(
            isinstance(item.get("forward_tokens_per_second"), (int, float))
            and item["forward_tokens_per_second"] > 0
            and isinstance(item.get("reserved_vram_headroom_fraction"), (int, float))
            and item["reserved_vram_headroom_fraction"] >= SFT_VRAM_HEADROOM_MINIMUM
            for item in measurements
        ),
        "SFT benchmark contains an invalid candidate",
    )

    smoke = payload.get("training_smoke", {})
    _require(
        smoke.get("optimizer_updates") == SFT_PREFLIGHT_MAXIMUM_OPTIMIZER_UPDATES,
        "SFT preflight optimizer-update count drift",
    )
    _require(
        smoke.get("stop_reason") == "preflight_optimizer_update_cap",
        "SFT preflight stop reason drift",
    )
    _require(smoke.get("dev_unique_samples") == 846, "SFT preflight dev roster drift")
    _require(smoke.get("dev_completion_label_tokens") == 18162, "SFT dev token count drift")
    for metric in (
        "train_nll",
        "dev_nll",
        "dev_teacher_forced_action_exact",
        "dev_teacher_forced_schema_valid",
    ):
        value = smoke.get(metric)
        _require(isinstance(value, (int, float)) and math.isfinite(value), f"invalid {metric}")

    telemetry = payload.get("telemetry", {})
    _require(telemetry.get("interval_seconds") == 5, "SFT telemetry interval drift")
    _require(telemetry.get("sample_count") == 497, "SFT telemetry sample-count drift")
    _require(
        telemetry.get("external_vram_headroom_fraction", -1) >= SFT_VRAM_HEADROOM_MINIMUM,
        "external telemetry violates SFT VRAM headroom gate",
    )

    adapter = payload.get("adapter_audit", {})
    _require_sha256(adapter.get("directory_sha256"), "adapter_audit.directory_sha256")
    _require(adapter.get("checkpoint_matches_final") is True, "SFT checkpoint/final adapter drift")
    _require(adapter.get("all_finite") is True, "SFT adapter has non-finite tensors")
    _require(
        adapter.get("tensor_count") == adapter.get("nonzero_tensor_count") == 256,
        "SFT adapter zero-tensor audit failed",
    )
    _require(adapter.get("lora") == SFT_LORA_CONFIG, "SFT adapter LoRA drift")

    artifact_hashes = payload.get("artifact_sha256", {})
    _require(
        set(artifact_hashes)
        == {
            "invocation",
            "benchmark",
            "training_report",
            "preflight_report",
            "stdout",
            "stderr",
            "gpu_telemetry",
            "training_progress",
        },
        "SFT preflight artifact hash set drift",
    )
    for field, value in artifact_hashes.items():
        _require_sha256(value, f"artifact_sha256.{field}")


def load_sft_preflight_selection(path: Path = SFT_SELECTION_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_sft_preflight_selection(payload)
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}
