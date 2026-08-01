import importlib.util
from pathlib import Path

import pytest

from miniwebwork.m4_protocol import (
    M4RunConfig,
    ONLINE_PASS_ACTION_TOKEN_CAP,
    build_m4_run_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "data" / "tasks" / "m4_rlvr_v1"
SEED_DIR = ROOT / "data" / "seed_m4_rlvr_v1"


def _load_module():
    path = ROOT / "scripts" / "m4_apply_online_update.py"
    spec = importlib.util.spec_from_file_location("m4_apply_online_update", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(task_hash: str, git_sha: str) -> dict:
    return {
        "schema_version": "3.3",
        "complete": True,
        "study_id": "m4_rlvr_v1",
        "git_sha": git_sha,
        "split": "train",
        "study_seed": 20260801,
        "collection_pass_index": 1,
        "K": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "generation_runtime": {"use_cache": False, "strict_on_policy": True},
        "adapter_sha256": "adapter",
        "task_source_sha256": task_hash,
        "max_collected_action_tokens": ONLINE_PASS_ACTION_TOKEN_CAP,
        "collected_action_tokens": 0,
        "completed_task_count": 0,
        "groups": [],
    }


def test_online_update_runner_accepts_only_matching_strict_m4_train_artifact():
    module = _load_module()
    config = M4RunConfig("grpo", 20260801, "train")
    manifest = build_m4_run_manifest(config, task_root=TASK_ROOT, seed_dir=SEED_DIR)
    artifact = _artifact(manifest["hashes"]["task_source_sha256"], manifest["git_sha"])

    module._validate_artifact(
        artifact,
        config=config,
        pass_index=1,
        adapter_hash="adapter",
        run_manifest=manifest,
    )

    artifact["task_source_sha256"] = manifest["hashes"]["train_public.jsonl"]
    with pytest.raises(ValueError, match="task source hash"):
        module._validate_artifact(
            artifact,
            config=config,
            pass_index=1,
            adapter_hash="adapter",
            run_manifest=manifest,
        )

    artifact["split"] = "test"
    with pytest.raises(PermissionError, match="train artifacts"):
        module._validate_artifact(
            artifact,
            config=config,
            pass_index=1,
            adapter_hash="adapter",
            run_manifest=manifest,
        )


def test_online_update_runner_rejects_token_cap_overrun_before_gpu_work():
    module = _load_module()
    config = M4RunConfig("gspo", 20260801, "train")
    manifest = build_m4_run_manifest(config, task_root=TASK_ROOT, seed_dir=SEED_DIR)
    artifact = _artifact(manifest["hashes"]["task_source_sha256"], manifest["git_sha"])
    artifact["collected_action_tokens"] = ONLINE_PASS_ACTION_TOKEN_CAP + 1

    with pytest.raises(ValueError, match="exceed"):
        module._validate_artifact(
            artifact,
            config=config,
            pass_index=1,
            adapter_hash="adapter",
            run_manifest=manifest,
        )
