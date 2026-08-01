import importlib.util
import json
from pathlib import Path

import pytest

from miniwebwork.m4_protocol import M4RunConfig


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "m4_run_final_eval.py"
    spec = importlib.util.spec_from_file_location("m4_run_final_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_eval_resolves_offline_adapter_only_with_completed_metrics(tmp_path: Path):
    module = _load_module()
    adapter = tmp_path / "sft" / "seed_20260801" / "training" / "seed_20260801" / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter.parent / "metrics.json").write_text(json.dumps({"seed": 20260801}))
    assert module._resolve_final_adapter("sft", 20260801, tmp_path) == adapter.resolve()


def test_final_eval_resolves_rsft_from_the_shared_offline_adapter_layout(tmp_path: Path):
    module = _load_module()
    adapter = tmp_path / "rsft" / "seed_20260801" / "training" / "seed_20260801" / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter.parent / "metrics.json").write_text(json.dumps({"seed": 20260801}))
    assert module._resolve_final_adapter("rsft", 20260801, tmp_path) == adapter.resolve()


def test_final_eval_resolves_second_online_pass_and_builds_final_test_command(tmp_path: Path):
    module = _load_module()
    first = tmp_path / "grpo" / "seed_20260801" / "pass_1" / "update" / "updated_adapter"
    second = tmp_path / "grpo" / "seed_20260801" / "pass_2" / "update" / "updated_adapter"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    summary = {
        "algorithm": "grpo",
        "seed": 20260801,
        "passes": [
            {"pass_index": 1, "next_adapter": str(first)},
            {"pass_index": 2, "next_adapter": str(second)},
        ],
    }
    summary_path = tmp_path / "grpo" / "seed_20260801" / "online_run_summary.json"
    summary_path.write_text(json.dumps(summary))
    adapter = module._resolve_final_adapter("grpo", 20260801, tmp_path)
    assert adapter == second.resolve()
    command = module._command(
        M4RunConfig("grpo", 20260801, "final_test"),
        adapter=adapter,
        output_dir=tmp_path / "eval",
        task_root=tmp_path / "tasks",
        seed_dir=tmp_path / "seed",
    )
    assert command[command.index("--phase") + 1] == "final_test"
    assert command[command.index("--adapter") + 1] == str(second.resolve())


def test_final_eval_strict_gate_requires_offline_canonical_initial_metadata(tmp_path: Path):
    module = _load_module()
    initial = tmp_path / "designated_initial"
    initial.mkdir()
    (initial / "adapter.bin").write_bytes(b"initial")
    canonical = {
        "path": str(initial.resolve()),
        "sha256": "initial-hash",
        "study_manifest_sha256": "study-manifest",
    }
    run_dir = tmp_path / "sft" / "seed_20260801"
    adapter = run_dir / "training" / "seed_20260801" / "final_adapter"
    adapter.mkdir(parents=True)
    metrics = {
        "seed": 20260801,
        "initial_adapter": str(initial),
        "initial_adapter_sha256": "initial-hash",
    }
    (adapter.parent / "metrics.json").write_text(json.dumps(metrics))
    run_manifest = {
        "initial_adapter": str(initial),
        "initial_adapter_sha256": "initial-hash",
        "canonical_initial_adapter": canonical,
        "run_manifest": {"frozen": "train"},
    }
    (run_dir / "resolved_run_manifest.json").write_text(json.dumps(run_manifest))
    assert module._resolve_final_adapter(
        "sft",
        20260801,
        tmp_path,
        canonical_initial_adapter=canonical,
        expected_train_manifest={"frozen": "train"},
    ) == adapter.resolve()

    run_manifest["run_manifest"] = {"wrong": "manifest"}
    (run_dir / "resolved_run_manifest.json").write_text(json.dumps(run_manifest))
    with pytest.raises(ValueError, match="frozen train manifest"):
        module._resolve_final_adapter(
            "sft",
            20260801,
            tmp_path,
            canonical_initial_adapter=canonical,
            expected_train_manifest={"frozen": "train"},
        )

    run_manifest["run_manifest"] = {"frozen": "train"}
    (run_dir / "resolved_run_manifest.json").write_text(json.dumps(run_manifest))
    metrics["initial_adapter_sha256"] = "wrong"
    (adapter.parent / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="initial adapter hash"):
        module._resolve_final_adapter(
            "sft",
            20260801,
            tmp_path,
            canonical_initial_adapter=canonical,
            expected_train_manifest={"frozen": "train"},
        )
