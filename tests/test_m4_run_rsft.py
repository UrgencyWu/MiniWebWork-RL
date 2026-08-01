import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rsft_runner_is_importable_without_starting_collection():
    path = ROOT / "scripts" / "m4_run_rsft.py"
    spec = importlib.util.spec_from_file_location("m4_run_rsft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._single_artifact


def test_rsft_runner_uses_the_shared_offline_final_adapter_layout(monkeypatch, tmp_path: Path):
    path = ROOT / "scripts" / "m4_run_rsft.py"
    spec = importlib.util.spec_from_file_location("m4_run_rsft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    initial_adapter = tmp_path / "initial_adapter"
    validation_data = tmp_path / "validation"
    output_dir = tmp_path / "run"
    initial_adapter.mkdir()
    validation_data.mkdir()
    (output_dir / "rsft_corpus").mkdir(parents=True)
    (output_dir / "rsft_corpus" / "manifest.json").write_text(
        json.dumps({"no_verified_successes": False})
    )
    commands: list[list[str]] = []

    monkeypatch.setattr(module.subprocess, "run", lambda command, check: commands.append(command))
    monkeypatch.setattr(
        module,
        "assert_m4_canonical_initial_adapter",
        lambda adapter, task_root: {
            "path": str(Path(adapter).resolve()),
            "sha256": "test-initial-hash",
            "study_manifest_sha256": "test-study-manifest",
        },
    )
    monkeypatch.setattr(
        module,
        "_single_artifact",
        lambda directory: directory / "collector" / "single_probe.json",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "m4_run_rsft.py",
            "--seed",
            "20260801",
            "--initial-adapter",
            str(initial_adapter),
            "--sft-validation-data-dir",
            str(validation_data),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert module.main() == 0
    collection_commands = commands[:2]
    assert all("--max-tasks" in command for command in collection_commands)
    assert {
        command[command.index("--max-tasks") + 1]
        for command in collection_commands
    } == {"12"}
    corpus_command = commands[-2]
    assert corpus_command[corpus_command.index("--task-root") + 1] == str(
        (ROOT / "data" / "tasks" / "m4_rlvr_v1").resolve()
    )
    offline_command = commands[-1]
    assert offline_command[offline_command.index("--output-dir") + 1] == str(output_dir.resolve())


def test_rsft_no_signal_materializes_the_unchanged_initial_adapter(tmp_path: Path):
    path = ROOT / "scripts" / "m4_run_rsft.py"
    spec = importlib.util.spec_from_file_location("m4_run_rsft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    initial_adapter = tmp_path / "initial"
    initial_adapter.mkdir()
    (initial_adapter / "adapter.bin").write_bytes(b"initial")
    output_dir = tmp_path / "run"
    task_root = ROOT / "data" / "tasks" / "m4_rlvr_v1"
    seed_dir = ROOT / "data" / "seed_m4_rlvr_v1"
    adapter = module._materialize_no_signal_adapter(
        config=module.M4RunConfig("rsft", 20260801, "train"),
        initial_adapter=initial_adapter,
        output_dir=output_dir,
        task_root=task_root,
        seed_dir=seed_dir,
        rsft_manifest={"selected_task_count": 0, "no_verified_successes": True},
        canonical_initial_adapter={
            "path": str(initial_adapter.resolve()),
            "sha256": module._directory_sha256(initial_adapter),
            "study_manifest_sha256": "test-study-manifest",
        },
    )

    assert module._directory_sha256(adapter) == module._directory_sha256(initial_adapter)
    assert json.loads((adapter.parent / "metrics.json").read_text())["no_signal"] is True
    assert json.loads((output_dir / "resolved_run_manifest.json").read_text())["no_signal"] is True
