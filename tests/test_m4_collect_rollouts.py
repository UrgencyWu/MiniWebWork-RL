import importlib.util
import sys
from pathlib import Path

from miniwebwork.m4_protocol import M4RunConfig, RSFT_TRAIN_TASKS_PER_PASS


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m4_collect_rollouts.py"
    spec = importlib.util.spec_from_file_location("m4_collect_rollouts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m4_collector_command_uses_matching_world_and_strict_distribution(tmp_path: Path):
    collector = _load_module()
    config = M4RunConfig("gspo", 20260801, "train")
    command = collector._collector_command(
        config,
        adapter=tmp_path / "adapter",
        output_dir=tmp_path / "out",
        task_root=tmp_path / "tasks",
        seed_dir=tmp_path / "seed",
        max_tasks=3,
        train_pass_index=1,
    )

    assert command[command.index("--task-dir") + 1].endswith("tasks/train")
    assert command[command.index("--seed-dir") + 1].endswith("seed")
    assert command[command.index("--split") + 1] == "train"
    assert command[command.index("--task-order-seed") + 1] == "20260801"
    assert command[command.index("--K") + 1] == "4"
    assert command[command.index("--study-seed") + 1] == "20260801"
    assert command[command.index("--collection-pass-index") + 1] == "1"
    assert command[command.index("--max-collected-action-tokens") + 1] == "125000"
    assert command[command.index("--temperature") + 1] == "1.0"
    assert command[command.index("--top-p") + 1] == "1.0"
    assert command[command.index("--top-k") + 1] == "0"
    assert command[command.index("--study-id") + 1] == "m4_rlvr_v1"


def test_m4_collector_main_accepts_sft_for_final_test(monkeypatch, tmp_path: Path):
    collector = _load_module()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        collector,
        "build_m4_run_manifest",
        lambda *args, **kwargs: {"schema_version": "test"},
    )
    monkeypatch.setattr(
        collector,
        "write_m4_run_manifest",
        lambda path, manifest: captured.update(manifest_path=path, manifest=manifest),
    )
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda command, check: captured.update(command=command, check=check),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "m4_collect_rollouts.py",
            "--algorithm",
            "sft",
            "--seed",
            "20260801",
            "--phase",
            "final_test",
            "--adapter",
            str(adapter),
            "--output-dir",
            str(tmp_path / "out"),
            "--task-root",
            str(tmp_path / "tasks"),
            "--seed-dir",
            str(tmp_path / "seed"),
        ],
    )

    assert collector.main() == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--split") + 1] == "test"
    assert command[command.index("--K") + 1] == "4"
    assert "--collection-pass-index" not in command
    assert "--max-collected-action-tokens" not in command


def test_rsft_train_command_uses_fixed_12_task_roster_and_same_generation_cap(tmp_path: Path):
    collector = _load_module()
    command = collector._collector_command(
        M4RunConfig("rsft", 20260801, "train"),
        adapter=tmp_path / "adapter",
        output_dir=tmp_path / "out",
        task_root=tmp_path / "tasks",
        seed_dir=tmp_path / "seed",
        max_tasks=RSFT_TRAIN_TASKS_PER_PASS,
        train_pass_index=1,
    )

    assert command[command.index("--max-tasks") + 1] == "12"
    assert command[command.index("--max-collected-action-tokens") + 1] == "125000"
