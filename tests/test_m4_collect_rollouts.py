import importlib.util
from pathlib import Path

from miniwebwork.m4_protocol import M4RunConfig


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
    assert command[command.index("--K") + 1] == "4"
    assert command[command.index("--study-seed") + 1] == "20260801"
    assert command[command.index("--collection-pass-index") + 1] == "1"
    assert command[command.index("--temperature") + 1] == "1.0"
    assert command[command.index("--top-p") + 1] == "1.0"
    assert command[command.index("--top-k") + 1] == "0"
    assert command[command.index("--study-id") + 1] == "m4_rlvr_v1"
