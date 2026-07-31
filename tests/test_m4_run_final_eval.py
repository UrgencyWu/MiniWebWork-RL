import importlib.util
import json
from pathlib import Path

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
