import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m2_3_mini_single_probe.py"
    spec = importlib.util.spec_from_file_location("m2_3_mini_single_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_labels_keep_adapter_identity_explicit():
    probe = _load_probe_module()

    assert probe._resolve_policy_label("A", None) == "A_M2.2R"
    assert probe._resolve_policy_label("B", None) == "B_M2.3-mini"
    assert probe._resolve_policy_label("custom", "M3_one_batch") == "M3_one_batch"


@pytest.mark.parametrize("label", [None, "", "   "])
def test_custom_policy_requires_nonblank_label(label):
    probe = _load_probe_module()

    with pytest.raises(ValueError):
        probe._resolve_policy_label("custom", label)


def test_collector_exposes_versioned_seed_and_explicit_rollout_limits():
    probe = _load_probe_module()
    parameters = inspect.signature(probe.run_rollout).parameters

    assert {"seed_dir", "max_model_turns", "max_environment_steps", "max_output_failures"} <= set(parameters)


def test_collector_action_token_budget_reserves_whole_rollout_groups():
    probe = _load_probe_module()

    assert probe._can_start_complete_task_group(
        100,
        1_200,
        trajectories=4,
        max_model_turns=2,
        max_new_tokens=128,
    )
    assert not probe._can_start_complete_task_group(
        200,
        1_200,
        trajectories=4,
        max_model_turns=2,
        max_new_tokens=128,
    )


def test_probe_main_passes_only_three_sampling_parameters_to_strict_distribution(
    monkeypatch, tmp_path: Path
):
    probe = _load_probe_module()
    seen: list[tuple[float, float, int]] = []

    class StopBeforeGpu(Exception):
        pass

    def strict_spy(temperature: float, top_p: float, top_k: int) -> bool:
        seen.append((temperature, top_p, top_k))
        return True

    def stop_load_policy(*args, **kwargs):
        raise StopBeforeGpu

    adapter = tmp_path / "adapter"
    seed_dir = tmp_path / "seed"
    adapter.mkdir()
    seed_dir.mkdir()

    monkeypatch.setattr(probe, "strict_raw_policy_distribution", strict_spy)
    monkeypatch.setattr(
        probe, "_load_tasks", lambda *args, **kwargs: ([{"task_id": "M4-TEST"}], "task-hash")
    )
    monkeypatch.setattr(probe, "_directory_sha256", lambda *args, **kwargs: "adapter-hash")
    monkeypatch.setattr(probe, "_file_sha256", lambda *args, **kwargs: "prompt-hash")
    monkeypatch.setattr(probe, "_git_sha", lambda: "test-sha")
    monkeypatch.setattr(probe, "Heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(probe, "load_policy", stop_load_policy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "m2_3_mini_single_probe.py",
            "--policy",
            "custom",
            "--policy-label",
            "test",
            "--adapter",
            str(adapter),
            "--task-dir",
            str(tmp_path / "tasks"),
            "--seed-dir",
            str(seed_dir),
            "--temperature",
            "1.0",
            "--top-p",
            "1.0",
            "--top-k",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(StopBeforeGpu):
        probe.main()

    assert seen == [(1.0, 1.0, 0)]
