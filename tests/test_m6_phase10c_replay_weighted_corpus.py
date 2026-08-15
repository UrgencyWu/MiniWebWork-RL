from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "m6_phase10c_replay_weighted_teacher_corpus",
        ROOT / "scripts/m6_phase10c_replay_weighted_teacher_corpus.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _observation(*, terminal: bool) -> dict:
    return {
        "schema_version": "miniwebwork.webshop.observation.v1",
        "task_id": "webshop_goal_00001",
        "instruction": "buy the blue product",
        "page_type": "done" if terminal else "item",
        "visible_text": "done" if terminal else "blue product",
        "available_actions": ["click[Buy Now]"] if not terminal else ["reset"],
        "terminal": terminal,
        "text_truncated": False,
    }


def test_fresh_replay_requires_exact_public_state_and_strict_result():
    module = _module()
    before = _observation(terminal=False)
    after = _observation(terminal=True)

    class Observation:
        def __init__(self, value):
            self.value = value

        def to_dict(self):
            return dict(self.value)

    class Environment:
        def __init__(self, **_kwargs):
            self.closed = False

        def reset(self, _task_id):
            return Observation(before)

        def step(self, _command):
            return SimpleNamespace(
                observation=Observation(after),
                info={"action_result": {"success": True}, "task_score": 1.0},
                terminated=True,
                truncated=False,
            )

        def close(self):
            self.closed = True

    episode = {
        "task_id": "webshop_goal_00001",
        "turns": [{
            "observation": before,
            "action": {"command": "click[Buy Now]"},
            "action_result": {"success": True},
            "post_action_observation": after,
            "schema_valid": True,
        }],
    }
    replay = module.fresh_replay_episode(
        episode,
        base_url="http://example.invalid",
        environment_factory=Environment,
    )
    assert replay["strict_success"] is True
    assert replay["public_state_exact"] is True
    assert replay["action_result_exact"] is True
    assert replay["executed_command_count"] == replay["expected_command_count"] == 1


def test_weighted_rows_equalize_capability_task_path_and_action_rows():
    module = _module()
    task_capabilities = {
        "nav_a": "nav",
        "match_a": "match",
        "finish_a": "finish",
    }
    rows = [
        {"task_id": "nav_a", "trajectory_id": "n1", "completion_label_tokens": 2},
        {"task_id": "nav_a", "trajectory_id": "n1", "completion_label_tokens": 3},
        {"task_id": "match_a", "trajectory_id": "m1", "completion_label_tokens": 2},
        {"task_id": "finish_a", "trajectory_id": "f1", "completion_label_tokens": 2},
        {"task_id": "finish_a", "trajectory_id": "f2", "completion_label_tokens": 2},
    ]
    weighted = module.build_weighted_rows(rows, task_capabilities=task_capabilities)
    assert math.isclose(sum(row["row_loss_weight"] for row in weighted), 1.0, abs_tol=1e-12)
    masses = {
        capability: sum(row["row_loss_weight"] for row in weighted if row["capability"] == capability)
        for capability in module.CAPABILITY_WEIGHTS
    }
    assert masses == module.CAPABILITY_WEIGHTS
    nav = [row["row_loss_weight"] for row in weighted if row["capability"] == "nav"]
    assert nav == [0.10, 0.10]
    finish_path_mass = {
        trajectory_id: sum(
            row["row_loss_weight"]
            for row in weighted
            if row["trajectory_id"] == trajectory_id
        )
        for trajectory_id in ("f1", "f2")
    }
    assert finish_path_mass == {"f1": 0.20, "f2": 0.20}


def test_task_split_is_deterministic_stratified_and_disjoint():
    module = _module()
    capabilities = {
        **{f"nav_{index}": "nav" for index in range(10)},
        **{f"match_{index}": "match" for index in range(10)},
        **{f"finish_{index}": "finish" for index in range(10)},
    }
    train_a, dev_a = module.stratified_task_split(capabilities)
    train_b, dev_b = module.stratified_task_split(capabilities)
    assert (train_a, dev_a) == (train_b, dev_b)
    assert not (train_a & dev_a)
    assert len(dev_a) == 3
    assert {capabilities[task] for task in dev_a} == {"nav", "match", "finish"}


def test_replay_wrapper_is_cpu_only_and_freezes_inputs():
    wrapper = (ROOT / "scripts/run_m6_phase10c_replay_weighted_teacher_corpus_job.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --cpus-per-task=8" in wrapper
    assert "#SBATCH --gres" not in wrapper
    assert "--workers 8" in wrapper
    assert wrapper.count("--collection-root") == 2
    assert "optimizer" not in wrapper
