from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "m6_phase7_build_targeted_prescan_roster.py"
    spec = importlib.util.spec_from_file_location("m6_phase7_build_targeted_prescan_roster", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _trajectory(*, success: bool, score: float, asin: str):
    return {
        "success": success,
        "task_score": score,
        "termination_reason": "purchase",
        "turns": [
            {"action": {"command": f"click[{asin}]"}, "observation": {"visible_text": "public"}},
            {"action": {"command": "click[Buy Now]"}, "observation": {"visible_text": "product"}},
        ],
    }


def test_candidate_metrics_exclude_existing_same_item_tasks():
    module = _module()
    groups = [
        {
            "task_id": "same",
            "trajectories": [
                _trajectory(success=True, score=1.0, asin="B00000000A"),
                _trajectory(success=False, score=0.5, asin="B00000000A"),
            ],
        },
        {
            "task_id": "different",
            "trajectories": [
                _trajectory(success=True, score=1.0, asin="B00000000A"),
                _trajectory(success=False, score=0.5, asin="B00000000B"),
                _trajectory(success=False, score=0.2, asin="B00000000C"),
            ],
        },
    ]
    pair_counts, same_item = module._candidate_metrics(groups)
    assert pair_counts == {"same": 1, "different": 2}
    assert same_item == {"same"}


def test_selection_is_option_first_then_pair_count_and_deterministic(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "TARGET_TASKS", 4)
    pair_counts = {f"task_{index}": index + 1 for index in range(8)}
    goals = {
        task_id: {"goal_options": ["size"] if index < 5 else [], "category": "test"}
        for index, task_id in enumerate(pair_counts)
    }
    selected, candidates = module._select_candidates(pair_counts, {"task_0"}, goals)
    assert candidates == sorted(set(pair_counts) - {"task_0"})
    assert selected == ["task_4", "task_3", "task_2", "task_1"]
    assert module._select_candidates(pair_counts, {"task_0"}, goals)[0] == selected


def test_phase7_targeted_contract_is_inference_only_and_bounded():
    source = (ROOT / "scripts" / "m6_phase7_build_targeted_prescan_roster.py").read_text(encoding="utf-8")
    job = (ROOT / "scripts" / "run_m6_phase7_targeted_prescan_job.sh").read_text(encoding="utf-8")
    assert "TARGET_TASKS = 32" in source and "SELECTION_SEED = 20260832" in source
    assert "promotion" in source and "holdout" in source
    assert '"training_performed": False' in source and '"optimizer_steps": 0' in source
    assert 'trajectory["target_asin"]' not in source and 'goal["target_asin"]' not in source
    assert '"env_state"' not in source
    assert "#SBATCH --gres=gpu:1" in job
    assert "#SBATCH --cpus-per-task=4" in job and "#SBATCH --mem=24G" in job
    assert "#SBATCH --time=24:00:00" in job
    assert "--mode phase4_data_synthesis --role train --k 4 --seed 20260832" in job
    assert "--max-model-turns 18 --max-environment-steps 15" in job
    assert "--task-roster-producer-git-sha" in job
    assert "optimizer" not in job
