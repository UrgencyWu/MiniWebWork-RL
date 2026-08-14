from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "m6_phase7_terminal_contrast_diagnostic.py"
    spec = importlib.util.spec_from_file_location("m6_phase7_terminal_contrast_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _trajectory(*, success: bool, score: float, asin: str, options: dict[str, str]):
    option_text = "\n".join(f"- {name} (selected: {value}):" for name, value in options.items())
    return {
        "success": success,
        "task_score": score,
        "termination_reason": "purchase",
        "turns": [
            {
                "action": {"command": f"click[{asin}]"},
                "observation": {"visible_text": "public search result"},
            },
            {
                "action": {"command": "click[Buy Now]"},
                "observation": {
                    "visible_text": f"Product page\n{option_text}",
                    "info": {"target_asin": "MUST-NOT-BE-USED"},
                },
                "post_action_observation": {"env_state": {"page_type": "done", "target_asin": "HIDDEN"}},
            },
        ],
    }


def _group(task_id: str, trajectories):
    return {"task_id": task_id, "trajectories": list(trajectories)}


def _goal(index: int):
    return {"goal_index": index, "category": "mugs", "instruction_attributes": ["blue"], "goal_options": ["navy"]}


def test_phase7_detects_same_item_option_contrast_without_hidden_answer():
    module = _module()
    task = "webshop_goal_01000"
    group = _group(task, [
        _trajectory(success=True, score=1.0, asin="B00000000A", options={"color": "Navy Blue"}),
        _trajectory(success=False, score=0.5, asin="B00000000A", options={"color": "Red"}),
        _trajectory(success=False, score=0.3, asin="B00000000B", options={"color": "Navy Blue"}),
        {"success": False, "task_score": 0.0, "termination_reason": "max_model_turns", "turns": []},
    ])
    result = module._analyze_groups([group], goal_map={task: _goal(1000)})
    assert result["mixed_task_count"] == 1
    assert result["strict_partial_pair_count"] == 2
    assert result["same_item_strict_partial_pair_count"] == 1
    assert result["option_contrast_pair_count"] == 1
    assert result["same_item_task_ids"] == [task]
    assert result["failure_counts"] == {
        "horizon_exhaustion": 1,
        "partial_purchase": 2,
        "strict_success": 1,
    }


def test_phase7_requires_public_nonempty_item_and_normalizes_options():
    module = _module()
    visible = _trajectory(success=True, score=1.0, asin="B00000000A", options={" Color ": " Navy   Blue "})
    choice = module._prebuy_public_choice(visible)
    assert choice == {"asin": "B00000000A", "selected_options": (("color", "navy blue"),)}
    visible["turns"][0]["action"] = {"command": "click[Description]"}
    visible["turns"][1]["observation"]["info"]["target_asin"] = "B000HIDDEN"
    assert module._prebuy_public_choice(visible) is None


def test_phase7_frozen_thresholds_are_not_result_dependent():
    module = _module()
    assert module.ONLINE_SAME_ITEM_MIXED_SHARE == 0.50
    assert module.ONLINE_OPTION_STRONG_SHARE == 0.50
    assert module.MINIMUM_REBUILD_SAME_ITEM_TASKS == 20
    assert module.MINIMUM_REBUILD_OPTION_TASKS == 12
    source = (ROOT / "scripts" / "m6_phase7_terminal_contrast_diagnostic.py").read_text(encoding="utf-8")
    assert "target_asin_or_hidden_answer_used\": False" in source
    assert "optimizer_steps\": 0" in source
    assert "promotion" in source and "holdout" in source
