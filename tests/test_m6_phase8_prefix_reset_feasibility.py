from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "m6_phase8_prefix_reset_feasibility.py"
    spec = importlib.util.spec_from_file_location("m6_phase8_prefix_reset_feasibility", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _turn(command: str, *, success: bool, observation: dict | None = None):
    return {
        "action": {"command": command},
        "action_result": {"success": success},
        "observation": observation or {"page_type": "search", "available_actions": ["search[<your query>]"]},
    }


def _strict_trajectory(*, prefix_success: bool = True):
    product = {
        "page_type": "item",
        "available_actions": [
            "search[<your query>]",
            "click[Description]",
            "click[red]",
            "click[blue]",
            "click[green]",
            "click[Buy Now]",
            "click[Back to Search]",
        ],
    }
    return {
        "success": True,
        "termination_reason": "purchase",
        "turns": [
            _turn("search[shirt]", success=prefix_success),
            _turn("click[B00000000A]", success=True),
            _turn("click[red]", success=True, observation=product),
            _turn("click[Buy Now]", success=True, observation=product),
        ],
    }


def test_public_prefix_ends_at_asin_and_detects_option_branches():
    module = _module()
    result = module._strict_prefix_candidate(_strict_trajectory())
    assert result is not None
    assert result["prefix_action_count"] == 2
    assert result["public_option_action_count"] == 3


def test_failed_pre_asin_action_is_not_replayable():
    module = _module()
    assert module._strict_prefix_candidate(_strict_trajectory(prefix_success=False)) is None


def test_navigation_and_asins_are_not_counted_as_option_actions():
    module = _module()
    actions = [
        "click[Description]",
        "click[Features]",
        "click[Reviews]",
        "click[Back to Search]",
        "click[Buy Now]",
        "click[B00000000A]",
        "click[2]",
        "click[red]",
        "click[blue]",
    ]
    assert module._option_actions(actions) == ("click[blue]", "click[red]")


def test_phase8_contract_is_cpu_only_and_fail_closed():
    module = _module()
    assert module.MINIMUM_REPLAYABLE_TASKS == 20
    assert module.MINIMUM_BRANCHABLE_TASKS == 12
    source = (ROOT / "scripts" / "m6_phase8_prefix_reset_feasibility.py").read_text(encoding="utf-8")
    assert '"training_performed": False' in source and '"optimizer_steps": 0' in source
    assert '"target_asin_or_hidden_answer_used": False' in source
    assert '"post_action_internal_state_used": False' in source
    assert "promotion" in source and "holdout" in source
    assert "optimizer.step(" not in source
