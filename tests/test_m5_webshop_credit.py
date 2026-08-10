from __future__ import annotations

import copy

import pytest

from miniwebwork.webshop_rl.credit import (
    ANCHOR_METHOD,
    BASELINE_METHOD,
    CREDIT_FORMULA_VERSION,
    assign_group_credit,
    policy_context_signature,
    public_state_anchor_signature,
)


def _observation(*, page: str, actions: list[str], step: int = 0, episode: str = "episode-a") -> dict:
    return {
        "schema_version": "m5-webshop-1.0",
        "task_id": "webshop_goal_01000",
        "episode_id": episode,
        "instruction": "buy a blue mug",
        "step_index": step,
        "page_type": page,
        "visible_text": f"public page: {page}",
        "text_truncated": False,
        "available_actions": actions,
        "terminal": page == "done",
        "target_asin": "must-never-affect-credit",
    }


def _trajectory(reward: float, anchors: list[str]) -> dict:
    return {
        "task_id": "webshop_goal_01000",
        "reward": reward,
        "turns": [
            {"public_state_anchor_sha256": anchor, "policy_context_token_sha256": "1" * 64}
            for anchor in anchors
        ],
    }


def test_credit_anchor_is_public_state_only_while_prompt_lineage_remains_distinct():
    left = _observation(page="search_results", actions=["click[B0001]"], step=2, episode="left")
    right = copy.deepcopy(left)
    right["episode_id"] = "right"
    right["step_index"] = 7
    right["target_asin"] = "another-hidden-target"
    assert public_state_anchor_signature(left) == public_state_anchor_signature(right)
    assert policy_context_signature([11, 12, 13]) != policy_context_signature([11, 99, 13])

    right["available_actions"].append("click[B0002]")
    assert public_state_anchor_signature(left) != public_state_anchor_signature(right)


def test_anchor_credit_survives_branch_convergence_and_marks_later_signal():
    initial = public_state_anchor_signature(_observation(page="home", actions=["search[<your query>]"]))
    converged = public_state_anchor_signature(
        _observation(page="search_results", actions=["click[B0001]"], step=2)
    )
    unique_a = public_state_anchor_signature(_observation(page="item", actions=["click[Buy Now]"], step=3))
    unique_b = public_state_anchor_signature(_observation(page="subpage", actions=["click[Back to Item]"], step=3))
    trajectories = [
        _trajectory(1.0, [initial, converged, unique_a]),
        _trajectory(0.0, [initial, converged, unique_b, unique_b]),
        _trajectory(1.0, [initial, unique_a]),
        _trajectory(0.0, [initial, unique_b]),
    ]

    report = assign_group_credit(trajectories, ANCHOR_METHOD)
    assert report["formula_version"] == CREDIT_FORMULA_VERSION
    assert report["metrics"]["shared_anchor_count"] >= 2
    assert report["metrics"]["non_initial_shared_anchor_count"] >= 1
    assert report["metrics"]["non_initial_informative_anchor_count"] >= 1
    assert report["turn_credit"][0][1]["micro_status"] == "informative"
    assert report["turn_credit"][1][1]["micro_status"] == "informative"
    # The repeated state in trajectory 1 is first-visit de-duplicated.
    assert report["turn_credit"][1][3]["micro_status"] == "no_shared_signal"


def test_baseline_is_exact_macro_broadcast_and_group_scope_fails_closed():
    initial = public_state_anchor_signature(_observation(page="home", actions=["search[<your query>]"]))
    trajectories = [_trajectory(float(index % 2), [initial, initial]) for index in range(4)]
    report = assign_group_credit(trajectories, BASELINE_METHOD)
    assert all(
        turn["micro_advantage"] == 0.0 and turn["turn_advantage"] == turn["macro_advantage"]
        for trajectory_credit in report["turn_credit"]
        for turn in trajectory_credit
    )

    changed = copy.deepcopy(trajectories)
    changed[-1]["task_id"] = "webshop_goal_01001"
    with pytest.raises(ValueError, match="crosses task"):
        assign_group_credit(changed, ANCHOR_METHOD)
