import torch

from miniwebwork.m4_training import build_online_update_plan, select_rsft_rollouts
from miniwebwork.rollout import NO_FAILURE, POLICY_FAILURE, RolloutRecord, RolloutStep


def _record(task_id: str, index: int, success: bool, token_count: int = 2) -> RolloutRecord:
    step = RolloutStep(
        turn=1,
        page_type="products",
        prompt_token_ids=[1, 2],
        generated_token_ids=list(range(1, token_count + 1)),
        token_logprobs=[-0.2] * token_count,
        sampling_logprobs=[-0.2] * token_count,
        strict_json_success=True,
        schema_valid=True,
        parsed_action={"action": "click", "target": "x"},
        env_action_success=True,
    )
    return RolloutRecord(
        task_id=task_id,
        task_type="cheapest_feasible",
        episode_id=f"EP-{task_id}-{index}",
        rollout_index=index,
        rollout_seed=index,
        policy="policy",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        success=success,
        reward=1.0 if success else 0.0,
        rollout_valid=True,
        failure_origin=NO_FAILURE if success else POLICY_FAILURE,
        termination_reason="verified_submission" if success else "premature_finish",
        model_turns=1,
        environment_steps=1,
        schema_valid_count=1,
        schema_invalid_count=0,
        steps=[step],
    )


def test_online_update_plan_keeps_skip_reasons_and_enforces_token_budget():
    selected = [_record("A", 0, False), _record("A", 1, True)]
    no_signal = [_record("B", 0, False), _record("B", 1, False)]
    budgeted_out = [_record("C", 0, False), _record("C", 1, True)]
    plan = build_online_update_plan(
        selected + no_signal + budgeted_out,
        "gspo",
        action_token_budget=4,
    )

    assert plan.selected_groups == 1
    assert plan.selected_action_tokens == 4
    assert plan.skipped_groups == 2
    assert [group.reason for group in plan.groups] == [
        "selected",
        "ineligible:rollout group has no mixed-reward learning signal",
        "budget_exhausted",
    ]
    assert torch.allclose(plan.groups[0].replay_group.advantages_for("gspo"), torch.tensor([-1.0, 1.0]))


def test_rsft_selection_is_success_only_and_deterministic_by_action_cost():
    candidates = [
        _record("A", 0, True, token_count=4),
        _record("A", 1, True, token_count=2),
        _record("A", 2, False, token_count=1),
        _record("B", 0, False),
    ]
    selections = select_rsft_rollouts(candidates)

    assert selections[0]["selection"] == {
        "task_id": "A",
        "selected_episode_id": "EP-A-1",
        "selected_rollout_index": 1,
        "candidate_count": 3,
        "valid_success_count": 2,
        "reason": "selected_verified_best_of_n",
    }
    assert selections[1]["selection"]["reason"] == "no_verified_success"
    assert selections[1]["record"] is None
