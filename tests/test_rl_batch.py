import pytest
import torch

from miniwebwork.rl.batch import (
    build_replay_group,
    old_policy_batch,
    pad_trajectory_logprobs,
)
from miniwebwork.rollout import NO_FAILURE, POLICY_FAILURE, RolloutRecord, RolloutStep


def _record(index: int, success: bool, token_counts=(2,)) -> RolloutRecord:
    steps = []
    for turn, token_count in enumerate(token_counts, start=1):
        ids = list(range(1, token_count + 1))
        steps.append(
            RolloutStep(
                turn=turn,
                page_type="products",
                prompt_token_ids=[10, 11, turn],
                generated_token_ids=ids,
                token_logprobs=[-0.1 * turn] * token_count,
                sampling_logprobs=[-0.1 * turn] * token_count,
                strict_json_success=True,
                schema_valid=True,
                parsed_action={"action": "click", "target": "x"},
                env_action_success=True,
            )
        )
    return RolloutRecord(
        task_id="TASK",
        task_type="no_feasible_product",
        episode_id=f"EP-{index}",
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
        model_turns=len(steps),
        environment_steps=len(steps),
        schema_valid_count=len(steps),
        schema_invalid_count=0,
        steps=steps,
    )


def test_build_replay_group_preserves_turn_boundaries_and_advantages():
    group = build_replay_group(
        [_record(0, False, (1, 2)), _record(1, True, (3,))]
    )

    assert group.task_id == "TASK"
    assert group.temperature == 1.0
    assert group.top_p == 1.0
    assert group.top_k == 0
    assert group.max_raw_sampling_logprob_abs_diff == 0.0
    assert group.logprob_match_tolerance == 5e-2
    assert len(group.trajectories) == 2
    assert len(group.trajectories[0].turns) == 2
    assert group.trajectories[0].action_token_count == 3
    assert group.trajectories[1].action_token_count == 3
    assert torch.allclose(group.advantages, torch.tensor([-1.0, 1.0]), atol=1e-6)

    old, mask = old_policy_batch(group)
    assert old.shape == (2, 3)
    assert mask.tolist() == [[True, True, True], [True, True, True]]


def test_warped_distribution_cannot_enter_update_batch():
    first = _record(0, False)
    second = _record(1, True)
    first.temperature = second.temperature = 0.4

    with pytest.raises(ValueError, match="temperature=1"):
        build_replay_group([first, second])


def test_logprob_mismatch_cannot_be_promoted_by_caller():
    failed = _record(0, False)
    failed.steps[0].sampling_logprobs = [-1.0] * len(
        failed.steps[0].generated_token_ids
    )

    with pytest.raises(ValueError, match="exceeds tolerance"):
        build_replay_group([failed, _record(1, True)])


def test_zero_variance_group_is_rejected():
    with pytest.raises(ValueError, match="no mixed-reward"):
        build_replay_group([_record(0, False), _record(1, False)])


def test_missing_prompt_evidence_is_rejected():
    failed = _record(0, False)
    failed.steps[0].prompt_token_ids = []

    with pytest.raises(ValueError, match="no prompt token evidence"):
        build_replay_group([failed, _record(1, True)])


def test_missing_sampling_evidence_is_rejected():
    failed = _record(0, False)
    failed.steps[0].sampling_logprobs = []

    with pytest.raises(ValueError, match="sampling log-probabilities are incomplete"):
        build_replay_group([failed, _record(1, True)])


def test_pad_trajectory_logprobs_preserves_lengths():
    padded, mask = pad_trajectory_logprobs(
        [torch.tensor([-0.1]), torch.tensor([-0.2, -0.3, -0.4])]
    )

    assert padded.shape == (2, 3)
    assert mask.tolist() == [[True, False, False], [True, True, True]]
    assert padded[0, 1:].tolist() == [0.0, 0.0]
