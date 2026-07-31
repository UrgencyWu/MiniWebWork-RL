import torch

from miniwebwork.rl.batch import build_replay_group
from miniwebwork.rl.m4_update import apply_m4_online_update, m4_group_policy_loss
from miniwebwork.rollout import NO_FAILURE, POLICY_FAILURE, RolloutRecord, RolloutStep


def _record(index: int, success: bool, token_counts=(2,)):
    steps = []
    for turn, token_count in enumerate(token_counts, start=1):
        steps.append(
            RolloutStep(
                turn=turn,
                page_type="products",
                prompt_token_ids=[10, index, turn],
                generated_token_ids=list(range(token_count)),
                token_logprobs=[-0.2] * token_count,
                sampling_logprobs=[-0.2] * token_count,
                strict_json_success=True,
                schema_valid=True,
                parsed_action={"action": "click", "target": "x"},
                env_action_success=True,
            )
        )
    return RolloutRecord(
        task_id="TASK",
        task_type="cheapest_feasible",
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


def test_m4_group_loss_supports_token_and_sequence_online_methods():
    group = build_replay_group([_record(0, False, (1, 2)), _record(1, True, (3,))])
    scale = torch.tensor(0.05, requires_grad=True)

    def current(turn):
        return torch.tensor(turn.old_policy_logprobs) + scale * len(turn.completion_token_ids)

    grpo = m4_group_policy_loss(group, "grpo", current)
    gspo = m4_group_policy_loss(group, "gspo", current)
    assert grpo.action_token_count == gspo.action_token_count == 6
    assert grpo.trajectory_count == gspo.trajectory_count == 2
    assert not torch.allclose(grpo.loss, gspo.loss)
    gspo.loss.backward()
    assert scale.grad is not None and torch.isfinite(scale.grad)


def test_m4_update_changes_trainable_parameter_from_strict_replay_group():
    group = build_replay_group([_record(0, False), _record(1, True)])
    parameter = torch.nn.Parameter(torch.tensor(0.03))

    def current(turn):
        return torch.tensor(turn.old_policy_logprobs) + parameter * (1 + turn.prompt_token_ids[1])

    result = apply_m4_online_update(
        [group], "rloo", [parameter], current, learning_rate=0.01
    )
    assert result["optimizer_updates"] == 1
    assert result["selected_groups"] == 1
    assert result["nonzero_gradient_parameter_tensors"] == 1
    assert result["changed_parameter_tensors"] == 1
