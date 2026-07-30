import pytest
import torch

from miniwebwork.model_agent.model_backend import (
    _sampling_logprobs,
    extract_generated_token_logprobs,
)


def test_generated_token_logprobs_align_with_last_prompt_position():
    # Prompt length 3, generated tokens [1, 2]. Positions 2 and 3 predict
    # the first and second generated token respectively.
    logits = torch.full((1, 5, 4), -10.0)
    logits[0, 2, 1] = 10.0
    logits[0, 3, 2] = 10.0

    values = extract_generated_token_logprobs(
        logits=logits,
        prompt_length=3,
        generated_ids=torch.tensor([1, 2]),
    )

    assert values.shape == (2,)
    assert torch.all(values > -1e-3)


def test_generated_token_logprobs_rejects_out_of_range_token_id():
    logits = torch.zeros((1, 3, 4))

    with pytest.raises(ValueError, match="vocabulary"):
        extract_generated_token_logprobs(
            logits,
            prompt_length=2,
            generated_ids=torch.tensor([4]),
        )


def test_generated_token_logprobs_rejects_insufficient_sequence_length():
    logits = torch.zeros((1, 2, 4))

    with pytest.raises(ValueError, match="Insufficient logits"):
        extract_generated_token_logprobs(
            logits,
            prompt_length=2,
            generated_ids=torch.tensor([1, 2]),
        )


def test_sampling_logprobs_align_generation_scores_with_emitted_tokens():
    scores = (
        torch.tensor([[0.0, 5.0, -float("inf")]]),
        torch.tensor([[4.0, 0.0, -float("inf")]]),
    )

    values = _sampling_logprobs(scores, torch.tensor([1, 0]))

    assert len(values) == 2
    assert values[0] > -0.02
    assert values[1] > -0.02


def test_sampling_logprobs_reject_score_length_mismatch():
    with pytest.raises(RuntimeError, match="score mismatch"):
        _sampling_logprobs(
            (torch.zeros((1, 3)),),
            torch.tensor([1, 2]),
        )
