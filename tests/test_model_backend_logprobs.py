import torch

from miniwebwork.model_agent.model_backend import extract_generated_token_logprobs


def test_generated_token_logprobs_align_with_last_prompt_position():
    # prompt length 3, generated tokens [1, 2].  Positions 2 and 3 predict
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
    generated_ids = torch.tensor([4])

    try:
        extract_generated_token_logprobs(logits, prompt_length=2, generated_ids=generated_ids)
    except ValueError as exc:
        assert "outside vocabulary" in str(exc) or "out of vocabulary" in str(exc)
    else:
        raise AssertionError("Expected out-of-range token id to fail")


def test_generated_token_logprobs_rejects_insufficient_sequence_length():
    logits = torch.zeros((1, 2, 4))
    generated_ids = torch.tensor([1, 2])

    try:
        extract_generated_token_logprobs(logits, prompt_length=2, generated_ids=generated_ids)
    except ValueError as exc:
        assert "Insufficient logits" in str(exc)
    else:
        raise AssertionError("Expected insufficient logits to fail")
