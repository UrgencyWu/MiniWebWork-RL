from __future__ import annotations

import json

import pytest

from miniwebwork.m6_phase10c_specialist_data import (
    SPECIALIST_FAMILIES,
    action_family,
    build_specialist_smoke_corpus,
    public_instruction_query,
    public_instruction_query_candidates,
)


def test_public_query_contains_only_instruction_tokens():
    instruction = "Find a blue 12 inch desk lamp under $40, please."
    query = public_instruction_query(instruction)
    assert query == "Find a blue 12 inch desk lamp under 40 please"
    assert "B000" not in query


def test_public_query_removes_request_boilerplate_and_price_clause():
    query = public_instruction_query(
        "i am looking for a high performance dslr camera lenses that are certified refurbished, "
        "and price lower than 230.00 dollars"
    )
    assert query == "a high performance dslr camera lenses that are certified refurbished"


def test_public_query_preserves_decimal_product_attributes():
    query = public_instruction_query(
        "looking for a honiway decorative wall mirror 12.3 inch rustic wood frame for living room. keep in touch"
    )
    assert query == "a honiway decorative wall mirror 12 3 inch rustic wood frame for living room"


def test_public_query_candidates_are_fixed_public_compressions():
    candidates = public_instruction_query_candidates(
        "i am looking for a high quality butterfly hair clip for women, and price lower than 40 dollars"
    )
    assert candidates == (
        "a high quality butterfly hair clip for women",
        "butterfly hair clip women",
    )


@pytest.mark.parametrize(
    ("command", "family"),
    [
        ("search[blue lamp]", "search"),
        ("click[Next >]", "navigation"),
        ("click[B012345678]", "candidate"),
        ("click[Blue]", "option"),
        ("click[Buy Now]", "buy"),
    ],
)
def test_action_family_is_public_and_deterministic(command: str, family: str):
    assert action_family(command) == family


def test_specialist_masks_are_narrow_but_cover_full_purchase_chain():
    assert SPECIALIST_FAMILIES["S_nav_sft"] == {"search", "navigation", "candidate"}
    assert SPECIALIST_FAMILIES["S_match_sft"] == {"candidate", "option"}
    assert SPECIALIST_FAMILIES["S_finish_sft"] == {"option", "buy"}
    assert set().union(*SPECIALIST_FAMILIES.values()) == {"search", "navigation", "candidate", "option", "buy"}


def test_public_query_rejects_empty_instruction():
    with pytest.raises(ValueError, match="no public query tokens"):
        public_instruction_query("[] !!!")


def test_failed_smoke_returns_a_persistent_stop_report(monkeypatch: pytest.MonkeyPatch):
    import miniwebwork.m6_phase10c_specialist_data as module

    class EmptyEnvironment:
        def close(self):
            pass

    monkeypatch.setattr(
        module,
        "build_verified_public_query_trajectory",
        lambda environment, goal, **_kwargs: (_ for _ in ()).throw(module.SpecialistDataFailure("not_public")),
    )
    tasks = [f"webshop_goal_{index:05d}" for index in range(16)]
    goals = [{"goal_index": index} for index in range(16)]
    report = build_specialist_smoke_corpus(
        specialist="S_nav_sft",
        goals=goals,
        task_ids=tasks,
        environment_factory=EmptyEnvironment,
        phase10c_split_content_sha256="a" * 64,
        producer_git_sha="b" * 40,
    )
    assert report["passed"] is False
    assert report["decision"] == "stop_specialist_data_method"
    assert report["verified_task_count"] == 0
    assert report["rejection_counts"] == {"not_public": 16}


def test_qwen35_query_parser_accepts_json_and_rejects_unseen_asin():
    from scripts.m6_phase10c_qwen35_query_smoke import parse_query_completion

    assert parse_query_completion('{"query":"butterfly hair clip women"}', instruction="find a butterfly clip") \
        == "butterfly hair clip women"
    with pytest.raises(ValueError, match="unseen ASIN"):
        parse_query_completion('{"query":"B012345678"}', instruction="find a butterfly clip")


def test_external_query_missing_task_is_persisted_as_a_data_rejection(monkeypatch: pytest.MonkeyPatch):
    import miniwebwork.m6_phase10c_specialist_data as module

    monkeypatch.setattr(module, "build_verified_public_query_trajectory", lambda *_args, **_kwargs: None)
    tasks = [f"webshop_goal_{index:05d}" for index in range(16)]
    goals = [{"goal_index": index} for index in range(16)]
    report = module.build_specialist_smoke_corpus(
        specialist="S_nav_sft",
        goals=goals,
        task_ids=tasks,
        environment_factory=lambda: None,
        phase10c_split_content_sha256="split",
        producer_git_sha="0" * 40,
        query_candidates_by_task={},
    )
    assert report["verified_task_count"] == 0
    assert report["rejection_counts"] == {"no_valid_external_query_candidates": 16}
    assert report["query_tokens_from_public_instruction_only"] is False
    assert report["query_generator_input_is_public_instruction_only"] is True


def test_qwen35_query_wrapper_is_query_only_and_two_gpu():
    from pathlib import Path

    wrapper = (Path(__file__).resolve().parents[1] / "scripts/run_m6_phase10c_qwen35_query_smoke_job.sh").read_text()
    assert "#SBATCH --gres=gpu:2" in wrapper
    assert "m6_phase10c_qwen35_query_smoke.py" in wrapper
    assert "optimizer" not in wrapper
