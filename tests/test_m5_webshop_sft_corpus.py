from __future__ import annotations

import json

import pytest

from miniwebwork.long_horizon_rl.contracts import sha256_json
from scripts import build_m5_webshop_sft_corpus as corpus_builder
from scripts.build_m5_webshop_sft_corpus import _validate_record


def test_oracle_record_self_hash_binds_protocol_and_git_lineage():
    git_sha = "a" * 40
    record = {
        "schema_version": "m5_webshop_oracle_record_v1",
        "split": "train",
        "goal_index": 1000,
        "task_id": "webshop_goal_01000",
        "protocol_sha256": "b" * 64,
        "git_sha": git_sha,
        "goals_sha256": "c" * 64,
        "status": "policy_excluded",
        "exclusion_reason": "diagnostic",
        "trajectory": None,
    }
    record["content_sha256"] = sha256_json(record)
    assert _validate_record(
        record,
        split="train",
        goal_index=1000,
        protocol_sha256="b" * 64,
        git_sha=git_sha,
        goals_sha256="c" * 64,
    ) == record
    with pytest.raises(ValueError, match="Git lineage"):
        _validate_record(
            record,
            split="train",
            goal_index=1000,
            protocol_sha256="b" * 64,
            git_sha="d" * 40,
            goals_sha256="c" * 64,
        )
    extra = dict(record, hidden_target="forbidden")
    extra["content_sha256"] = sha256_json(
        {key: value for key, value in extra.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="keys drift"):
        _validate_record(
            extra,
            split="train",
            goal_index=1000,
            protocol_sha256="b" * 64,
            git_sha=git_sha,
            goals_sha256="c" * 64,
        )


def test_failed_selection_writes_a_self_hashed_lineage_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_builder, "eligible_goal_indices", lambda split: (1000,))
    monkeypatch.setattr(
        corpus_builder,
        "deterministic_candidate_order",
        lambda *, candidates, seed, namespace: candidates,
    )
    monkeypatch.setattr(
        corpus_builder,
        "_build_record",
        lambda **kwargs: {
            "goal_index": 1000,
            "status": "policy_excluded",
            "exclusion_reason": "target_asin_not_in_public_top50",
        },
    )
    goals = [{} for _ in range(1001)]
    with pytest.raises(ValueError, match="insufficient verified train"):
        corpus_builder._collect_split(
            split="train",
            target_count=1,
            goals=goals,
            base_url="http://unused.test",
            output_root=tmp_path,
            protocol_sha256="b" * 64,
            git_sha="a" * 40,
            goals_sha256="c" * 64,
            seed=20260810,
            workers=1,
        )
    diagnostic = json.loads((tmp_path / "train_selection_diagnostic.json").read_text())
    content_sha256 = diagnostic.pop("content_sha256")
    assert content_sha256 == sha256_json(diagnostic)
    assert diagnostic["passed"] is False
    assert diagnostic["verified_count"] == 0
    assert diagnostic["git_sha"] == "a" * 40
