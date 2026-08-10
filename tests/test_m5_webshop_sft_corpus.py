from __future__ import annotations

import pytest

from miniwebwork.long_horizon_rl.contracts import sha256_json
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
    extra["content_sha256"] = sha256_json({key: value for key, value in extra.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="keys drift"):
        _validate_record(
            extra,
            split="train",
            goal_index=1000,
            protocol_sha256="b" * 64,
            git_sha=git_sha,
            goals_sha256="c" * 64,
        )
