import hashlib
from pathlib import Path

import pytest

from miniwebwork.m4_long_horizon_protocol import (
    FORMAL_METHODS,
    ONLINE_METHODS,
    SPLIT_COUNTS,
    assert_dataset_binding,
    assert_formal_submission_closed,
    assert_preflight_output,
    audit_horizon_dataset,
    classify_horizon,
    deterministic_task_order,
    load_study_manifest,
    validate_study_manifest,
)


def test_focused_manifest_is_preflight_only_and_excludes_algorithm_zoo():
    loaded = load_study_manifest()
    payload = loaded["payload"]
    assert payload["formal_submission_allowed"] is False
    assert FORMAL_METHODS == ("verified_sft", "multi_turn_grpo", "step_aware_gpo")
    assert ONLINE_METHODS == ("multi_turn_grpo", "step_aware_gpo")
    assert set(payload["formal_matrix"]["excluded_formal_algorithms"]) == {"rsft", "rloo", "gspo"}
    assert_formal_submission_closed(payload)


def test_study_manifest_binds_exact_checked_dataset_and_seed():
    binding = assert_dataset_binding()
    assert binding["dataset_manifest_sha256"] == (
        "ade9302269232a44bf92d28e8d9a357ef8e377509df09e69fdcf3c7cfe46fa5f"
    )
    assert binding["seed_manifest_sha256"] == (
        "d5742a3af8588c235a1d81f623ad14b5f0d793d9c99ad2b8580becf2c52383fd"
    )


def test_manifest_rejects_early_formal_opening():
    payload = load_study_manifest()["payload"]
    payload["formal_submission_allowed"] = True
    with pytest.raises(ValueError, match="opened early"):
        validate_study_manifest(payload)


def test_preflight_output_is_physically_isolated(tmp_path: Path):
    payload = load_study_manifest()["payload"]
    accepted = assert_preflight_output(
        Path(payload["output_contract"]["preflight_root"]) / "rollout-smoke",
        payload,
    )
    assert accepted.name == "rollout-smoke"
    with pytest.raises(ValueError, match="not a preflight output"):
        assert_preflight_output(Path(payload["output_contract"]["formal_root"]) / "seed-1", payload)
    with pytest.raises(ValueError, match="not a preflight output"):
        assert_preflight_output(tmp_path, payload)


@pytest.mark.parametrize(
    ("actions", "expected"),
    [(6, "basic"), (8, "basic"), (9, "medium"), (12, "medium"), (13, "long"), (20, "long")],
)
def test_horizon_boundaries(actions: int, expected: str):
    assert classify_horizon(actions) == expected


@pytest.mark.parametrize("actions", [True, 5, 21])
def test_horizon_rejects_values_outside_frozen_contract(actions):
    with pytest.raises(ValueError):
        classify_horizon(actions)


def _dataset():
    rows_by_split = {}
    for split, count in SPLIT_COUNTS.items():
        rows = []
        for index in range(count):
            if index % 6 < 2:
                horizon = 7
                stratum = "basic"
            elif index % 6 < 4:
                horizon = 10
                stratum = "medium"
            else:
                horizon = 15
                stratum = "long"
            identity = f"{split}-{index:04d}"
            rows.append({
                "task_id": f"TASK-{identity}",
                "oracle_min_env_actions": horizon,
                "horizon_stratum": stratum,
                "oracle_trace_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                "world_signature": f"world-{identity}",
                "product_signature": f"product-{identity}",
                "supplier_signature": f"supplier-{identity}",
                "constraint_signature": f"constraint-{identity}",
                "answer_signature": f"answer-{identity}",
            })
        rows_by_split[split] = rows
    return rows_by_split


def test_horizon_dataset_contract_passes_exact_stratified_isolated_roster():
    report = audit_horizon_dataset(_dataset())
    assert report["valid"] is True
    assert report["splits"]["train"]["task_count"] == 240
    assert report["splits"]["test"]["medium_long_fraction"] == pytest.approx(2 / 3)


def test_horizon_dataset_contract_rejects_cross_split_leakage():
    rows = _dataset()
    rows["test"][0]["answer_signature"] = rows["train"][0]["answer_signature"]
    with pytest.raises(ValueError, match="answer_signature leakage"):
        audit_horizon_dataset(rows)


def test_deterministic_task_order_is_reproducible_and_seeded():
    task_ids = [f"task-{index}" for index in range(20)]
    first = deterministic_task_order(task_ids, 20260801)
    assert first == deterministic_task_order(task_ids, 20260801)
    assert first != deterministic_task_order(task_ids, 20260802)
    assert set(first) == set(task_ids)
