import json
from pathlib import Path

import pytest

from miniwebwork.sft.m4_rsft_dataset import build_m4_rsft_dataset


TASK_IDS = tuple(f"M4-TRAIN-W{index:03d}-CHEAPEST_FEASIBLE" for index in range(1, 13))


def _record(task_id: str, pass_index: int, rollout_index: int, *, success: bool, token_count: int):
    tokens = list(range(3, 3 + token_count))
    return {
        "task_id": task_id,
        "task_type": "cheapest_feasible",
        "episode_id": f"EP-P{pass_index}-{task_id}-{rollout_index}",
        "rollout_index": rollout_index,
        "rollout_seed": 1000 * pass_index + rollout_index,
        "policy": "m4_rsft_seed20260801",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "success": success,
        "reward": 1.0 if success else 0.0,
        "rollout_valid": True,
        "failure_origin": "none" if success else "policy",
        "termination_reason": "verified_submission" if success else "premature_finish",
        "model_turns": 1,
        "environment_steps": 1,
        "schema_valid_count": 1,
        "schema_invalid_count": 0,
        "steps": [
            {
                "turn": 1,
                "page_type": "products",
                "prompt_token_ids": [1, 2],
                "generated_token_ids": tokens,
                "token_logprobs": [-0.2] * token_count,
                "sampling_logprobs": [-0.2] * token_count,
                "strict_json_success": True,
                "schema_valid": True,
                "parsed_action": {"action_type": "search_products"},
                "env_action_success": True,
            }
        ],
    }


def _artifact(pass_index: int) -> dict:
    records = []
    for task_id in TASK_IDS:
        for rollout_index in range(4):
            is_success = task_id == TASK_IDS[0] and (
                (pass_index == 1 and rollout_index == 1)
                or (pass_index == 2 and rollout_index == 2)
            )
            token_count = 3 if pass_index == 1 and rollout_index == 1 else 2
            records.append(
                _record(task_id, pass_index, rollout_index, success=is_success, token_count=token_count)
            )
    artifact = {
        "schema_version": "3.3",
        "complete": True,
        "study_id": "m4_rlvr_v1",
        "split": "train",
        "K": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "generation_runtime": {"use_cache": False, "strict_on_policy": True},
        "seed": 8000 + pass_index,
        "study_seed": 20260801,
        "collection_pass_index": pass_index,
        "adapter_sha256": "same-initial-adapter",
        "task_source_sha256": "combined-train-task-source",
        "task_order_seed": 20260801,
        "task_order_sha256": "fixed-rsft-roster",
        "full_task_order_sha256": "full-frozen-train-order",
        "max_model_turns": 20,
        "max_new_tokens": 128,
        "max_collected_action_tokens": 125_000,
        "available_task_count": 240,
        "max_tasks": 12,
        "requested_task_count": 12,
        "completed_task_count": 12,
        "stopped_for_action_token_budget": False,
        "records": records,
    }
    artifact["collected_action_tokens"] = sum(
        len(step["generated_token_ids"])
        for record in records
        for step in record["steps"]
    )
    return artifact


def _write_artifact(path: Path, artifact: dict) -> Path:
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_rsft_builder_uses_only_verified_success_and_preserves_selection_audit(tmp_path: Path):
    first = _write_artifact(tmp_path / "pass1.json", _artifact(1))
    second = _write_artifact(tmp_path / "pass2.json", _artifact(2))

    manifest = build_m4_rsft_dataset(
        [first, second],
        tmp_path / "out",
        seed=20260801,
        expected_task_ids=TASK_IDS,
    )

    assert manifest["selected_task_count"] == 1
    assert manifest["unselected_task_count"] == 11
    assert manifest["source_pass_indices"] == [1, 2]
    assert manifest["source_task_count"] == 12
    assert manifest["source_task_universe_count"] == 240
    assert manifest["source_task_coverage_fraction"] == 0.05
    assert manifest["source_roster_overlap_count"] == 12
    assert manifest["source_roster_overlap_fraction"] == 1.0
    assert manifest["source_total_collected_action_tokens"] <= 250_000
    selected = manifest["selection_audit"][0]
    assert selected["task_id"] == TASK_IDS[0]
    assert selected["candidate_count"] == 8
    assert selected["valid_success_count"] == 2
    assert selected["selected_episode_id"] == f"EP-P2-{TASK_IDS[0]}-2"
    row = json.loads((tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8"))
    assert row["episode_id"] == f"EP-P2-{TASK_IDS[0]}-2"
    assert row["labels"] == [-100, -100, 3, 4]


def test_rsft_builder_rejects_missing_ordered_pass_provenance(tmp_path: Path):
    first = _write_artifact(tmp_path / "pass1.json", _artifact(1))
    duplicate = _artifact(2)
    duplicate["collection_pass_index"] = 1
    second = _write_artifact(tmp_path / "not-pass2.json", duplicate)

    with pytest.raises(ValueError, match="pass indices"):
        build_m4_rsft_dataset(
            [first, second], tmp_path / "out", seed=20260801, expected_task_ids=TASK_IDS
        )


def test_rsft_builder_rejects_passes_with_different_or_nonfixed_rosters(tmp_path: Path):
    first = _write_artifact(tmp_path / "pass1.json", _artifact(1))
    second_artifact = _artifact(2)
    for record in second_artifact["records"]:
        if record["task_id"] == TASK_IDS[-1]:
            record["task_id"] = "M4-TRAIN-W999-CHEAPEST_FEASIBLE"
    second = _write_artifact(tmp_path / "pass2.json", second_artifact)

    with pytest.raises(ValueError, match="identical train task roster"):
        build_m4_rsft_dataset(
            [first, second], tmp_path / "out", seed=20260801, expected_task_ids=TASK_IDS
        )


def test_rsft_builder_preserves_an_auditable_no_signal_outcome_without_oracle_fallback(tmp_path: Path):
    first_artifact = _artifact(1)
    second_artifact = _artifact(2)
    for artifact in (first_artifact, second_artifact):
        for record in artifact["records"]:
            record["success"] = False
            record["reward"] = 0.0
            record["failure_origin"] = "policy"
    first = _write_artifact(tmp_path / "pass1.json", first_artifact)
    second = _write_artifact(tmp_path / "pass2.json", second_artifact)

    manifest = build_m4_rsft_dataset(
        [first, second],
        tmp_path / "out",
        seed=20260801,
        expected_task_ids=TASK_IDS,
    )

    assert manifest["selected_task_count"] == 0
    assert manifest["no_verified_successes"] is True
    assert (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8") == ""
