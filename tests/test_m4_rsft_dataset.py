import json
from pathlib import Path

import pytest

from miniwebwork.sft.m4_rsft_dataset import build_m4_rsft_dataset


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
    for task_id in ("A", "B"):
        for rollout_index in range(4):
            is_success = task_id == "A" and (
                (pass_index == 1 and rollout_index == 1)
                or (pass_index == 2 and rollout_index == 2)
            )
            token_count = 3 if pass_index == 1 and rollout_index == 1 else 2
            records.append(
                _record(task_id, pass_index, rollout_index, success=is_success, token_count=token_count)
            )
    return {
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
        "records": records,
    }


def _write_artifact(path: Path, artifact: dict) -> Path:
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_rsft_builder_uses_only_verified_success_and_preserves_selection_audit(tmp_path: Path):
    first = _write_artifact(tmp_path / "pass1.json", _artifact(1))
    second = _write_artifact(tmp_path / "pass2.json", _artifact(2))

    manifest = build_m4_rsft_dataset([first, second], tmp_path / "out", seed=20260801)

    assert manifest["selected_task_count"] == 1
    assert manifest["unselected_task_count"] == 1
    assert manifest["source_pass_indices"] == [1, 2]
    selected = manifest["selection_audit"][0]
    assert selected["task_id"] == "A"
    assert selected["candidate_count"] == 8
    assert selected["valid_success_count"] == 2
    assert selected["selected_episode_id"] == "EP-P2-A-2"
    row = json.loads((tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8"))
    assert row["episode_id"] == "EP-P2-A-2"
    assert row["labels"] == [-100, -100, 3, 4]


def test_rsft_builder_rejects_missing_ordered_pass_provenance(tmp_path: Path):
    first = _write_artifact(tmp_path / "pass1.json", _artifact(1))
    duplicate = _artifact(2)
    duplicate["collection_pass_index"] = 1
    second = _write_artifact(tmp_path / "not-pass2.json", duplicate)

    with pytest.raises(ValueError, match="pass indices"):
        build_m4_rsft_dataset([first, second], tmp_path / "out", seed=20260801)
