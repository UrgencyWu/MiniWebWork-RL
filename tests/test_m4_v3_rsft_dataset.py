import hashlib
import json
from pathlib import Path

from miniwebwork.sft.m4_rsft_dataset import _record_from_dict
from miniwebwork.sft.m4_v3_rsft_dataset import _pack, build_m4_v3_rsft_dataset


def _record(task_id: str, pass_index: int, rollout_index: int, success: bool):
    tokens = [3, 4, 5]
    return {
        "task_id": task_id,
        "task_type": "cheapest_feasible",
        "episode_id": f"EP-P{pass_index}-{task_id}-{rollout_index}",
        "rollout_index": rollout_index,
        "rollout_seed": 1000 * pass_index + rollout_index,
        "policy": "m4_v3_rsft_seed20260801",
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
        "steps": [{
            "turn": 1,
            "page_type": "products",
            "prompt_token_ids": [1, 2],
            "generated_token_ids": tokens,
            "token_logprobs": [-0.2] * len(tokens),
            "sampling_logprobs": [-0.2] * len(tokens),
            "strict_json_success": True,
            "schema_valid": True,
            "parsed_action": {"action_type": "search_products"},
            "env_action_success": True,
        }],
    }


def _artifact(pass_index: int):
    task_ids = [f"M4-TRAIN-W{index:03d}-CHEAPEST_FEASIBLE" for index in range(1, 13)]
    records = [
        _record(task_id, pass_index, rollout_index, task_id == task_ids[0] and rollout_index == pass_index)
        for task_id in task_ids
        for rollout_index in range(4)
    ]
    groups = [{"task_id": task_id} for task_id in task_ids]
    for index, task_id in enumerate(task_ids):
        group_records = records[index * 4 : (index + 1) * 4]
        canonical_records = [_record_from_dict(record).to_dict() for record in group_records]
        encoded = json.dumps(
            canonical_records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        groups[index]["group_sha256"] = hashlib.sha256(encoded).hexdigest()
    return {
        "schema_version": "3.3",
        "complete": True,
        "study_id": "m4_rlvr_v3",
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
        "task_source_sha256": "train-source",
        "task_order_seed": 20260801,
        "task_order_sha256": "full-roster-order",
        "full_task_order_sha256": "full-order",
        "max_model_turns": 20,
        "max_new_tokens": 128,
        "max_collected_action_tokens": 125000,
        "available_task_count": 240,
        "max_tasks": None,
        "requested_task_count": 240,
        "completed_task_count": 12,
        "stopped_for_action_token_budget": True,
        "records": records,
        "groups": groups,
        "collected_action_tokens": len(records) * 3,
    }


def test_pack_reaches_the_v3_supervision_target_without_partial_examples():
    rows = [{"sample_id": "x", "labels": [-100, 1, 2], "input_ids": [0, 1, 2]}]
    packed, realized = _pack(rows, 250000)
    assert realized == 250000
    assert len(packed) == 125000
    assert all(row["labels"][0] == -100 for row in packed)


def test_v3_builder_uses_full_roster_metadata_and_deterministic_packing(tmp_path: Path):
    first = tmp_path / "pass1.json"
    second = tmp_path / "pass2.json"
    first.write_text(json.dumps(_artifact(1)), encoding="utf-8")
    second.write_text(json.dumps(_artifact(2)), encoding="utf-8")
    manifest = build_m4_v3_rsft_dataset([first, second], tmp_path / "out", seed=20260801)
    assert manifest["source_task_universe_count"] == 240
    assert manifest["source_task_count_per_pass"] == [12, 12]
    assert manifest["target_supervised_completion_tokens"] == 250000
    assert manifest["shortfall_supervised_completion_tokens"] < 3
    assert manifest["zero_completion_label_sample_fraction"] == 0.0
