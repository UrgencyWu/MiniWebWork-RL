import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _record(task_id: str, pass_index: int, rollout_index: int, success: bool):
    tokens = [3, 4, 5]
    return {
        "task_id": task_id,
        "task_type": "cheapest_feasible",
        "episode_id": f"EP-P{pass_index}-{task_id}-{rollout_index}",
        "rollout_index": rollout_index,
        "rollout_seed": 1000 * pass_index + rollout_index,
        "policy": "v3-test",
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


def _load_probe():
    path = ROOT / "scripts" / "m2_3_mini_single_probe.py"
    spec = importlib.util.spec_from_file_location("m4_v3_resume_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_accepts_only_an_exact_completed_group_prefix(monkeypatch, tmp_path: Path):
    probe = _load_probe()
    tasks = [{"task_id": "M4-TRAIN-W001-CHEAPEST_FEASIBLE"}, {"task_id": "M4-TRAIN-W002-CHEAPEST_FEASIBLE"}]
    monkeypatch.setattr(probe, "_load_tasks", lambda *args, **kwargs: (list(tasks), "task-hash"))
    monkeypatch.setattr(probe, "_directory_sha256", lambda *args, **kwargs: "adapter-hash")
    monkeypatch.setattr(probe, "_file_sha256", lambda *args, **kwargs: "prompt-hash")
    monkeypatch.setattr(probe, "_git_sha", lambda: "v3-sha")

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            self.payload = {}
        def write(self):
            pass
        def update(self, **kwargs):
            pass
        def finish_task(self):
            pass

    class StopBeforeGpu(Exception):
        pass

    monkeypatch.setattr(probe, "Heartbeat", FakeHeartbeat)
    monkeypatch.setattr(probe, "load_policy", lambda *args, **kwargs: (_ for _ in ()).throw(StopBeforeGpu()))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    task_order = [task["task_id"] for task in tasks]
    random.Random(20260801).shuffle(task_order)
    task_order_sha = hashlib.sha256("\n".join(task_order).encode("utf-8")).hexdigest()
    record = _record(task_order[0], 1, 0, success=True)
    records = [dict(record, rollout_index=index) for index in range(4)]
    resume = tmp_path / "incremental.json"
    resume.write_text(json.dumps({
        "complete": False,
        "git_sha": "v3-sha",
        "policy": "v3-test",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "K": 4,
        "seed": 1234,
        "study_seed": 20260801,
        "collection_pass_index": 1,
        "task_source_sha256": "task-hash",
        "task_order_seed": 20260801,
        "task_order_sha256": task_order_sha,
        "adapter_sha256": "adapter-hash",
        "max_new_tokens": 128,
        "max_collected_action_tokens": 125000,
        "full_task_order_sha256": task_order_sha,
        "completed_task_count": 1,
        "collected_action_tokens": 12,
        "stopped_for_action_token_budget": False,
        "groups": [{"task_id": task_order[0]}],
        "records": records,
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "m2_3_mini_single_probe.py", "--policy", "custom", "--policy-label", "v3-test",
        "--adapter", str(adapter), "--task-dir", str(tmp_path / "tasks"), "--seed-dir", str(seed_dir),
        "--temperature", "1.0", "--top-p", "1.0", "--top-k", "0", "--K", "4",
        "--seed", "1234", "--study-seed", "20260801", "--collection-pass-index", "1",
        "--split", "train", "--task-order-seed", "20260801", "--max-model-turns", "20",
        "--max-env-steps", "20", "--max-new-tokens", "128", "--max-collected-action-tokens", "125000",
        "--study-id", "m4_rlvr_v3", "--resume-from", str(resume), "--output-dir", str(tmp_path / "out"),
    ])
    with pytest.raises(StopBeforeGpu):
        probe.main()
