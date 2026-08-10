from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from miniwebwork.m5_webshop_protocol import (
    PROTOCOL_PATH,
    SPLIT_EXCLUSIONS_PATH,
    content_tree_audit,
    deterministic_candidate_order,
    deterministic_selection,
    eligible_goal_indices,
    load_protocol,
    load_split_exclusions,
    load_upstream_lock,
    sft_candidate_orders,
    split_for_goal_index,
    task_id_for_goal_index,
    validate_protocol,
    validate_split_exclusions,
    validate_upstream_lock,
)
from miniwebwork.webshop_rl.credit import (
    ADVANTAGE_EPSILON,
    CREDIT_FORMULA_VERSION,
    MICRO_ADVANTAGE_WEIGHT,
    MICRO_RETURN_GAMMA,
)


def test_frozen_m5_protocol_and_upstream_lock_are_self_consistent():
    protocol = load_protocol()
    lock = load_upstream_lock()
    assert protocol["path"] == str(PROTOCOL_PATH.resolve())
    assert len(protocol["sha256"]) == 64
    assert lock["payload"]["total_runtime_bytes"] == sum(
        item["size"] for item in lock["payload"]["runtime_files"]
    )
    assert protocol["payload"]["formal_submission_allowed"] is False
    assert protocol["payload"]["scope"]["formal_models"] == 8
    assert protocol["payload"]["online"]["credit_formula_version"] == CREDIT_FORMULA_VERSION
    credit = protocol["payload"]["online"]["credit_parameters"]
    assert credit["advantage_epsilon"] == ADVANTAGE_EPSILON
    assert credit["micro_return_gamma"] == MICRO_RETURN_GAMMA
    assert credit["micro_advantage_weight"] == MICRO_ADVANTAGE_WEIGHT
    split_lock = load_split_exclusions()
    assert split_lock["path"] == str(SPLIT_EXCLUSIONS_PATH.resolve())
    assert split_lock["payload"]["eligible_counts"] == {"test": 500, "dev": 499, "train": 10885}
    runtime = protocol["payload"]["training_runtime"]
    assert runtime["critical_packages"]["chardet"] == "5.2.0"
    agent_r1 = protocol["payload"]["upstream_sources"]["agent_r1_code"]
    assert agent_r1["archive_size"] == 1628704
    assert agent_r1["archive_sha256"] == "07e6a35a159e7ed148d1e4b2b47d5e0158e3b8f60a71e5626f911477dfc7d57b"
    assert agent_r1["source_content_tree_sha256"] == "f45b0e09e4a500c3c9915159c7e395952a16d12378924c5f99ce8d42d55ecd9a"
    assert agent_r1["webshop_content_tree_sha256"] == "bf79abafad937aa6da0a5cbec69766f3bd1bd5420407e38a58e17d1b36b51c1f"
    assert protocol["payload"]["slurm"]["shared_environment_service"]["renewal_mechanism"] == (
        "sbatch_successor_afterany"
    )
    assert protocol["payload"]["server_runtime"]["source_fetch_policy"] == {
        "git_attempts": 2,
        "git_attempt_timeout_seconds": 60,
        "locked_archive_attempts": 4,
    }
    assert protocol["payload"]["slurm"]["sft_corpus"] == {
        "gpus": 0,
        "cpus": 4,
        "memory_gib": 8,
        "workers": 4,
    }


def test_slurm_service_renews_without_privileged_scontrol_and_cpu_jobs_hide_gpus():
    root = Path(__file__).resolve().parents[1]
    service = (root / "scripts" / "run_m5_webshop_service_job.sh").read_text(encoding="utf-8")
    assert "scontrol" not in service
    assert "sbatch --parsable" in service
    assert 'afterany:${SLURM_JOB_ID}' in service
    assert '--reference-audit "$data_audit"' in service
    assert '--reference-audit "$environment_audit"' in service
    setup = (root / "scripts" / "run_m5_webshop_server_setup_job.sh").read_text(encoding="utf-8")
    assert "http.version=HTTP/1.1" in setup
    assert "for delay in 0 5" in setup
    assert 'git_fetch_timeout_seconds=60' in setup
    assert '--kill-after=10s "${git_fetch_timeout_seconds}s"' in setup
    assert "scripts/m5_agent_r1_source.py" in setup
    sft_corpus = (root / "scripts" / "run_m5_webshop_sft_corpus_job.sh").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=4" in sft_corpus
    assert "--workers 4" in sft_corpus
    for name in (
        "run_m5_webshop_cpu_regression_job.sh",
        "run_m5_webshop_data_preflight_job.sh",
        "run_m5_webshop_server_setup_job.sh",
        "run_m5_webshop_service_job.sh",
        "run_m5_webshop_service_health_job.sh",
        "run_m5_webshop_sft_corpus_job.sh",
        "run_m5_training_runtime_setup_job.sh",
    ):
        script = (root / "scripts" / name).read_text(encoding="utf-8")
        assert 'export CUDA_VISIBLE_DEVICES=""' in script


def test_content_tree_hash_contract_is_fixed_and_rejects_directory_symlinks(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "alpha.txt").write_bytes(b"a")
    (tmp_path / "nested" / "beta.bin").write_bytes(b"\x00\x01")
    assert content_tree_audit(tmp_path, "") == {
        "prefix": ".",
        "sha256": "8a6217414aa226759912dd87192a5de083a6356550e0c5003ef2f499be24b283",
        "file_count": 2,
        "total_bytes": 3,
    }
    link = tmp_path / "linked-directory"
    try:
        link.symlink_to(tmp_path / "nested", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        content_tree_audit(tmp_path, "")


def test_split_roles_are_exhaustive_and_task_ids_are_canonical():
    assert split_for_goal_index(0) == "test"
    assert split_for_goal_index(499) == "test"
    assert split_for_goal_index(500) == "dev"
    assert split_for_goal_index(999) == "dev"
    assert split_for_goal_index(1000) == "train"
    assert split_for_goal_index(12086) == "train"
    assert task_id_for_goal_index(7) == "webshop_goal_00007"
    assert len(eligible_goal_indices("test")) == 500
    assert len(eligible_goal_indices("dev")) == 499
    assert len(eligible_goal_indices("train")) == 10885
    assert 871 not in eligible_goal_indices("dev")
    assert 1031 not in eligible_goal_indices("train")
    for invalid in (-1, 12087, True, 1.5):
        with pytest.raises(ValueError):
            split_for_goal_index(invalid)  # type: ignore[arg-type]


def test_sft_candidate_order_is_deterministic_disjoint_and_test_blind():
    first = sft_candidate_orders()
    second = sft_candidate_orders()
    assert first == second
    assert len(first["train"]) == len(set(first["train"])) == 10885
    assert len(first["dev"]) == len(set(first["dev"])) == 499
    assert set(first["train"]).isdisjoint(first["dev"])
    assert min(first["train"]) >= 1000
    assert 500 <= min(first["dev"]) and max(first["dev"]) < 1000
    assert deterministic_selection(start=10, stop=20, count=5, seed=3, namespace="x") != deterministic_selection(
        start=10, stop=20, count=5, seed=4, namespace="x"
    )
    order = deterministic_candidate_order(candidates=(10, 11, 12), seed=3, namespace="x")
    assert sorted(order) == [10, 11, 12]
    assert order == deterministic_candidate_order(candidates=(10, 11, 12), seed=3, namespace="x")


def test_protocol_fails_closed_on_method_budget_split_or_authorization_drift():
    original = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    mutations = []
    changed = copy.deepcopy(original)
    changed["formal_submission_allowed"] = True
    mutations.append(changed)
    changed = copy.deepcopy(original)
    changed["scope"]["online_methods"].append("gspo")
    mutations.append(changed)
    changed = copy.deepcopy(original)
    changed["dataset"]["split_contract"]["test"]["stop_exclusive"] = 501
    mutations.append(changed)
    changed = copy.deepcopy(original)
    changed["online"]["generated_action_token_cap_per_run"] = 499999
    mutations.append(changed)
    changed = copy.deepcopy(original)
    changed["online"]["anchor_contract"]["excluded_from_grouping"].remove("prompt tokens")
    mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(ValueError):
            validate_protocol(mutation)


def test_upstream_lock_rejects_path_traversal_duplicate_and_byte_drift():
    original = load_upstream_lock()["payload"]
    changed = copy.deepcopy(original)
    changed["runtime_files"][0]["path"] = "../goals.json"
    with pytest.raises(ValueError):
        validate_upstream_lock(changed)
    changed = copy.deepcopy(original)
    changed["runtime_files"][1]["path"] = changed["runtime_files"][0]["path"]
    with pytest.raises(ValueError):
        validate_upstream_lock(changed)
    changed = copy.deepcopy(original)
    changed["total_runtime_bytes"] += 1
    with pytest.raises(ValueError):
        validate_upstream_lock(changed)


def test_split_exclusions_fail_closed_on_test_filter_count_or_hash_drift():
    original = load_split_exclusions()["payload"]
    changed = copy.deepcopy(original)
    changed["excluded_goal_indices"]["test"] = [3]
    with pytest.raises(ValueError):
        validate_split_exclusions(changed)
    changed = copy.deepcopy(original)
    changed["eligible_counts"]["dev"] = 500
    with pytest.raises(ValueError):
        validate_split_exclusions(changed)
    changed = copy.deepcopy(original)
    changed["eligible_index_sha256"]["train"] = "0" * 64
    with pytest.raises(ValueError):
        validate_split_exclusions(changed)
