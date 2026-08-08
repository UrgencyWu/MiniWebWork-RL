from __future__ import annotations

import copy
from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.contracts import (
    COLLECTION_SCHEMA,
    RunIdentity,
    directory_sha256,
    sha256_json,
)
from miniwebwork.long_horizon_rl.credit import CREDIT_FORMULA_VERSION
from miniwebwork.long_horizon_rl.iteration import (
    LEARNER_REPORT_SCHEMA,
    IterationStore,
)


def _artifact(path: Path, content: str) -> str:
    path.mkdir(parents=True)
    (path / "weights.bin").write_text(content, encoding="utf-8")
    return directory_sha256(path)


def _identity(adapter_sha: str, iteration=0, policy="policy_0000") -> RunIdentity:
    return RunIdentity(
        study_id="m4_long_horizon_credit_v1",
        git_sha="1" * 40,
        method="multi_turn_grpo",
        seed=20260801,
        iteration_index=iteration,
        policy_version=policy,
        dataset_manifest_sha256="2" * 64,
        seed_manifest_sha256="3" * 64,
        prompt_contract="browser_agent_v4_long_memory",
        prompt_sha256="4" * 64,
        credit_formula_version=CREDIT_FORMULA_VERSION,
        task_order_sha256="5" * 64,
        base_model_manifest_sha256="8" * 64,
        runtime_contract_sha256="9" * 64,
        input_adapter_sha256=adapter_sha,
    )


def _collection(identity: RunIdentity, *, all_tokens=120, committed_tokens=100):
    payload = {
        "schema_version": COLLECTION_SCHEMA,
        "identity": identity.to_dict(),
        "identity_sha256": identity.sha256,
        "iteration_index": identity.iteration_index,
        "group_count": 2,
        "group_sha256": ["a" * 64, "b" * 64],
        "group_set_sha256": sha256_json(["a" * 64, "b" * 64]),
        "attempt_journal_prefix_sha256": "c" * 64,
        "all_generated_action_tokens": all_tokens,
        "committed_group_action_tokens": committed_tokens,
        "stopped_for_token_budget": False,
        "task_sampler_state": {"cursor": 2},
        "complete": True,
    }
    payload["collection_sha256"] = sha256_json(payload)
    return payload


def _learner_report(identity, collection, *, updates=2, effective_tokens=80):
    return {
        "schema_version": LEARNER_REPORT_SCHEMA,
        "identity_sha256": identity.sha256,
        "collection_sha256": collection["collection_sha256"],
        "policy_epochs": 2,
        "trajectory_minibatch_size": 4,
        "optimizer_updates": updates,
        "effective_optimizer_action_tokens": effective_tokens,
        "all_generated_action_tokens": collection["all_generated_action_tokens"],
        "mean_ratio": 1.0,
        "clip_fraction": 0.0,
        "approx_kl": 0.001,
        "entropy": 0.5,
        "gradient_norm": 0.2 if updates else 0.0,
        "parameter_change_norm": 0.1 if updates else 0.0,
    }


def _initialized_store(tmp_path):
    adapter = tmp_path / "bootstrap_adapter"
    optimizer = tmp_path / "bootstrap_optimizer"
    adapter_sha = _artifact(adapter, "adapter-v0")
    _artifact(optimizer, "optimizer-v0")
    identity = _identity(adapter_sha)
    store = IterationStore(tmp_path / "run")
    store.initialize(
        identity=identity,
        input_adapter_path=adapter,
        input_optimizer_path=optimizer,
        task_sampler_state={"cursor": 0},
    )
    return store, identity


def test_iteration_commit_atomically_advances_adapter_optimizer_tokens_and_sampler(tmp_path):
    store, identity = _initialized_store(tmp_path)
    collection = _collection(identity)
    paths = store.begin_update(identity=identity, collection_manifest=collection)
    output_adapter_sha = _artifact(Path(paths["output_adapter"]), "adapter-v1")
    output_optimizer_sha = _artifact(Path(paths["output_optimizer"]), "optimizer-v1")
    result = store.commit_update(
        identity=identity,
        learner_report=_learner_report(identity, collection),
    )
    state = result["state"]
    assert state["current_iteration_index"] == 1
    assert state["current_policy_version"] == "policy_0001"
    assert state["current_adapter"]["sha256"] == output_adapter_sha
    assert state["current_optimizer"]["sha256"] == output_optimizer_sha
    assert state["global_generated_action_tokens"] == 120
    assert state["task_sampler_state"] == {"cursor": 2}
    assert not Path(paths["stage"]).exists()
    assert (store.iterations_dir / "iteration-0000/iteration_manifest.json").is_file()


def test_crash_after_directory_commit_reconciles_forward_without_relearning(tmp_path):
    store, identity = _initialized_store(tmp_path)
    collection = _collection(identity)
    paths = store.begin_update(identity=identity, collection_manifest=collection)
    output_adapter_sha = _artifact(Path(paths["output_adapter"]), "adapter-v1")
    _artifact(Path(paths["output_optimizer"]), "optimizer-v1")

    def crash(point):
        assert point == "after_iteration_directory_commit"
        raise RuntimeError("intentional fault")

    with pytest.raises(RuntimeError, match="intentional fault"):
        store.commit_update(
            identity=identity,
            learner_report=_learner_report(identity, collection),
            fault_injector=crash,
        )
    assert store.load_state()["current_iteration_index"] == 0
    resumed = IterationStore(store.root)
    reconciled = resumed.reconcile_committed_iterations()
    assert reconciled["reconciled_iterations"] == 1
    assert reconciled["state"]["current_iteration_index"] == 1
    assert reconciled["state"]["current_adapter"]["sha256"] == output_adapter_sha
    assert reconciled["state"]["global_generated_action_tokens"] == 120


def test_interrupted_partial_stage_is_archived_and_same_collection_can_restart(tmp_path):
    store, identity = _initialized_store(tmp_path)
    collection = _collection(identity)
    paths = store.begin_update(identity=identity, collection_manifest=collection)
    Path(paths["output_adapter"]).mkdir()
    (Path(paths["output_adapter"]) / "partial.bin").write_text("partial", encoding="utf-8")
    resumed = IterationStore(store.root)
    archived = resumed.recover_interrupted_update(identity)
    assert archived is not None
    assert Path(archived["path"]).is_dir()
    restarted = resumed.begin_update(identity=identity, collection_manifest=collection)
    assert Path(restarted["stage"]).is_dir()
    assert resumed.load_state()["global_generated_action_tokens"] == 0


def test_iteration_identity_and_collection_drift_fail_closed(tmp_path):
    store, identity = _initialized_store(tmp_path)
    wrong = _identity("f" * 64)
    with pytest.raises(ValueError, match="adapter mismatch"):
        store.assert_identity_matches_state(wrong)
    collection = _collection(identity)
    drifted = copy.deepcopy(collection)
    drifted["all_generated_action_tokens"] += 1
    with pytest.raises(ValueError, match="content hash"):
        store.begin_update(identity=identity, collection_manifest=drifted)


def test_effective_optimizer_tokens_are_unique_and_never_multiplied_by_policy_epochs(tmp_path):
    store, identity = _initialized_store(tmp_path)
    collection = _collection(identity, all_tokens=120, committed_tokens=100)
    paths = store.begin_update(identity=identity, collection_manifest=collection)
    _artifact(Path(paths["output_adapter"]), "adapter-v1")
    _artifact(Path(paths["output_optimizer"]), "optimizer-v1")
    report = _learner_report(identity, collection, effective_tokens=200)
    with pytest.raises(ValueError, match="effective optimizer-token"):
        store.commit_update(identity=identity, learner_report=report)
