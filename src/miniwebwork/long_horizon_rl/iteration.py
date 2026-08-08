"""Atomic policy-iteration commits and crash reconciliation for online RL."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from ..m4_long_horizon_protocol import ONLINE_LEARNER_CONFIG, STUDY_ID
from .contracts import (
    COLLECTION_SCHEMA,
    RunIdentity,
    atomic_write_json,
    bounded_file_sentinel,
    canonical_json_bytes,
    directory_sha256,
    sha256_file,
    sha256_json,
)

RUN_STATE_SCHEMA = "m4_long_horizon_run_state_v1"
ITERATION_SCHEMA = "m4_long_horizon_iteration_v1"
LEARNER_REPORT_SCHEMA = "m4_long_horizon_learner_report_v1"
ITERATION_LEDGER_EVENT_SCHEMA = "m4_long_horizon_iteration_ledger_event_v1"
POLICY_VERSION_PATTERN = re.compile(r"^policy_([0-9]{4,})$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.exists(), f"artifact is missing: {resolved}")
    if resolved.is_file():
        return {"kind": "file", "path": str(resolved), "sha256": sha256_file(resolved)}
    _require(resolved.is_dir(), f"unsupported artifact type: {resolved}")
    return {"kind": "directory", "path": str(resolved), "sha256": directory_sha256(resolved)}


def _state_content_sha256(state: Mapping[str, Any]) -> str:
    payload = dict(state)
    payload.pop("state_sha256", None)
    return sha256_json(payload)


def _iteration_content_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("iteration_manifest_sha256", None)
    return sha256_json(payload)


def _ledger_event_hash(event: Mapping[str, Any]) -> str:
    payload = dict(event)
    payload.pop("event_sha256", None)
    return sha256_json(payload)


class IterationLedger:
    """Small append-only fsync ledger spanning policy versions."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events = list(self._read())
        self._size = self.path.stat().st_size if self.path.exists() else 0
        self._stat_signature = self._current_stat_signature()
        self._content_sentinel = bounded_file_sentinel(self.path)

    def _current_stat_signature(self) -> tuple[int, int, int, int, int]:
        if not self.path.exists():
            return (0, 0, 0, 0, 0)
        stat = self.path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _read(self) -> tuple[dict[str, Any], ...]:
        if not self.path.is_file():
            return ()
        events: list[dict[str, Any]] = []
        prior = "0" * 64
        for index, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines()
        ):
            _require(bool(line.strip()), f"blank iteration ledger line {index + 1}")
            event = json.loads(line)
            _require(
                event.get("schema_version") == ITERATION_LEDGER_EVENT_SCHEMA,
                f"iteration ledger schema drift at line {index + 1}",
            )
            _require(event.get("sequence") == index, "iteration ledger sequence drift")
            _require(event.get("previous_sha256") == prior, "iteration ledger chain drift")
            expected = _ledger_event_hash(event)
            _require(event.get("event_sha256") == expected, "iteration ledger hash mismatch")
            _require(isinstance(event.get("payload"), dict), "iteration ledger payload drift")
            events.append(event)
            prior = expected
        return tuple(events)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        _require(
            self._current_stat_signature() == self._stat_signature
            and bounded_file_sentinel(self.path) == self._content_sentinel,
            "iteration ledger changed outside this writer",
        )
        return tuple(dict(event) for event in self._events)

    def append(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        _require(isinstance(event_type, str) and bool(event_type), "ledger event type is empty")
        normalized = json.loads(canonical_json_bytes(dict(payload)))
        _require(
            self._current_stat_signature() == self._stat_signature
            and bounded_file_sentinel(self.path) == self._content_sentinel,
            "iteration ledger changed outside this writer",
        )
        event = {
            "schema_version": ITERATION_LEDGER_EVENT_SCHEMA,
            "sequence": len(self._events),
            "previous_sha256": (
                self._events[-1]["event_sha256"] if self._events else "0" * 64
            ),
            "event_type": event_type,
            "timestamp_ns": time.time_ns(),
            "payload": normalized,
        }
        event["event_sha256"] = _ledger_event_hash(event)
        line = canonical_json_bytes(event) + b"\n"
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            written = 0
            while written < len(line):
                written += os.write(descriptor, line[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._events.append(event)
        self._size += len(line)
        self._stat_signature = self._current_stat_signature()
        self._content_sentinel = bounded_file_sentinel(self.path)
        return event


class IterationStore:
    """Advance adapter, optimizer, sampler, and token state as one audited unit."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.root / ".staging"
        self.iterations_dir = self.root / "iterations"
        self.invalidated_stages_dir = self.root / "invalidated_stages"
        for path in (self.staging_dir, self.iterations_dir, self.invalidated_stages_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "run_state.json"
        self.ledger = IterationLedger(self.root / "iteration_ledger.jsonl")

    @property
    def initialized(self) -> bool:
        return self.state_path.is_file()

    def _validate_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        _require(state.get("schema_version") == RUN_STATE_SCHEMA, "run-state schema drift")
        _require(state.get("study_id") == STUDY_ID, "run-state study drift")
        _require(state.get("state_sha256") == _state_content_sha256(state), "run-state hash drift")
        iteration_index = state.get("current_iteration_index")
        _require(isinstance(iteration_index, int) and iteration_index >= 0, "run-state iteration drift")
        _require(
            state.get("current_policy_version") == f"policy_{iteration_index:04d}",
            "run-state policy/iteration drift",
        )
        _require(
            isinstance(state.get("global_generated_action_tokens"), int)
            and state["global_generated_action_tokens"] >= 0,
            "run-state token ledger drift",
        )
        for field in ("current_adapter", "current_optimizer"):
            artifact = state.get(field, {})
            _require(artifact.get("kind") in {"file", "directory"}, f"run-state {field} kind drift")
            _require(
                isinstance(artifact.get("sha256"), str)
                and len(artifact["sha256"]) == 64,
                f"run-state {field} hash drift",
            )
            artifact_path = Path(str(artifact.get("path", ""))).expanduser()
            if not artifact_path.is_absolute():
                artifact_path = self.root / artifact_path
            actual = _artifact_descriptor(artifact_path)
            _require(actual["kind"] == artifact["kind"], f"run-state {field} kind mismatch")
            _require(actual["sha256"] == artifact["sha256"], f"run-state {field} artifact drift")
        return dict(state)

    def load_state(self) -> dict[str, Any]:
        _require(self.state_path.is_file(), "run state is not initialized")
        return self._validate_state(json.loads(self.state_path.read_text(encoding="utf-8")))

    def initialize(
        self,
        *,
        identity: RunIdentity,
        input_adapter_path: Path,
        input_optimizer_path: Path,
        task_sampler_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        identity.validate()
        _require(identity.iteration_index == 0, "initial identity must be iteration zero")
        _require(identity.policy_version == "policy_0000", "initial policy version drift")
        _require(not self.state_path.exists(), "run state is already initialized")
        _require(not self.ledger.events, "cannot initialize with a non-empty iteration ledger")
        adapter = _artifact_descriptor(input_adapter_path)
        optimizer = _artifact_descriptor(input_optimizer_path)
        _require(
            adapter["sha256"] == identity.input_adapter_sha256,
            "bootstrap adapter/identity hash mismatch",
        )
        state = {
            "schema_version": RUN_STATE_SCHEMA,
            "study_id": identity.study_id,
            "method": identity.method,
            "seed": identity.seed,
            "git_sha": identity.git_sha,
            "dataset_manifest_sha256": identity.dataset_manifest_sha256,
            "seed_manifest_sha256": identity.seed_manifest_sha256,
            "prompt_sha256": identity.prompt_sha256,
            "credit_formula_version": identity.credit_formula_version,
            "task_order_sha256": identity.task_order_sha256,
            "base_model_manifest_sha256": identity.base_model_manifest_sha256,
            "runtime_contract_sha256": identity.runtime_contract_sha256,
            "current_iteration_index": 0,
            "current_policy_version": "policy_0000",
            "current_adapter": adapter,
            "current_optimizer": optimizer,
            "global_generated_action_tokens": 0,
            "task_sampler_state": dict(task_sampler_state),
            "last_iteration_manifest_sha256": None,
            "state_revision": 0,
        }
        state["state_sha256"] = _state_content_sha256(state)
        atomic_write_json(self.state_path, state)
        self.ledger.append(
            "run_initialized",
            {
                "identity_sha256": identity.sha256,
                "state_sha256": state["state_sha256"],
                "adapter_sha256": adapter["sha256"],
                "optimizer_sha256": optimizer["sha256"],
            },
        )
        return state

    def assert_identity_matches_state(self, identity: RunIdentity) -> dict[str, Any]:
        identity.validate()
        state = self.load_state()
        expected = {
            "study_id": identity.study_id,
            "method": identity.method,
            "seed": identity.seed,
            "git_sha": identity.git_sha,
            "dataset_manifest_sha256": identity.dataset_manifest_sha256,
            "seed_manifest_sha256": identity.seed_manifest_sha256,
            "prompt_sha256": identity.prompt_sha256,
            "credit_formula_version": identity.credit_formula_version,
            "task_order_sha256": identity.task_order_sha256,
            "base_model_manifest_sha256": identity.base_model_manifest_sha256,
            "runtime_contract_sha256": identity.runtime_contract_sha256,
            "current_iteration_index": identity.iteration_index,
            "current_policy_version": identity.policy_version,
        }
        for field, value in expected.items():
            _require(state.get(field) == value, f"run-state/identity {field} mismatch")
        _require(
            state["current_adapter"]["sha256"] == identity.input_adapter_sha256,
            "run-state/identity adapter mismatch",
        )
        return state

    @staticmethod
    def _validate_collection(
        collection: Mapping[str, Any],
        identity: RunIdentity,
    ) -> dict[str, Any]:
        payload = dict(collection)
        _require(payload.get("schema_version") == COLLECTION_SCHEMA, "collection schema drift")
        _require(payload.get("identity_sha256") == identity.sha256, "collection identity mismatch")
        _require(payload.get("iteration_index") == identity.iteration_index, "collection iteration mismatch")
        _require(payload.get("complete") is True, "collection is not frozen complete")
        _require(payload.get("group_count", 0) > 0, "learner collection has no committed groups")
        _require(
            payload.get("collection_sha256")
            == sha256_json({key: value for key, value in payload.items() if key != "collection_sha256"}),
            "collection content hash drift",
        )
        all_tokens = payload.get("all_generated_action_tokens")
        committed_tokens = payload.get("committed_group_action_tokens")
        _require(
            isinstance(all_tokens, int)
            and isinstance(committed_tokens, int)
            and all_tokens >= committed_tokens > 0,
            "collection token ledger drift",
        )
        _require(isinstance(payload.get("task_sampler_state"), dict), "collection sampler state drift")
        return payload

    def begin_update(
        self,
        *,
        identity: RunIdentity,
        collection_manifest: Mapping[str, Any],
    ) -> dict[str, str]:
        state = self.assert_identity_matches_state(identity)
        collection = self._validate_collection(collection_manifest, identity)
        stage = self.staging_dir / f"iteration-{identity.iteration_index:04d}"
        _require(not stage.exists(), "iteration stage already exists; recover it explicitly")
        _require(
            not (self.iterations_dir / f"iteration-{identity.iteration_index:04d}").exists(),
            "iteration directory already committed; reconcile before updating",
        )
        stage.mkdir()
        atomic_write_json(stage / "input_run_state.json", state)
        atomic_write_json(stage / "run_identity.json", identity.to_dict())
        atomic_write_json(stage / "collection_manifest.json", collection)
        _fsync_directory(stage)
        _fsync_directory(self.staging_dir)
        self.ledger.append(
            "update_stage_started",
            {
                "iteration_index": identity.iteration_index,
                "identity_sha256": identity.sha256,
                "collection_sha256": collection["collection_sha256"],
                "stage_relative_path": str(stage.relative_to(self.root)),
            },
        )
        return {
            "stage": str(stage),
            "output_adapter": str(stage / "output_adapter"),
            "output_optimizer": str(stage / "output_optimizer"),
            "learner_report": str(stage / "learner_report.json"),
        }

    @staticmethod
    def _validate_learner_report(
        report: Mapping[str, Any],
        identity: RunIdentity,
        collection: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = dict(report)
        _require(payload.get("schema_version") == LEARNER_REPORT_SCHEMA, "learner report schema drift")
        _require(payload.get("identity_sha256") == identity.sha256, "learner identity mismatch")
        _require(
            payload.get("collection_sha256") == collection["collection_sha256"],
            "learner collection mismatch",
        )
        _require(payload.get("policy_epochs") == ONLINE_LEARNER_CONFIG["policy_epochs"], "policy epoch drift")
        _require(
            payload.get("trajectory_minibatch_size")
            == ONLINE_LEARNER_CONFIG["trajectory_minibatch_size"],
            "learner minibatch drift",
        )
        updates = payload.get("optimizer_updates")
        effective_tokens = payload.get("effective_optimizer_action_tokens")
        all_tokens = payload.get("all_generated_action_tokens")
        _require(isinstance(updates, int) and updates >= 0, "optimizer update count drift")
        _require(
            isinstance(effective_tokens, int)
            and 0 <= effective_tokens <= collection["committed_group_action_tokens"],
            "effective optimizer-token ledger drift",
        )
        _require(all_tokens == collection["all_generated_action_tokens"], "learner all-token ledger drift")
        _require((updates == 0) is (effective_tokens == 0), "optimizer updates/effective tokens mismatch")
        for field in (
            "mean_ratio",
            "clip_fraction",
            "approx_kl",
            "entropy",
            "gradient_norm",
            "parameter_change_norm",
        ):
            value = payload.get(field)
            _require(isinstance(value, (int, float)) and math.isfinite(value), f"invalid learner {field}")
        _require(payload["parameter_change_norm"] >= 0, "negative parameter change")
        if updates > 0:
            _require(payload["parameter_change_norm"] > 0, "optimizer updated without parameter change")
        return payload

    def commit_update(
        self,
        *,
        identity: RunIdentity,
        learner_report: Mapping[str, Any],
        fault_injector: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        state = self.assert_identity_matches_state(identity)
        stage = self.staging_dir / f"iteration-{identity.iteration_index:04d}"
        _require(stage.is_dir(), "iteration stage is missing")
        collection = json.loads((stage / "collection_manifest.json").read_text(encoding="utf-8"))
        collection = self._validate_collection(collection, identity)
        report = self._validate_learner_report(learner_report, identity, collection)
        learner_report_file_sha256 = atomic_write_json(stage / "learner_report.json", report)
        output_adapter = _artifact_descriptor(stage / "output_adapter")
        output_optimizer = _artifact_descriptor(stage / "output_optimizer")
        updates = report["optimizer_updates"]
        if updates == 0:
            _require(
                output_adapter["sha256"] == state["current_adapter"]["sha256"],
                "zero-update iteration changed adapter",
            )
        else:
            _require(
                output_adapter["sha256"] != state["current_adapter"]["sha256"],
                "optimizer update did not change adapter artifact",
            )
        next_index = identity.iteration_index + 1
        final = self.iterations_dir / f"iteration-{identity.iteration_index:04d}"
        _require(not final.exists(), "iteration final directory already exists")
        manifest = {
            "schema_version": ITERATION_SCHEMA,
            "study_id": identity.study_id,
            "method": identity.method,
            "seed": identity.seed,
            "iteration_index": identity.iteration_index,
            "input_policy_version": identity.policy_version,
            "output_policy_version": f"policy_{next_index:04d}",
            "identity": identity.to_dict(),
            "identity_sha256": identity.sha256,
            "input_adapter_sha256": state["current_adapter"]["sha256"],
            "input_optimizer_sha256": state["current_optimizer"]["sha256"],
            "output_adapter": {
                "kind": output_adapter["kind"],
                "relative_path": str(
                    (final / "output_adapter").relative_to(self.root)
                ),
                "sha256": output_adapter["sha256"],
            },
            "output_optimizer": {
                "kind": output_optimizer["kind"],
                "relative_path": str(
                    (final / "output_optimizer").relative_to(self.root)
                ),
                "sha256": output_optimizer["sha256"],
            },
            "collection_sha256": collection["collection_sha256"],
            "collection_group_set_sha256": collection["group_set_sha256"],
            "collection_group_count": collection["group_count"],
            "iteration_generated_action_tokens": collection["all_generated_action_tokens"],
            "global_generated_action_tokens_before": state["global_generated_action_tokens"],
            "global_generated_action_tokens_after": (
                state["global_generated_action_tokens"]
                + collection["all_generated_action_tokens"]
            ),
            "task_sampler_state_before": state["task_sampler_state"],
            "task_sampler_state_after": collection["task_sampler_state"],
            "learner_report_sha256": sha256_json(report),
            "learner_report_file_sha256": learner_report_file_sha256,
            "complete": True,
        }
        manifest["iteration_manifest_sha256"] = _iteration_content_sha256(manifest)
        atomic_write_json(stage / "iteration_manifest.json", manifest)
        _fsync_directory(stage)
        os.rename(stage, final)
        _fsync_directory(self.staging_dir)
        _fsync_directory(self.iterations_dir)
        self.ledger.append(
            "iteration_directory_committed",
            {
                "iteration_index": identity.iteration_index,
                "iteration_manifest_sha256": manifest["iteration_manifest_sha256"],
                "final_relative_path": str(final.relative_to(self.root)),
            },
        )
        if fault_injector is not None:
            fault_injector("after_iteration_directory_commit")
        new_state = self._state_after_manifest(state, manifest)
        atomic_write_json(self.state_path, new_state)
        self.ledger.append(
            "run_state_advanced",
            {
                "iteration_index": identity.iteration_index,
                "new_state_revision": new_state["state_revision"],
                "state_sha256": new_state["state_sha256"],
            },
        )
        return {"manifest": manifest, "state": new_state, "path": str(final)}

    def _load_iteration_manifest(self, path: Path) -> dict[str, Any]:
        manifest_path = path / "iteration_manifest.json"
        _require(manifest_path.is_file(), f"iteration manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(manifest.get("schema_version") == ITERATION_SCHEMA, "iteration schema drift")
        _require(manifest.get("complete") is True, "iteration is not complete")
        _require(
            manifest.get("iteration_manifest_sha256") == _iteration_content_sha256(manifest),
            "iteration manifest hash drift",
        )
        for field in ("output_adapter", "output_optimizer"):
            descriptor = manifest[field]
            artifact_path = self.root / descriptor["relative_path"]
            actual = _artifact_descriptor(artifact_path)
            _require(actual["kind"] == descriptor["kind"], f"iteration {field} kind drift")
            _require(actual["sha256"] == descriptor["sha256"], f"iteration {field} hash drift")
        collection_path = path / "collection_manifest.json"
        _require(collection_path.is_file(), "committed collection snapshot is missing")
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        _require(
            collection.get("collection_sha256") == manifest["collection_sha256"],
            "committed collection snapshot drift",
        )
        _require(
            collection.get("collection_sha256")
            == sha256_json({
                key: value for key, value in collection.items() if key != "collection_sha256"
            }),
            "committed collection content hash drift",
        )
        learner_report_path = path / "learner_report.json"
        _require(learner_report_path.is_file(), "committed learner report is missing")
        learner_report = json.loads(learner_report_path.read_text(encoding="utf-8"))
        _require(
            sha256_json(learner_report) == manifest["learner_report_sha256"],
            "committed learner report content drift",
        )
        _require(
            sha256_file(learner_report_path) == manifest["learner_report_file_sha256"],
            "committed learner report file drift",
        )
        return manifest

    @staticmethod
    def _state_after_manifest(
        state: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        _require(
            manifest["iteration_index"] == state["current_iteration_index"],
            "iteration/run-state index mismatch",
        )
        _require(
            manifest["input_policy_version"] == state["current_policy_version"],
            "iteration/run-state policy mismatch",
        )
        _require(
            manifest["input_adapter_sha256"] == state["current_adapter"]["sha256"],
            "iteration/run-state adapter mismatch",
        )
        _require(
            manifest["input_optimizer_sha256"] == state["current_optimizer"]["sha256"],
            "iteration/run-state optimizer mismatch",
        )
        _require(
            manifest["global_generated_action_tokens_before"]
            == state["global_generated_action_tokens"],
            "iteration global token predecessor mismatch",
        )
        new_state = dict(state)
        new_state.update(
            current_iteration_index=state["current_iteration_index"] + 1,
            current_policy_version=manifest["output_policy_version"],
            current_adapter={
                "kind": manifest["output_adapter"]["kind"],
                "path": manifest["output_adapter"]["relative_path"],
                "sha256": manifest["output_adapter"]["sha256"],
            },
            current_optimizer={
                "kind": manifest["output_optimizer"]["kind"],
                "path": manifest["output_optimizer"]["relative_path"],
                "sha256": manifest["output_optimizer"]["sha256"],
            },
            global_generated_action_tokens=manifest["global_generated_action_tokens_after"],
            task_sampler_state=manifest["task_sampler_state_after"],
            last_iteration_manifest_sha256=manifest["iteration_manifest_sha256"],
            state_revision=state["state_revision"] + 1,
        )
        new_state["state_sha256"] = _state_content_sha256(new_state)
        return new_state

    def reconcile_committed_iterations(self) -> dict[str, Any]:
        """Advance stale run_state only through fully verified committed directories."""

        state = self.load_state()
        reconciled = 0
        while True:
            index = state["current_iteration_index"]
            path = self.iterations_dir / f"iteration-{index:04d}"
            if not path.exists():
                break
            manifest = self._load_iteration_manifest(path)
            state = self._state_after_manifest(state, manifest)
            atomic_write_json(self.state_path, state)
            self.ledger.append(
                "run_state_reconciled",
                {
                    "iteration_index": index,
                    "iteration_manifest_sha256": manifest["iteration_manifest_sha256"],
                    "state_sha256": state["state_sha256"],
                },
            )
            reconciled += 1
        return {"reconciled_iterations": reconciled, "state": state}

    def recover_interrupted_update(self, identity: RunIdentity) -> dict[str, Any] | None:
        """Archive a partial learner stage; the frozen collection remains replayable."""

        self.assert_identity_matches_state(identity)
        stage = self.staging_dir / f"iteration-{identity.iteration_index:04d}"
        if not stage.exists():
            return None
        prefix = f"iteration-{identity.iteration_index:04d}-attempt-"
        existing = sorted(
            path for path in self.invalidated_stages_dir.iterdir()
            if path.name.startswith(prefix)
        )
        destination = self.invalidated_stages_dir / f"{prefix}{len(existing):04d}"
        _require(not destination.exists(), "invalidated stage archive collision")
        os.rename(stage, destination)
        _fsync_directory(self.staging_dir)
        _fsync_directory(self.invalidated_stages_dir)
        files = any(path.is_file() for path in destination.rglob("*"))
        archive_sha256 = directory_sha256(destination) if files else sha256_json([])
        event = self.ledger.append(
            "interrupted_stage_archived",
            {
                "iteration_index": identity.iteration_index,
                "identity_sha256": identity.sha256,
                "archive_relative_path": str(destination.relative_to(self.root)),
                "archive_directory_sha256": archive_sha256,
                "replay_frozen_collection": True,
            },
        )
        return {"path": str(destination), "event": event}
