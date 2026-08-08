"""Crash-auditable attempt journal and K=4 collection commit primitives."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    COLLECTION_SCHEMA,
    GROUP_SCHEMA,
    RunIdentity,
    atomic_write_json,
    bounded_file_sentinel,
    canonical_json_bytes,
    directory_sha256,
    group_content_sha256,
    sha256_file,
    sha256_json,
    validate_committed_group,
    validate_trajectory_evidence,
)

JOURNAL_SCHEMA = "m4_long_horizon_attempt_journal_v3"
JOURNAL_EVENT_SCHEMA = "m4_long_horizon_attempt_event_v3"
SAFE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _event_hash(event: Mapping[str, Any]) -> str:
    payload = dict(event)
    payload.pop("event_sha256", None)
    return sha256_json(payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_journal(path: Path) -> tuple[dict[str, Any], ...]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return ()
    events: list[dict[str, Any]] = []
    prior_hash = "0" * 64
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"blank journal line at {line_number}")
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid journal JSON at line {line_number}") from exc
        if event.get("schema_version") != JOURNAL_EVENT_SCHEMA:
            raise ValueError(f"journal event schema drift at line {line_number}")
        if event.get("sequence") != line_number - 1:
            raise ValueError(f"journal sequence drift at line {line_number}")
        if event.get("previous_sha256") != prior_hash:
            raise ValueError(f"journal hash chain drift at line {line_number}")
        expected = _event_hash(event)
        if event.get("event_sha256") != expected:
            raise ValueError(f"journal event hash mismatch at line {line_number}")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"journal payload is not an object at line {line_number}")
        prior_hash = expected
        events.append(event)
    return tuple(events)


class AppendOnlyAttemptJournal:
    """Append JSON events with a sequence/hash chain, flush, and fsync.

    A completed model turn is charged when its ``turn_generated`` event is
    appended.  Group validity never subtracts those tokens.  The class is
    thread-safe because browser workers complete turns concurrently.
    """

    def __init__(self, path: Path, identity: RunIdentity):
        identity.validate()
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self._lock = threading.RLock()
        self._events = list(read_journal(self.path))
        self._file_size = self.path.stat().st_size if self.path.exists() else 0
        self._stat_signature = self._current_stat_signature()
        self._content_sentinel = bounded_file_sentinel(self.path)
        self._generated_action_tokens = 0
        self._committed_group_ids: list[str] = []
        self._turn_charge_keys: set[tuple[str, int, str, int]] = set()
        for event in self._events:
            self._apply_event_cache(event)
        if self._events:
            first = self._events[0]
            if first.get("event_type") != "journal_created":
                raise ValueError("existing journal lacks journal_created event")
            existing_identity = first["payload"].get("identity")
            if existing_identity != identity.to_dict():
                raise ValueError("journal run identity mismatch")
        else:
            self.append(
                "journal_created",
                {
                    "journal_schema": JOURNAL_SCHEMA,
                    "identity": identity.to_dict(),
                    "identity_sha256": identity.sha256,
                },
            )

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

    def _assert_file_unchanged(self) -> None:
        if (
            self._current_stat_signature() != self._stat_signature
            or bounded_file_sentinel(self.path) != self._content_sentinel
        ):
            raise ValueError("journal file changed outside this writer")

    def _apply_event_cache(self, event: Mapping[str, Any]) -> None:
        payload = event["payload"]
        if event["event_type"] == "turn_generated":
            self._generated_action_tokens += int(payload["generated_action_tokens"])
            key = (
                str(payload["group_id"]),
                int(payload["attempt_index"]),
                str(payload["trajectory_id"]),
                int(payload["turn_index"]),
            )
            if key in self._turn_charge_keys:
                raise ValueError(f"duplicate durable turn charge: {key}")
            self._turn_charge_keys.add(key)
        elif event["event_type"] == "group_committed":
            self._committed_group_ids.append(str(payload["group_id"]))

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._assert_file_unchanged()
            return tuple(dict(event) for event in self._events)

    def append(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("journal event type must be non-empty")
        if not isinstance(payload, Mapping):
            raise ValueError("journal event payload must be a mapping")
        # Round-trip now so no non-JSON value can enter a durable event.
        normalized_payload = json.loads(canonical_json_bytes(dict(payload)))
        with self._lock:
            self._assert_file_unchanged()
            sequence = len(self._events)
            previous = self._events[-1]["event_sha256"] if self._events else "0" * 64
            event = {
                "schema_version": JOURNAL_EVENT_SCHEMA,
                "sequence": sequence,
                "previous_sha256": previous,
                "event_type": event_type,
                "timestamp_ns": time.time_ns(),
                "payload": normalized_payload,
            }
            event["event_sha256"] = _event_hash(event)
            line = canonical_json_bytes(event) + b"\n"
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                written = 0
                while written < len(line):
                    written += os.write(descriptor, line[written:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._events.append(event)
            self._file_size += len(line)
            self._stat_signature = self._current_stat_signature()
            self._content_sentinel = bounded_file_sentinel(self.path)
            self._apply_event_cache(event)
            return event

    @property
    def generated_action_tokens(self) -> int:
        with self._lock:
            self._assert_file_unchanged()
            return self._generated_action_tokens

    @property
    def committed_group_ids(self) -> tuple[str, ...]:
        with self._lock:
            self._assert_file_unchanged()
            return tuple(self._committed_group_ids)

    def next_attempt_index(self, group_id: str) -> int:
        attempts = [
            int(event["payload"]["attempt_index"])
            for event in self.events
            if event["event_type"] == "group_started"
            and event["payload"].get("group_id") == group_id
        ]
        return max(attempts, default=-1) + 1

    def append_turn_generated(
        self,
        *,
        group_id: str,
        iteration_index: int,
        attempt_index: int,
        trajectory_id: str,
        rollout_index: int,
        turn_index: int,
        request_id: str,
        sampling_seed: int,
        policy_version: str,
        adapter_sha256: str,
        rollout_adapter_sha256: str,
        adapter_semantic_sha256: str,
        generated_token_ids: Sequence[int],
    ) -> dict[str, Any]:
        token_ids = [int(token_id) for token_id in generated_token_ids]
        if not token_ids:
            raise ValueError("generated turn token IDs cannot be empty")
        if iteration_index != self.identity.iteration_index:
            raise ValueError("turn charge iteration/identity mismatch")
        if policy_version != self.identity.policy_version:
            raise ValueError("turn charge policy/identity mismatch")
        if adapter_sha256 != self.identity.input_adapter_sha256:
            raise ValueError("turn charge adapter/identity mismatch")
        if rollout_adapter_sha256 != self.identity.input_rollout_adapter_sha256:
            raise ValueError("turn charge rollout adapter/identity mismatch")
        if adapter_semantic_sha256 != self.identity.input_adapter_semantic_sha256:
            raise ValueError("turn charge adapter semantic/identity mismatch")
        if attempt_index < 0 or rollout_index < 0 or turn_index <= 0 or sampling_seed < 0:
            raise ValueError("turn charge indices and sampling seed must be non-negative")
        if not request_id:
            raise ValueError("turn charge request id cannot be empty")
        key = (group_id, attempt_index, trajectory_id, turn_index)
        with self._lock:
            if key in self._turn_charge_keys:
                raise ValueError(f"turn was already charged: {key}")
            return self.append(
                "turn_generated",
                {
                    "group_id": group_id,
                    "iteration_index": iteration_index,
                    "attempt_index": attempt_index,
                    "trajectory_id": trajectory_id,
                    "rollout_index": rollout_index,
                    "turn_index": turn_index,
                    "request_id": request_id,
                    "sampling_seed": sampling_seed,
                    "policy_version": policy_version,
                    "adapter_sha256": adapter_sha256,
                    "rollout_adapter_sha256": rollout_adapter_sha256,
                    "adapter_semantic_sha256": adapter_semantic_sha256,
                    "generated_token_ids_sha256": sha256_json(token_ids),
                    "generated_action_tokens": len(token_ids),
                },
            )


class CollectionStore:
    """Own immutable group artifacts and one append-only attempt ledger."""

    def __init__(self, root: Path, identity: RunIdentity):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.groups_dir = self.root / "groups"
        self.groups_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir = self.root / "attempts"
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        self.invalidated_attempts_dir = self.root / "invalidated_attempts"
        self.invalidated_attempts_dir.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.journal = AppendOnlyAttemptJournal(self.root / "attempt_journal.jsonl", identity)
        self.identity_path = self.root / "run_identity.json"
        if self.identity_path.exists():
            existing = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if existing != identity.to_dict():
                raise ValueError("collection run identity mismatch")
        else:
            atomic_write_json(self.identity_path, identity.to_dict())

    def start_group(self, *, group_id: str, task_id: str, iteration_index: int) -> int:
        if SAFE_ARTIFACT_ID.fullmatch(group_id) is None:
            raise ValueError("group id is not a safe artifact identifier")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("group task id must be non-empty")
        if iteration_index != self.identity.iteration_index:
            raise ValueError("group start iteration/identity mismatch")
        if group_id in self.journal.committed_group_ids:
            raise ValueError(f"group is already committed: {group_id}")
        attempt_index = self.journal.next_attempt_index(group_id)
        self.journal.append(
            "group_started",
            {
                "group_id": group_id,
                "task_id": task_id,
                "iteration_index": iteration_index,
                "attempt_index": attempt_index,
                "policy_version": self.identity.policy_version,
                "adapter_sha256": self.identity.input_adapter_sha256,
                "rollout_adapter_sha256": (
                    self.identity.input_rollout_adapter_sha256
                ),
                "adapter_semantic_sha256": (
                    self.identity.input_adapter_semantic_sha256
                ),
                "token_total_before_attempt": self.journal.generated_action_tokens,
            },
        )
        attempt_path = self._attempt_path(group_id, attempt_index)
        if attempt_path.exists():
            raise FileExistsError(f"attempt path already exists: {attempt_path}")
        attempt_path.mkdir(parents=True)
        _fsync_directory(attempt_path.parent)
        return attempt_index

    def _attempt_path(self, group_id: str, attempt_index: int) -> Path:
        if SAFE_ARTIFACT_ID.fullmatch(group_id) is None:
            raise ValueError("group id is not a safe artifact identifier")
        if attempt_index < 0:
            raise ValueError("attempt index must be non-negative")
        return self.attempts_dir / group_id / f"attempt-{attempt_index:04d}"

    def _trajectory_path(
        self,
        group_id: str,
        attempt_index: int,
        trajectory_id: str,
    ) -> Path:
        if SAFE_ARTIFACT_ID.fullmatch(trajectory_id) is None:
            raise ValueError("trajectory id is not a safe artifact identifier")
        return self._attempt_path(group_id, attempt_index) / trajectory_id

    def write_turn_artifact(self, turn: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically persist full turn evidence after the minimal token charge."""

        from .contracts import validate_turn_evidence

        validated = validate_turn_evidence(turn, self.identity)
        group_id = validated["group_id"]
        attempt_index = validated["attempt_index"]
        trajectory_id = validated["trajectory_id"]
        turn_index = validated["turn_index"]
        charge_events = [
            event["payload"]
            for event in self.journal.events
            if event["event_type"] == "turn_generated"
            and event["payload"].get("group_id") == group_id
            and event["payload"].get("attempt_index") == attempt_index
            and event["payload"].get("trajectory_id") == trajectory_id
            and event["payload"].get("turn_index") == turn_index
        ]
        if len(charge_events) != 1:
            raise ValueError("turn artifact requires exactly one durable token charge")
        charge = charge_events[0]
        for field in (
            "iteration_index",
            "rollout_index",
            "request_id",
            "sampling_seed",
            "policy_version",
            "adapter_sha256",
            "rollout_adapter_sha256",
            "adapter_semantic_sha256",
        ):
            if charge.get(field) != validated.get(field):
                raise ValueError(f"turn artifact/charge {field} mismatch")
        if charge["generated_token_ids_sha256"] != validated["generated_token_sha256"]:
            raise ValueError("turn artifact differs from charged generated token IDs")
        if charge["generated_action_tokens"] != len(validated["generated_token_ids"]):
            raise ValueError("turn artifact differs from charged token count")

        trajectory_path = self._trajectory_path(group_id, attempt_index, trajectory_id)
        trajectory_path.mkdir(parents=True, exist_ok=True)
        path = trajectory_path / f"turn-{turn_index:04d}.json"
        if path.exists():
            raise FileExistsError(f"turn artifact already exists: {path}")
        artifact_file_sha256 = atomic_write_json(path, validated)
        turn_sha256 = sha256_json(validated)
        event = self.journal.append(
            "turn_artifact_committed",
            {
                "group_id": group_id,
                "attempt_index": attempt_index,
                "trajectory_id": trajectory_id,
                "rollout_index": validated["rollout_index"],
                "turn_index": turn_index,
                "request_id": validated["request_id"],
                "turn_sha256": turn_sha256,
                "artifact_relative_path": str(path.relative_to(self.root)),
                "artifact_file_sha256": artifact_file_sha256,
            },
        )
        return {"path": str(path), "event": event, "turn": validated}

    def mark_trajectory_completed(
        self,
        *,
        group_id: str,
        attempt_index: int,
        trajectory: Mapping[str, Any],
    ) -> dict[str, Any]:
        validated = validate_trajectory_evidence(trajectory, self.identity)
        if validated["group_id"] != group_id:
            raise ValueError("completed trajectory/group id mismatch")
        if validated["attempt_index"] != attempt_index:
            raise ValueError("completed trajectory/attempt mismatch")
        turn_events = [
            event["payload"]
            for event in self.journal.events
            if event["event_type"] == "turn_artifact_committed"
            and event["payload"].get("group_id") == group_id
            and event["payload"].get("attempt_index") == attempt_index
            and event["payload"].get("trajectory_id") == validated["trajectory_id"]
        ]
        by_turn = {event["turn_index"]: event for event in turn_events}
        if len(by_turn) != len(validated["turns"]):
            raise ValueError("trajectory completion requires every turn artifact")
        for turn in validated["turns"]:
            event = by_turn.get(turn["turn_index"])
            if event is None or event["turn_sha256"] != sha256_json(turn):
                raise ValueError("trajectory turn differs from durable turn artifact")
            path = self.root / event["artifact_relative_path"]
            if not path.is_file() or sha256_file(path) != event["artifact_file_sha256"]:
                raise ValueError("trajectory turn artifact file hash drift")

        trajectory_path = self._trajectory_path(
            group_id,
            attempt_index,
            validated["trajectory_id"],
        )
        path = trajectory_path / "trajectory.json"
        if path.exists():
            raise FileExistsError(f"trajectory artifact already exists: {path}")
        artifact_file_sha256 = atomic_write_json(path, validated)
        event = self.journal.append(
            "trajectory_completed",
            {
                "group_id": group_id,
                "attempt_index": attempt_index,
                "trajectory_id": validated["trajectory_id"],
                "rollout_index": validated["rollout_index"],
                "rollout_valid": validated["rollout_valid"],
                "reward": validated["reward"],
                "generated_action_tokens": validated["generated_action_tokens"],
                "trajectory_sha256": sha256_json(validated),
                "artifact_relative_path": str(path.relative_to(self.root)),
                "artifact_file_sha256": artifact_file_sha256,
            },
        )
        return {"path": str(path), "event": event, "trajectory": validated}

    def mark_group_invalid(
        self,
        *,
        group_id: str,
        attempt_index: int,
        reason: str,
    ) -> None:
        terminal = [
            event
            for event in self.journal.events
            if event["payload"].get("group_id") == group_id
            and event["payload"].get("attempt_index") == attempt_index
            and event["event_type"] in {"group_invalid", "group_committed"}
        ]
        if terminal:
            raise ValueError("group attempt already has a terminal journal event")
        self.journal.append(
            "group_invalid",
            {
                "group_id": group_id,
                "attempt_index": attempt_index,
                "reason": str(reason)[:500],
                "retained_generated_action_tokens": self.journal.generated_action_tokens,
                "resample_entire_group": True,
            },
        )

    def recover_incomplete_attempts(self) -> tuple[dict[str, Any], ...]:
        """Invalidate and archive any attempt that lacked a durable group commit."""

        recovered: list[dict[str, Any]] = []
        events = self.journal.events
        starts = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"]): event["payload"]
            for event in events
            if event["event_type"] == "group_started"
        }
        committed = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] == "group_committed"
        }
        invalid = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] == "group_invalid"
        }
        for key in sorted(starts):
            if key in committed:
                continue
            group_id, attempt_index = key
            if key not in invalid:
                self.mark_group_invalid(
                    group_id=group_id,
                    attempt_index=attempt_index,
                    reason="resume_detected_incomplete_attempt",
                )
            event = self.archive_invalid_attempt(
                group_id=group_id,
                attempt_index=attempt_index,
            )
            if event is not None:
                recovered.append(event)
        return tuple(recovered)

    def archive_invalid_attempt(
        self,
        *,
        group_id: str,
        attempt_index: int,
    ) -> dict[str, Any] | None:
        """Archive exactly one terminal-invalid attempt without touching peers."""

        key = (group_id, attempt_index)
        events = self.journal.events
        starts = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] == "group_started"
        }
        committed = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] == "group_committed"
        }
        invalid = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] == "group_invalid"
        }
        archived = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] == "attempt_archived"
        }
        if key in archived:
            return None
        if key not in starts:
            raise ValueError("cannot archive an attempt without a start event")
        if key in committed:
            raise ValueError("cannot archive a committed group attempt")
        if key not in invalid:
            raise ValueError("cannot archive an attempt before terminal invalidation")

        source = self._attempt_path(group_id, attempt_index)
        destination = (
            self.invalidated_attempts_dir
            / group_id
            / f"attempt-{attempt_index:04d}"
        )
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"invalidated attempt archive collision: {destination}")
            os.rename(source, destination)
            _fsync_directory(source.parent)
            _fsync_directory(destination.parent)
        orphan_group = self.groups_dir / f"{group_id}.json"
        if orphan_group.exists():
            destination.mkdir(parents=True, exist_ok=True)
            orphan_destination = destination / "orphan-group-artifact.json"
            if orphan_destination.exists():
                raise FileExistsError(
                    f"orphan group archive collision: {orphan_destination}"
                )
            os.rename(orphan_group, orphan_destination)
            _fsync_directory(orphan_group.parent)
            _fsync_directory(destination)
        archived_files = destination.exists() and any(
            path.is_file() for path in destination.rglob("*")
        )
        return self.journal.append(
            "attempt_archived",
            {
                "group_id": group_id,
                "attempt_index": attempt_index,
                "archive_relative_path": (
                    str(destination.relative_to(self.root)) if destination.exists() else None
                ),
                "archive_directory_sha256": (
                    (
                        directory_sha256(destination)
                        if archived_files
                        else sha256_json([])
                    )
                    if destination.exists()
                    else None
                ),
                "retained_generated_action_tokens": self.journal.generated_action_tokens,
            },
        )

    def commit_group(self, group: Mapping[str, Any], *, attempt_index: int) -> dict[str, Any]:
        validated = validate_committed_group(group, self.identity)
        group_id = validated["group_id"]
        if validated["attempt_index"] != attempt_index:
            raise ValueError("group artifact/attempt mismatch")
        if SAFE_ARTIFACT_ID.fullmatch(group_id) is None:
            raise ValueError("group id is not a safe artifact identifier")
        if group_id in self.journal.committed_group_ids:
            raise ValueError(f"refusing to commit group twice: {group_id}")
        starts = [
            event
            for event in self.journal.events
            if event["event_type"] == "group_started"
            and event["payload"].get("group_id") == group_id
        ]
        if not starts or starts[-1]["payload"].get("attempt_index") != attempt_index:
            raise ValueError("group commit does not match the latest started attempt")
        invalidated = any(
            event["event_type"] == "group_invalid"
            and event["payload"].get("group_id") == group_id
            and event["payload"].get("attempt_index") == attempt_index
            for event in self.journal.events
        )
        if invalidated:
            raise ValueError("cannot commit an invalidated group attempt")
        completed = [
            event["payload"]
            for event in self.journal.events
            if event["event_type"] == "trajectory_completed"
            and event["payload"].get("group_id") == group_id
            and event["payload"].get("attempt_index") == attempt_index
        ]
        if len(completed) != self.identity.group_size:
            raise ValueError("group commit requires exactly K completed trajectory journal events")
        by_rollout = {item["rollout_index"]: item for item in completed}
        if sorted(by_rollout) != list(range(self.identity.group_size)) or len(by_rollout) != len(completed):
            raise ValueError("completed trajectory journal events are not unique rollout indices 0..K-1")
        for trajectory in validated["trajectories"]:
            event = by_rollout[trajectory["rollout_index"]]
            if event["rollout_valid"] is not True:
                raise ValueError("infrastructure-invalid trajectory cannot enter a group commit")
            if event["trajectory_sha256"] != sha256_json(trajectory):
                raise ValueError("committed trajectory differs from its durable completion event")
            artifact_path = self.root / event["artifact_relative_path"]
            if (
                not artifact_path.is_file()
                or sha256_file(artifact_path) != event["artifact_file_sha256"]
            ):
                raise ValueError("completed trajectory artifact file hash drift")
        path = self.groups_dir / f"{group_id}.json"
        if path.exists():
            raise FileExistsError(f"group artifact already exists without a commit event: {path}")
        artifact_content_sha256 = atomic_write_json(path, validated)
        artifact_file_sha256 = sha256_file(path)
        event = self.journal.append(
            "group_committed",
            {
                "group_id": group_id,
                "attempt_index": attempt_index,
                "group_sha256": validated["group_sha256"],
                "artifact_relative_path": str(path.relative_to(self.root)),
                "artifact_content_sha256": artifact_content_sha256,
                "artifact_file_sha256": artifact_file_sha256,
                "generated_action_tokens": validated["generated_action_tokens"],
                "token_total_after_commit": self.journal.generated_action_tokens,
            },
        )
        return {"path": str(path), "event": event, "group": validated}

    def load_committed_groups(self) -> tuple[dict[str, Any], ...]:
        groups = []
        for event in self.journal.events:
            if event["event_type"] != "group_committed":
                continue
            payload = event["payload"]
            path = self.root / payload["artifact_relative_path"]
            if sha256_file(path) != payload["artifact_file_sha256"]:
                raise ValueError(f"committed group artifact hash drift: {path}")
            group = json.loads(path.read_text(encoding="utf-8"))
            validate_committed_group(group, self.identity)
            groups.append(group)
        return tuple(sorted(groups, key=lambda group: group["group_id"]))

    def freeze_collection(
        self,
        *,
        iteration_index: int,
        stopped_for_token_budget: bool,
        task_sampler_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        events = self.journal.events
        starts = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] == "group_started"
        }
        terminals = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] in {"group_invalid", "group_committed"}
        }
        if starts - terminals:
            raise ValueError("cannot freeze collection with an incomplete group attempt")
        invalid = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] == "group_invalid"
        }
        archived = {
            (event["payload"]["group_id"], event["payload"]["attempt_index"])
            for event in events
            if event["event_type"] == "attempt_archived"
        }
        if invalid - archived:
            raise ValueError("cannot freeze collection before invalid attempts are archived")
        path = self.root / "collection_manifest.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("schema_version") != COLLECTION_SCHEMA:
                raise ValueError("frozen collection manifest schema drift")
            if existing.get("identity_sha256") != self.identity.sha256:
                raise ValueError("frozen collection identity mismatch")
            if existing.get("iteration_index") != iteration_index:
                raise ValueError("frozen collection iteration mismatch")
            if existing.get("stopped_for_token_budget") != bool(stopped_for_token_budget):
                raise ValueError("frozen collection budget-stop mismatch")
            if existing.get("task_sampler_state") != dict(task_sampler_state):
                raise ValueError("frozen collection sampler state mismatch")
            if existing.get("collection_sha256") != sha256_json({
                key: value for key, value in existing.items() if key != "collection_sha256"
            }):
                raise ValueError("frozen collection manifest hash mismatch")
            return existing
        groups = self.load_committed_groups()
        group_hashes = [group["group_sha256"] for group in groups]
        manifest = {
            "schema_version": COLLECTION_SCHEMA,
            "identity": self.identity.to_dict(),
            "identity_sha256": self.identity.sha256,
            "iteration_index": iteration_index,
            "group_count": len(groups),
            "group_sha256": group_hashes,
            "group_set_sha256": sha256_json(group_hashes),
            "attempt_journal_prefix_sha256": sha256_file(self.journal.path),
            "all_generated_action_tokens": self.journal.generated_action_tokens,
            "committed_group_action_tokens": sum(group["generated_action_tokens"] for group in groups),
            "stopped_for_token_budget": bool(stopped_for_token_budget),
            "task_sampler_state": dict(task_sampler_state),
            "complete": True,
        }
        manifest["collection_sha256"] = sha256_json(manifest)
        atomic_write_json(path, manifest)
        self.journal.append(
            "collection_frozen",
            {
                "iteration_index": iteration_index,
                "collection_sha256": manifest["collection_sha256"],
                "group_count": len(groups),
                "all_generated_action_tokens": self.journal.generated_action_tokens,
                "journal_prefix_sha256": manifest["attempt_journal_prefix_sha256"],
            },
        )
        return manifest


def build_committed_group(
    *,
    identity: RunIdentity,
    iteration_index: int,
    attempt_index: int,
    group_id: str,
    task_id: str,
    policy_version: str,
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    group = {
        "schema_version": GROUP_SCHEMA,
        "identity_sha256": identity.sha256,
        "study_id": identity.study_id,
        "method": identity.method,
        "seed": identity.seed,
        "iteration_index": iteration_index,
        "attempt_index": attempt_index,
        "group_id": group_id,
        "task_id": task_id,
        "policy_version": policy_version,
        "adapter_sha256": identity.input_adapter_sha256,
        "rollout_adapter_sha256": identity.input_rollout_adapter_sha256,
        "adapter_semantic_sha256": identity.input_adapter_semantic_sha256,
        "K": identity.group_size,
        "trajectories": [dict(trajectory) for trajectory in trajectories],
        "generated_action_tokens": sum(int(trajectory["generated_action_tokens"]) for trajectory in trajectories),
    }
    group["group_sha256"] = group_content_sha256(group)
    return validate_committed_group(group, identity)
