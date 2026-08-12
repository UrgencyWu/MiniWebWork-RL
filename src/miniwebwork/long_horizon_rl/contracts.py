"""Fail-closed evidence and public-state contracts for long-horizon RL."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from ..m4_long_horizon_protocol import (
    FORMAL_METHODS,
    GROUP_SIZE,
    PROMPT_CONTRACT,
    STUDY_ID,
)

RUN_IDENTITY_SCHEMA = "m4_long_horizon_run_identity_v3"
TURN_SCHEMA = "m4_long_horizon_turn_evidence_v3"
TRAJECTORY_SCHEMA = "m4_long_horizon_trajectory_v3"
GROUP_SCHEMA = "m4_long_horizon_group_v3"
COLLECTION_SCHEMA = "m4_long_horizon_collection_v3"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_ID_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
POLICY_VERSION_PATTERN = re.compile(r"^policy_[0-9]{4,}$")
FILE_SENTINEL_BLOCK_BYTES = 64 * 1024


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_file_sentinel(path: Path) -> tuple[int, str, str]:
    """Hash bounded head/tail guards to defeat coarse remote stat caching."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        empty = hashlib.sha256(b"").hexdigest()
        return (0, empty, empty)
    size = source.stat().st_size
    with source.open("rb") as handle:
        head = handle.read(FILE_SENTINEL_BLOCK_BYTES)
        handle.seek(max(0, size - FILE_SENTINEL_BLOCK_BYTES))
        tail = handle.read(FILE_SENTINEL_BLOCK_BYTES)
    return (
        size,
        hashlib.sha256(head).hexdigest(),
        hashlib.sha256(tail).hexdigest(),
    )


def directory_sha256(path: Path) -> str:
    root = Path(path).expanduser().resolve()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"directory contains no files: {root}")
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(str(candidate.relative_to(root)).encode("utf-8"))
        digest.update(sha256_file(candidate).encode("ascii"))
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> str:
    """Atomically write canonical JSON and return the content SHA256."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(value) + b"\n"
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}"
    )
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(content).hexdigest()


def publish_immutable_bytes(path: Path, content: bytes) -> str:
    """Publish bytes once, allowing only byte-identical idempotent retries.

    Training/evaluation evidence must not be replaced by a later retry.  A
    fully fsynced temporary file is hard-linked into place, so readers either
    see the complete artifact or no artifact.  If another process or an older
    attempt already published the destination, only identical bytes are
    accepted.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.publish-{os.getpid()}-{os.urandom(4).hex()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise ValueError(f"immutable artifact already differs: {destination}")
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def publish_immutable_json(path: Path, value: Any) -> str:
    """Publish canonical JSON without permitting a different overwrite."""

    return publish_immutable_bytes(path, canonical_json_bytes(value) + b"\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, field: str) -> str:
    _require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None, f"invalid {field}")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    _require(value >= minimum, f"{field} must be >= {minimum}")
    return value


def _normalize_text(value: Any, *, maximum: int) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:maximum]


def _normalized_path(observation: Mapping[str, Any]) -> str:
    raw_path = str(observation.get("path") or "")
    if not raw_path:
        raw_path = urlsplit(str(observation.get("url") or "")).path
    else:
        raw_path = urlsplit(raw_path).path
    raw_path = re.sub(r"/{2,}", "/", raw_path or "/")
    return raw_path[:500]


def _public_element(element: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist only fields visible to the policy on the rendered page."""

    options = []
    for option in element.get("options", []) if isinstance(element.get("options"), list) else []:
        if isinstance(option, Mapping):
            options.append({
                "value": _normalize_text(option.get("value"), maximum=200),
                "label": _normalize_text(option.get("label"), maximum=200),
            })
    return {
        "role": _normalize_text(element.get("role"), maximum=50),
        "tag": _normalize_text(element.get("tag"), maximum=50),
        "name": _normalize_text(element.get("name"), maximum=100),
        "text": _normalize_text(element.get("text"), maximum=200),
        "value": _normalize_text(element.get("value"), maximum=200),
        "input_type": _normalize_text(element.get("input_type"), maximum=50),
        "testid": _normalize_text(element.get("testid"), maximum=100),
        "options": options,
        "disabled": bool(element.get("disabled", False)),
    }


def canonical_public_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen whitelist used to build an anchor-state signature.

    Episode IDs, URL origins/query strings, DOM-generated element IDs, oracle
    fields, verifier output, expected answers, and any unknown keys are ignored.
    This makes hidden-field injection a no-op instead of a source of credit.
    """

    _require(isinstance(observation, Mapping), "observation must be a mapping")
    raw_elements = observation.get("elements", [])
    _require(isinstance(raw_elements, list), "observation elements must be a list")
    elements = [_public_element(item) for item in raw_elements if isinstance(item, Mapping)]
    elements.sort(key=lambda item: canonical_json_bytes(item))
    raw_result = observation.get("last_action_result")
    if isinstance(raw_result, Mapping):
        last_action_result: dict[str, Any] | None = {
            "success": bool(raw_result.get("success", False)),
            "error_code": _normalize_text(raw_result.get("error_code"), maximum=100),
            "message": _normalize_text(raw_result.get("message"), maximum=200),
            "page_changed": bool(raw_result.get("page_changed", False)),
        }
    else:
        last_action_result = None
    return {
        "schema_version": _normalize_text(observation.get("schema_version"), maximum=20),
        "task_id": _normalize_text(observation.get("task_id"), maximum=200),
        "instruction": _normalize_text(observation.get("instruction"), maximum=2000),
        "path": _normalized_path(observation),
        "page_type": _normalize_text(observation.get("page_type"), maximum=100),
        "title": _normalize_text(observation.get("title"), maximum=300),
        "visible_text": _normalize_text(observation.get("visible_text"), maximum=8000),
        "text_truncated": bool(observation.get("text_truncated", False)),
        "elements": elements,
        "last_action_result": last_action_result,
        "terminal": bool(observation.get("terminal", False)),
    }


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    _require(isinstance(token_ids, Sequence) and not isinstance(token_ids, (str, bytes)), "token ids must be a sequence")
    normalized = []
    for token_id in token_ids:
        _require_int(token_id, "token id")
        normalized.append(int(token_id))
    return sha256_json(normalized)


def public_anchor_signature(
    observation: Mapping[str, Any],
    *,
    prompt_token_ids: Sequence[int] | None = None,
) -> str:
    """Hash public environment state and, when available, exact policy context."""

    payload: dict[str, Any] = {"public_state": canonical_public_state(observation)}
    if prompt_token_ids is not None:
        payload["policy_context_token_sha256"] = token_ids_sha256(prompt_token_ids)
    return sha256_json(payload)


@dataclass(frozen=True)
class RunIdentity:
    study_id: str
    git_sha: str
    method: str
    seed: int
    iteration_index: int
    policy_version: str
    dataset_manifest_sha256: str
    seed_manifest_sha256: str
    prompt_contract: str
    prompt_sha256: str
    credit_formula_version: str
    task_order_sha256: str
    base_model_manifest_sha256: str
    runtime_contract_sha256: str
    input_adapter_sha256: str
    input_rollout_adapter_sha256: str
    input_adapter_semantic_sha256: str
    group_size: int = GROUP_SIZE
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    schema_version: str = RUN_IDENTITY_SCHEMA

    def validate(self) -> None:
        _require(self.schema_version == RUN_IDENTITY_SCHEMA, "run identity schema drift")
        _require(self.study_id == STUDY_ID, "run study id drift")
        _require(self.method in FORMAL_METHODS, "unsupported focused formal method")
        _require_int(self.seed, "seed")
        _require(
            isinstance(self.git_sha, str)
            and GIT_OBJECT_ID_PATTERN.fullmatch(self.git_sha) is not None,
            "invalid git_sha",
        )
        _require_int(self.iteration_index, "iteration_index")
        _require(
            isinstance(self.policy_version, str)
            and POLICY_VERSION_PATTERN.fullmatch(self.policy_version) is not None,
            "invalid policy_version",
        )
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        _require_sha256(self.seed_manifest_sha256, "seed_manifest_sha256")
        _require(self.prompt_contract == PROMPT_CONTRACT, "prompt contract drift")
        _require_sha256(self.prompt_sha256, "prompt_sha256")
        _require(isinstance(self.credit_formula_version, str) and self.credit_formula_version, "credit formula version is missing")
        _require_sha256(self.task_order_sha256, "task_order_sha256")
        _require_sha256(self.base_model_manifest_sha256, "base_model_manifest_sha256")
        _require_sha256(self.runtime_contract_sha256, "runtime_contract_sha256")
        _require_sha256(self.input_adapter_sha256, "input_adapter_sha256")
        _require_sha256(
            self.input_rollout_adapter_sha256,
            "input_rollout_adapter_sha256",
        )
        _require_sha256(
            self.input_adapter_semantic_sha256,
            "input_adapter_semantic_sha256",
        )
        _require(self.group_size == GROUP_SIZE, "run group size drift")
        _require(math.isclose(self.temperature, 1.0, abs_tol=1e-12), "temperature drift")
        _require(math.isclose(self.top_p, 1.0, abs_tol=1e-12), "top_p drift")
        _require(self.top_k == 0, "top_k drift")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunIdentity":
        identity = cls(**dict(value))
        identity.validate()
        return identity


def validate_turn_evidence(turn: Mapping[str, Any], identity: RunIdentity | None = None) -> dict[str, Any]:
    _require(isinstance(turn, Mapping), "turn evidence must be a mapping")
    _require(turn.get("schema_version") == TURN_SCHEMA, "turn evidence schema drift")
    for field in (
        "study_id",
        "method",
        "policy_version",
        "group_id",
        "trajectory_id",
        "task_id",
        "request_id",
        "generation_backend",
    ):
        _require(isinstance(turn.get(field), str) and turn[field], f"turn lacks {field}")
    _require(turn["study_id"] == STUDY_ID, "turn study id drift")
    _require(turn["method"] in FORMAL_METHODS, "turn method drift")
    _require(
        POLICY_VERSION_PATTERN.fullmatch(turn["policy_version"]) is not None,
        "turn policy_version drift",
    )
    for field in (
        "seed",
        "iteration_index",
        "attempt_index",
        "rollout_index",
        "turn_index",
        "sampling_seed",
    ):
        _require_int(turn.get(field), field)
    _require_sha256(turn.get("adapter_sha256"), "turn adapter_sha256")
    _require_sha256(
        turn.get("rollout_adapter_sha256"),
        "turn rollout_adapter_sha256",
    )
    _require_sha256(
        turn.get("adapter_semantic_sha256"),
        "turn adapter_semantic_sha256",
    )
    _require_sha256(turn.get("rendered_prompt_sha256"), "rendered_prompt_sha256")
    prompt_ids = turn.get("prompt_token_ids")
    generated_ids = turn.get("generated_token_ids")
    _require(isinstance(prompt_ids, list) and prompt_ids, "turn lacks prompt token IDs")
    _require(isinstance(generated_ids, list) and generated_ids, "turn lacks generated token IDs")
    for token_id in prompt_ids + generated_ids:
        _require_int(token_id, "turn token id")
    _require(
        turn.get("generated_token_sha256") == token_ids_sha256(generated_ids),
        "generated token hash mismatch",
    )
    for field in ("behavior_logprobs", "sampling_logprobs"):
        values = turn.get(field)
        _require(isinstance(values, list) and len(values) == len(generated_ids), f"{field} length mismatch")
        _require(all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values), f"{field} contains non-finite values")
    expected_prompt_token_hash = token_ids_sha256(prompt_ids)
    _require(turn.get("prompt_token_sha256") == expected_prompt_token_hash, "prompt token hash mismatch")
    observation = turn.get("observation")
    _require(isinstance(observation, Mapping), "turn lacks observation")
    _require(canonical_public_state(observation)["task_id"] == turn["task_id"], "turn observation task mismatch")
    expected_anchor = public_anchor_signature(observation, prompt_token_ids=prompt_ids)
    _require(turn.get("anchor_signature") == expected_anchor, "turn anchor signature mismatch")
    if identity is not None:
        identity.validate()
        _require(turn["study_id"] == identity.study_id, "turn/identity study mismatch")
        _require(turn["method"] == identity.method, "turn/identity method mismatch")
        _require(turn["seed"] == identity.seed, "turn/identity seed mismatch")
        _require(
            turn["iteration_index"] == identity.iteration_index,
            "turn/identity iteration mismatch",
        )
        _require(
            turn["policy_version"] == identity.policy_version,
            "turn/identity policy mismatch",
        )
        _require(turn["adapter_sha256"] == identity.input_adapter_sha256, "turn/identity adapter mismatch")
        _require(
            turn["rollout_adapter_sha256"]
            == identity.input_rollout_adapter_sha256,
            "turn/identity rollout adapter mismatch",
        )
        _require(
            turn["adapter_semantic_sha256"]
            == identity.input_adapter_semantic_sha256,
            "turn/identity adapter semantic mismatch",
        )
    return dict(turn)


def validate_trajectory_evidence(
    trajectory: Mapping[str, Any],
    identity: RunIdentity | None = None,
    *,
    require_infra_valid: bool = False,
) -> dict[str, Any]:
    _require(isinstance(trajectory, Mapping), "trajectory must be a mapping")
    _require(trajectory.get("schema_version") == TRAJECTORY_SCHEMA, "trajectory schema drift")
    for field in (
        "study_id",
        "method",
        "trajectory_id",
        "group_id",
        "task_id",
        "policy_version",
        "adapter_sha256",
        "rollout_adapter_sha256",
        "adapter_semantic_sha256",
    ):
        _require(isinstance(trajectory.get(field), str) and trajectory[field], f"trajectory lacks {field}")
    _require(trajectory["study_id"] == STUDY_ID, "trajectory study id drift")
    _require(trajectory["method"] in FORMAL_METHODS, "trajectory method drift")
    _require_int(trajectory.get("seed"), "trajectory seed")
    _require_int(trajectory.get("iteration_index"), "trajectory iteration_index")
    _require_int(trajectory.get("attempt_index"), "trajectory attempt_index")
    _require_sha256(trajectory["adapter_sha256"], "trajectory adapter_sha256")
    _require_sha256(
        trajectory["rollout_adapter_sha256"],
        "trajectory rollout_adapter_sha256",
    )
    _require_sha256(
        trajectory["adapter_semantic_sha256"],
        "trajectory adapter_semantic_sha256",
    )
    _require_int(trajectory.get("rollout_index"), "trajectory rollout_index")
    rollout_valid = trajectory.get("rollout_valid") is True
    if require_infra_valid:
        _require(rollout_valid, "committed trajectory is infrastructure-invalid")
    reward = trajectory.get("reward")
    if rollout_valid:
        _require(isinstance(reward, (int, float)) and float(reward) in (0.0, 1.0), "valid trajectory reward must be 0 or 1")
        _require(trajectory.get("success") is (float(reward) == 1.0), "trajectory success/reward mismatch")
    else:
        _require(reward is None, "infrastructure-invalid trajectory reward must be null")
    turns = trajectory.get("turns")
    _require(isinstance(turns, list), "trajectory turns must be a list")
    if require_infra_valid:
        _require(bool(turns), "committed trajectory has no generated turn")
    for index, turn in enumerate(turns, start=1):
        validate_turn_evidence(turn, identity)
        _require(turn["study_id"] == trajectory["study_id"], "turn/trajectory study mismatch")
        _require(turn["method"] == trajectory["method"], "turn/trajectory method mismatch")
        _require(turn["seed"] == trajectory["seed"], "turn/trajectory seed mismatch")
        _require(turn["iteration_index"] == trajectory["iteration_index"], "turn/trajectory iteration mismatch")
        _require(turn["attempt_index"] == trajectory["attempt_index"], "turn/trajectory attempt mismatch")
        _require(turn["trajectory_id"] == trajectory["trajectory_id"], "turn/trajectory id mismatch")
        _require(turn["group_id"] == trajectory["group_id"], "turn/group id mismatch")
        _require(turn["task_id"] == trajectory["task_id"], "turn/task id mismatch")
        _require(turn["rollout_index"] == trajectory["rollout_index"], "turn rollout index mismatch")
        _require(turn["policy_version"] == trajectory["policy_version"], "turn/trajectory policy mismatch")
        _require(turn["adapter_sha256"] == trajectory["adapter_sha256"], "turn/trajectory adapter mismatch")
        _require(
            turn["rollout_adapter_sha256"]
            == trajectory["rollout_adapter_sha256"],
            "turn/trajectory rollout adapter mismatch",
        )
        _require(
            turn["adapter_semantic_sha256"]
            == trajectory["adapter_semantic_sha256"],
            "turn/trajectory adapter semantic mismatch",
        )
        _require(turn["turn_index"] == index, "trajectory turn indices are not contiguous")
    expected_tokens = sum(len(turn["generated_token_ids"]) for turn in turns)
    _require(trajectory.get("generated_action_tokens") == expected_tokens, "trajectory token count mismatch")
    if identity is not None:
        identity.validate()
        _require(trajectory["study_id"] == identity.study_id, "trajectory/identity study mismatch")
        _require(trajectory["method"] == identity.method, "trajectory/identity method mismatch")
        _require(trajectory["seed"] == identity.seed, "trajectory/identity seed mismatch")
        _require(
            trajectory["iteration_index"] == identity.iteration_index,
            "trajectory/identity iteration mismatch",
        )
        _require(
            trajectory["policy_version"] == identity.policy_version,
            "trajectory/identity policy mismatch",
        )
        _require(
            trajectory["adapter_sha256"] == identity.input_adapter_sha256,
            "trajectory/identity adapter mismatch",
        )
        _require(
            trajectory["rollout_adapter_sha256"]
            == identity.input_rollout_adapter_sha256,
            "trajectory/identity rollout adapter mismatch",
        )
        _require(
            trajectory["adapter_semantic_sha256"]
            == identity.input_adapter_semantic_sha256,
            "trajectory/identity adapter semantic mismatch",
        )
    return dict(trajectory)


def group_content_sha256(group: Mapping[str, Any]) -> str:
    payload = dict(group)
    payload.pop("group_sha256", None)
    return sha256_json(payload)


def validate_committed_group(
    group: Mapping[str, Any],
    identity: RunIdentity | None = None,
) -> dict[str, Any]:
    _require(isinstance(group, Mapping), "group must be a mapping")
    _require(group.get("schema_version") == GROUP_SCHEMA, "group schema drift")
    _require(group.get("study_id") == STUDY_ID, "group study id drift")
    _require(group.get("method") in FORMAL_METHODS, "group method drift")
    _require_int(group.get("seed"), "group seed")
    _require_int(group.get("iteration_index"), "group iteration_index")
    _require_int(group.get("attempt_index"), "group attempt_index")
    _require_sha256(group.get("adapter_sha256"), "group adapter_sha256")
    _require_sha256(
        group.get("rollout_adapter_sha256"),
        "group rollout_adapter_sha256",
    )
    _require_sha256(
        group.get("adapter_semantic_sha256"),
        "group adapter_semantic_sha256",
    )
    _require(group.get("K") == GROUP_SIZE, "committed group K drift")
    trajectories = group.get("trajectories")
    _require(isinstance(trajectories, list) and len(trajectories) == GROUP_SIZE, "committed group requires exactly K trajectories")
    for trajectory in trajectories:
        validate_trajectory_evidence(trajectory, identity, require_infra_valid=True)
    rollout_indices = [trajectory["rollout_index"] for trajectory in trajectories]
    _require(sorted(rollout_indices) == list(range(GROUP_SIZE)), "committed group rollout indices must be 0..K-1")
    for field in (
        "study_id", "method", "seed", "iteration_index", "attempt_index", "group_id",
        "task_id", "policy_version", "adapter_sha256",
        "rollout_adapter_sha256", "adapter_semantic_sha256",
    ):
        values = {trajectory[field] for trajectory in trajectories}
        _require(values == {group.get(field)}, f"committed group {field} mismatch")
    if identity is not None:
        _require(group.get("identity_sha256") == identity.sha256, "group run identity mismatch")
        _require(group.get("study_id") == identity.study_id, "group/identity study mismatch")
        _require(group.get("method") == identity.method, "group/identity method mismatch")
        _require(group.get("seed") == identity.seed, "group/identity seed mismatch")
        _require(
            group.get("iteration_index") == identity.iteration_index,
            "group/identity iteration mismatch",
        )
        _require(
            group.get("policy_version") == identity.policy_version,
            "group/identity policy mismatch",
        )
        _require(
            group.get("adapter_sha256") == identity.input_adapter_sha256,
            "group/identity adapter mismatch",
        )
        _require(
            group.get("rollout_adapter_sha256")
            == identity.input_rollout_adapter_sha256,
            "group/identity rollout adapter mismatch",
        )
        _require(
            group.get("adapter_semantic_sha256")
            == identity.input_adapter_semantic_sha256,
            "group/identity adapter semantic mismatch",
        )
    expected_tokens = sum(trajectory["generated_action_tokens"] for trajectory in trajectories)
    _require(group.get("generated_action_tokens") == expected_tokens, "group token count mismatch")
    expected_hash = group_content_sha256(group)
    _require(group.get("group_sha256") == expected_hash, "group content hash mismatch")
    return dict(group)
