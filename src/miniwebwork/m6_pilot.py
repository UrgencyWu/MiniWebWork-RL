"""Auditable authorization for the approved M6.1 156-task mini pilot.

The original M6 protocol and failed 160-task corpus audit remain immutable.
This module permits only the exact two-collection, 156-task development pilot
approved after that failure; it cannot authorize formal training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .long_horizon_rl.contracts import SHA256_PATTERN, sha256_file, sha256_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAIVER_PATH = PROJECT_ROOT / "data" / "m6_mini_pilot_waiver_v1.json"
REPLAY_PARITY_CALIBRATION_PATH = (
    PROJECT_ROOT / "data" / "m6_mini_replay_parity_calibration_v1.json"
)
WAIVER_SCHEMA = "m6_mini_pilot_waiver_v1"
AUTHORIZATION_SCHEMA = "m6_mini_pilot_authorization_v1"
REPLAY_PARITY_CALIBRATION_SCHEMA = "m6_mini_replay_parity_calibration_v1"
APPROVED_METHODS = ("multi_turn_grpo", "anchor_gigpo")
SFT_AUDIT_INPUT_FILES = (
    "corpus.json",
    "corpus_audit.json",
    "train.jsonl",
    "dev.jsonl",
    "retention.json",
    "pilot_authorization.json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sft_audit_input_key(filename: str) -> str:
    """Return the exact key used when the M6 corpus token audit was built."""

    _require(filename in SFT_AUDIT_INPUT_FILES, "M6 SFT audit input filename drift")
    return Path(filename).stem


def validate_sft_corpus_git_compatibility(
    *,
    corpus_producer_git_sha: Any,
    consumer_git_sha: str,
    explicitly_authorized_producer_git_sha: str | None,
) -> str:
    """Fail closed when an existing corpus crosses a repair-only Git boundary."""

    producer = str(corpus_producer_git_sha)
    for value, label in ((producer, "producer"), (consumer_git_sha, "consumer")):
        _require(
            len(value) == 40 and all(character in "0123456789abcdef" for character in value),
            f"M6 SFT corpus {label} Git drift",
        )
    if producer != consumer_git_sha:
        _require(
            explicitly_authorized_producer_git_sha == producer,
            "M6 corpus producer/consumer Git compatibility was not explicitly authorized",
        )
    elif explicitly_authorized_producer_git_sha is not None:
        _require(
            explicitly_authorized_producer_git_sha == producer,
            "M6 redundant corpus producer Git authorization drift",
        )
    return producer


def validate_artifact_git_compatibility(
    *,
    artifact_name: str,
    producer_git_sha: Any,
    consumer_git_sha: str,
    explicitly_authorized_producer_git_sha: str | None,
) -> str:
    """Fail closed when a derived artifact crosses a repair-only Git boundary."""

    _require(isinstance(artifact_name, str) and artifact_name, "M6 artifact name is missing")
    producer = str(producer_git_sha)
    for value, label in ((producer, "producer"), (consumer_git_sha, "consumer")):
        _require(
            len(value) == 40 and all(character in "0123456789abcdef" for character in value),
            f"M6 {artifact_name} {label} Git drift",
        )
    if producer != consumer_git_sha:
        _require(
            explicitly_authorized_producer_git_sha == producer,
            f"M6 {artifact_name} producer/consumer Git compatibility was not explicitly authorized",
        )
    elif explicitly_authorized_producer_git_sha is not None:
        _require(
            explicitly_authorized_producer_git_sha == producer,
            f"M6 redundant {artifact_name} producer Git authorization drift",
        )
    return producer


def validate_pilot_waiver(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == WAIVER_SCHEMA, "M6 pilot waiver schema drift")
    _require(value.get("study_id") == "m6_monotonic_posttraining_v1", "M6 pilot study drift")
    _require(value.get("scope") == "development_only_mini_pilot", "M6 pilot scope drift")
    _require(value.get("decision") == "ALLOW_156_TASK_MINI_PILOT", "M6 pilot decision drift")
    _require(value.get("formal_training_allowed") is False, "M6 pilot cannot authorize formal training")
    _require(value.get("original_minimum_success_tasks") == 160, "M6 original task gate drift")
    _require(value.get("authorized_minimum_success_tasks") == 156, "M6 pilot task gate drift")
    _require(value.get("observed_success_tasks") == 156, "M6 pilot observed task count drift")
    _require(value.get("observed_replay_success_trajectories") == 493, "M6 pilot trajectory count drift")
    _require(value.get("observed_completion_label_tokens") == 31362, "M6 pilot token count drift")
    _require(value.get("only_waived_check") == "mini_success_task_count", "M6 pilot waived-check drift")
    _require(
        SHA256_PATTERN.fullmatch(str(value.get("source_protocol_sha256", ""))) is not None,
        "M6 pilot source protocol SHA drift",
    )
    producer = str(value.get("source_producer_git_sha", ""))
    _require(len(producer) == 40 and all(character in "0123456789abcdef" for character in producer), "M6 pilot producer Git drift")
    collection_hashes = value.get("source_collection_report_content_sha256")
    _require(
        isinstance(collection_hashes, list)
        and len(collection_hashes) == 2
        and len(set(collection_hashes)) == 2
        and all(SHA256_PATTERN.fullmatch(str(item)) is not None for item in collection_hashes),
        "M6 pilot collection hashes drift",
    )
    _require(value.get("source_collection_seeds") == [20260812, 20260813], "M6 pilot collection seeds drift")
    _require(
        SHA256_PATTERN.fullmatch(str(value.get("failed_corpus_audit_content_sha256", ""))) is not None,
        "M6 pilot failed-audit SHA drift",
    )
    methods = value.get("approved_rl_methods")
    _require(isinstance(methods, Mapping) and tuple(methods) == APPROVED_METHODS, "M6 pilot RL method roster drift")
    _require(methods["multi_turn_grpo"].get("verifier_td_lambda") == 0.0, "M6 GRPO credit weight drift")
    _require(methods["anchor_gigpo"].get("verifier_td_lambda") == 0.5, "M6 anchor credit weight drift")
    controls = value.get("shared_rl_controls")
    _require(
        controls
        == {
            "group_size": 8,
            "learning_rate": 0.000001,
            "policy_epochs": 1,
            "frozen_sft_reference": True,
            "same_curriculum": True,
            "same_action_token_budget": True,
        },
        "M6 pilot shared RL controls drift",
    )
    return value


def load_pilot_waiver(path: Path = WAIVER_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "M6 pilot waiver root must be an object")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": validate_pilot_waiver(payload)}


def build_pilot_authorization(
    *,
    corpus_audit: Mapping[str, Any],
    source_collections: Sequence[Mapping[str, Any]],
    waiver_path: Path = WAIVER_PATH,
) -> dict[str, Any]:
    from .webshop_rl.m6_corpus import validate_conditional_learnability_audit

    waiver = load_pilot_waiver(waiver_path)
    contract = waiver["payload"]
    audit = validate_conditional_learnability_audit(corpus_audit)
    checks = audit["checks"]
    _require(audit.get("passed") is False, "M6 pilot waiver is unnecessary for a passing corpus")
    _require(
        checks.get("mini_success_task_count") is False
        and all(bool(value) for key, value in checks.items() if key != "mini_success_task_count"),
        "M6 pilot may waive only the success-task count",
    )
    metrics = audit["metrics"]
    _require(audit["content_sha256"] == contract["failed_corpus_audit_content_sha256"], "M6 pilot corpus-audit binding drift")
    _require(metrics.get("task_count") == contract["observed_success_tasks"], "M6 pilot corpus task count drift")
    _require(metrics.get("trajectory_count") == contract["observed_replay_success_trajectories"], "M6 pilot corpus trajectory count drift")
    _require(metrics.get("completion_label_tokens") == contract["observed_completion_label_tokens"], "M6 pilot corpus token count drift")
    _require(len(source_collections) == 2, "M6 pilot requires the two approved Raw collections")
    observed_hashes = [item.get("collection_report_content_sha256") for item in source_collections]
    observed_seeds = [item.get("seed") for item in source_collections]
    _require(observed_hashes == contract["source_collection_report_content_sha256"], "M6 pilot source collection drift")
    _require(observed_seeds == contract["source_collection_seeds"], "M6 pilot source seed drift")
    _require(
        all(item.get("git_sha") == contract["source_producer_git_sha"] for item in source_collections),
        "M6 pilot source producer Git drift",
    )
    _require(
        all(item.get("protocol_sha256") == contract["source_protocol_sha256"] for item in source_collections),
        "M6 pilot source protocol drift",
    )
    report = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "study_id": contract["study_id"],
        "development_only": True,
        "formal_training_allowed": False,
        "passed": True,
        "decision": contract["decision"],
        "waiver_file_sha256": waiver["sha256"],
        "original_minimum_success_tasks": contract["original_minimum_success_tasks"],
        "authorized_minimum_success_tasks": contract["authorized_minimum_success_tasks"],
        "observed_success_tasks": metrics["task_count"],
        "observed_replay_success_trajectories": metrics["trajectory_count"],
        "observed_completion_label_tokens": metrics["completion_label_tokens"],
        "only_waived_check": contract["only_waived_check"],
        "failed_corpus_audit_content_sha256": audit["content_sha256"],
        "source_protocol_sha256": contract["source_protocol_sha256"],
        "source_producer_git_sha": contract["source_producer_git_sha"],
        "source_collection_report_content_sha256": observed_hashes,
        "source_collection_seeds": observed_seeds,
        "approved_rl_methods": contract["approved_rl_methods"],
        "shared_rl_controls": contract["shared_rl_controls"],
    }
    report["content_sha256"] = sha256_json(report)
    return report


def validate_pilot_authorization(
    payload: Mapping[str, Any],
    *,
    corpus_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = dict(payload)
    waiver = load_pilot_waiver()
    contract = waiver["payload"]
    _require(value.get("schema_version") == AUTHORIZATION_SCHEMA, "M6 pilot authorization schema drift")
    _require(value.get("development_only") is True and value.get("formal_training_allowed") is False, "M6 pilot authorization scope drift")
    _require(value.get("passed") is True and value.get("decision") == "ALLOW_156_TASK_MINI_PILOT", "M6 pilot authorization decision drift")
    _require(value.get("authorized_minimum_success_tasks") == 156 and value.get("observed_success_tasks") == 156, "M6 pilot authorized task count drift")
    _require(value.get("original_minimum_success_tasks") == 160, "M6 pilot authorization original task gate drift")
    _require(value.get("observed_replay_success_trajectories") == 493, "M6 pilot authorization trajectory count drift")
    _require(value.get("observed_completion_label_tokens") == 31362, "M6 pilot authorization token count drift")
    _require(value.get("only_waived_check") == "mini_success_task_count", "M6 pilot authorization waived-check drift")
    _require(value.get("waiver_file_sha256") == waiver["sha256"], "M6 pilot authorization waiver-file drift")
    for field in (
        "failed_corpus_audit_content_sha256",
        "source_protocol_sha256",
        "source_producer_git_sha",
        "source_collection_report_content_sha256",
        "source_collection_seeds",
        "approved_rl_methods",
        "shared_rl_controls",
    ):
        _require(value.get(field) == contract.get(field), f"M6 pilot authorization {field} drift")
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 pilot authorization self-hash drift")
    if corpus_audit is not None:
        _require(
            value.get("failed_corpus_audit_content_sha256") == corpus_audit.get("content_sha256"),
            "M6 pilot authorization/corpus binding drift",
        )
    return value


def validate_pilot_method(method: str, authorization: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_pilot_authorization(authorization)
    _require(method in APPROVED_METHODS, "M6 pilot RL method is not approved")
    _require(method in value["approved_rl_methods"], "M6 pilot authorization method drift")
    return dict(value["approved_rl_methods"][method])


def validate_pilot_replay_parity_calibration(
    payload: Mapping[str, Any],
    *,
    protocol_sha256: str,
    parity_contract: Mapping[str, Any],
    learner_microbatch_size: int,
    frozen_sft_adapter_sha256: str,
    frozen_sft_adapter_semantic_sha256: str,
) -> dict[str, Any]:
    """Validate the narrow, development-only vLLM/HF P99 calibration.

    This does not revise the M6 protocol.  It permits one empirically calibrated
    P99 threshold for the 156-task mini pilot while every independent parity
    guardrail remains byte-for-byte equal to the frozen protocol contract.
    """

    value = dict(payload)
    _require(
        value.get("schema_version") == REPLAY_PARITY_CALIBRATION_SCHEMA,
        "M6 replay calibration schema drift",
    )
    _require(
        value.get("study_id") == "m6_monotonic_posttraining_v1",
        "M6 replay calibration study drift",
    )
    _require(
        value.get("scope") == "development_only_156_task_mini_pilot"
        and value.get("development_only") is True
        and value.get("formal_training_allowed") is False,
        "M6 replay calibration scope drift",
    )
    _require(
        value.get("only_adjusted_check") == "replay_p99_absolute_difference",
        "M6 replay calibration adjusted more than P99",
    )
    _require(
        value.get("source_protocol_sha256") == protocol_sha256,
        "M6 replay calibration protocol drift",
    )
    base_contract = value.get("base_parity_contract")
    _require(
        isinstance(base_contract, Mapping) and dict(base_contract) == dict(parity_contract),
        "M6 replay calibration base-contract drift",
    )
    _require(
        value.get("learner_microbatch_size") == learner_microbatch_size == 4,
        "M6 replay calibration microbatch drift",
    )
    _require(
        value.get("logprob_precision") == "float32",
        "M6 replay calibration precision drift",
    )
    _require(
        value.get("frozen_sft_adapter_sha256") == frozen_sft_adapter_sha256
        and value.get("frozen_sft_adapter_semantic_sha256")
        == frozen_sft_adapter_semantic_sha256,
        "M6 replay calibration frozen-SFT drift",
    )

    source = value.get("source_probe")
    _require(isinstance(source, Mapping), "M6 replay calibration source probe is missing")
    _require(
        source.get("expected_K") == 4
        and source.get("group_count") == 200
        and int(source.get("token_count", 0)) > 0,
        "M6 replay calibration requires the complete frozen K4 dev probe",
    )
    _require(
        source.get("microbatch_size") == learner_microbatch_size
        and source.get("logprob_precision") == "float32"
        and source.get("training_updates_performed") == 0,
        "M6 replay calibration source execution drift",
    )
    for field in (
        "content_sha256",
        "collection_report_content_sha256",
        "input_adapter_sha256",
        "input_adapter_semantic_sha256",
    ):
        _require(
            SHA256_PATTERN.fullmatch(str(source.get(field, ""))) is not None,
            f"M6 replay calibration source {field} drift",
        )
    _require(
        source.get("input_adapter_sha256") == frozen_sft_adapter_sha256
        and source.get("input_adapter_semantic_sha256")
        == frozen_sft_adapter_semantic_sha256,
        "M6 replay calibration source adapter drift",
    )
    _require(
        source.get("protocol_sha256") == protocol_sha256,
        "M6 replay calibration source protocol drift",
    )
    for field in ("producer_git_sha", "consumer_git_sha"):
        git_sha = str(source.get(field, ""))
        _require(
            len(git_sha) == 40
            and all(character in "0123456789abcdef" for character in git_sha),
            f"M6 replay calibration source {field} drift",
        )
    checks = source.get("checks")
    _require(
        isinstance(checks, Mapping)
        and set(checks)
        == {
            "mean_absolute_difference",
            "p95_absolute_difference",
            "p99_absolute_difference",
            "p999_absolute_difference",
            "initial_ratio_clip_fraction",
            "mean_importance_ratio",
        }
        and checks.get("p99_absolute_difference") is False
        and all(
            bool(result)
            for name, result in checks.items()
            if name != "p99_absolute_difference"
        ),
        "M6 replay calibration source did not isolate P99",
    )
    parity = source.get("parity")
    _require(isinstance(parity, Mapping), "M6 replay calibration source parity is missing")
    _require(
        float(parity.get("mean_absolute_logprob_difference", math.inf))
        <= float(parity_contract["replay_mean_absolute_difference"])
        and float(parity.get("p95_absolute_logprob_difference", math.inf))
        <= float(parity_contract["replay_p95_absolute_difference"])
        and float(parity.get("p999_absolute_logprob_difference", math.inf))
        <= float(parity_contract["replay_p999_absolute_difference"])
        and float(parity.get("initial_ratio_clip_fraction", math.inf))
        <= float(parity_contract["replay_initial_ratio_clip_fraction"])
        and abs(float(parity.get("mean_importance_ratio", math.inf)) - 1.0)
        <= float(parity_contract["mean_importance_ratio_absolute_deviation"]),
        "M6 replay calibration source violated an independent guardrail",
    )
    observed_p99 = float(parity.get("p99_absolute_logprob_difference", math.nan))
    _require(
        math.isfinite(observed_p99)
        and observed_p99 > float(parity_contract["replay_p99_absolute_difference"])
        and observed_p99 <= float(parity_contract["replay_p999_absolute_difference"]),
        "M6 replay calibration observed P99 is invalid",
    )
    _require(
        int(parity.get("token_count", 0)) == int(source["token_count"]),
        "M6 replay calibration token count drift",
    )

    grid = float(value.get("calibration_grid", math.nan))
    calibrated_p99 = float(
        value.get("calibrated_replay_p99_absolute_difference", math.nan)
    )
    _require(
        value.get("calibration_rule") == "ceil_full_k4_p99_to_next_0.025"
        and math.isclose(grid, 0.025, abs_tol=1e-12),
        "M6 replay calibration rule drift",
    )
    expected_p99 = math.ceil((observed_p99 - 1e-12) / grid) * grid
    _require(
        math.isclose(calibrated_p99, expected_p99, abs_tol=1e-12)
        and calibrated_p99 <= float(parity_contract["replay_p999_absolute_difference"]),
        "M6 replay calibration P99 threshold drift",
    )
    effective = value.get("effective_parity_contract")
    expected_effective = dict(parity_contract)
    expected_effective["replay_p99_absolute_difference"] = calibrated_p99
    _require(
        isinstance(effective, Mapping) and dict(effective) == expected_effective,
        "M6 replay calibration changed an independent guardrail",
    )
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 replay calibration self-hash drift")
    return value


def load_pilot_replay_parity_calibration(
    path: Path = REPLAY_PARITY_CALIBRATION_PATH,
    *,
    protocol_sha256: str,
    parity_contract: Mapping[str, Any],
    learner_microbatch_size: int,
    frozen_sft_adapter_sha256: str,
    frozen_sft_adapter_semantic_sha256: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "M6 replay calibration root must be an object")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "payload": validate_pilot_replay_parity_calibration(
            payload,
            protocol_sha256=protocol_sha256,
            parity_contract=parity_contract,
            learner_microbatch_size=learner_microbatch_size,
            frozen_sft_adapter_sha256=frozen_sft_adapter_sha256,
            frozen_sft_adapter_semantic_sha256=frozen_sft_adapter_semantic_sha256,
        ),
    }
