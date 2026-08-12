"""Fail-closed contracts for the M6 Raw -> SFT -> RL study."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .long_horizon_rl.contracts import sha256_file, sha256_json
from .m5_webshop_protocol import (
    eligible_goal_indices,
    normalized_instruction,
    task_id_for_goal_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "data" / "m6_posttraining_protocol_v1.json"
SCHEMA_VERSION = "m6_posttraining_protocol_v1"
STUDY_ID = "m6_monotonic_posttraining_v1"
SPLIT_SCHEMA = "m6_webshop_split_v1"
EXPOSURE_SCHEMA = "m5_goal_exposure_registry_v1"
MINI_GATE_SCHEMA = "m6_mini_gate_report_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{field} must be finite",
    )
    return float(value)


def audit_behavior_sampling_logprobs(
    behavior: Sequence[float],
    sampling: Sequence[float],
    *,
    maximum_absolute_difference: float,
) -> dict[str, Any]:
    """Fail closed unless stored behavior and actual sampling log-probs agree."""

    threshold = _finite_number(
        maximum_absolute_difference,
        "M6 behavior/sampling maximum absolute difference",
    )
    _require(threshold > 0.0, "M6 behavior/sampling threshold must be positive")
    _require(bool(behavior) and len(behavior) == len(sampling), "M6 behavior/sampling vectors differ")
    differences = [
        abs(
            _finite_number(left, "M6 behavior log-prob")
            - _finite_number(right, "M6 sampling log-prob")
        )
        for left, right in zip(behavior, sampling)
    ]
    report = {
        "token_count": len(differences),
        "mean_absolute_logprob_difference": sum(differences) / len(differences),
        "maximum_absolute_logprob_difference": max(differences),
        "threshold": threshold,
        "passed": max(differences) <= threshold,
    }
    _require(report["passed"] is True, f"M6 behavior/sampling parity failed: {report}")
    return report


def repository_git_sha() -> str:
    value = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(GIT_SHA_RE.fullmatch(value) is not None, "invalid M6 repository Git SHA")
    return value


def validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == SCHEMA_VERSION, "M6 protocol schema drift")
    _require(value.get("protocol_revision") == 1, "M6 protocol revision drift")
    _require(value.get("study_id") == STUDY_ID, "M6 study identity drift")
    _require(value.get("status") == "implementation_only", "M6 protocol left implementation-only state")
    _require(value.get("formal_submission_allowed") is False, "M6 protocol cannot authorize formal submission")

    goal = value.get("primary_goal")
    _require(isinstance(goal, Mapping), "M6 primary goal is missing")
    _require(goal.get("metric") == "strict task success iff official task_score >= 0.999", "M6 primary metric drift")
    _require(goal.get("required_chain") == "raw < sft < rl", "M6 monotonic goal drift")
    _require(goal.get("minimum_sft_minus_raw_pp") == 3.0, "M6 SFT target drift")
    _require(goal.get("minimum_rl_minus_sft_pp") == 3.0, "M6 RL target drift")

    split = value.get("split")
    _require(isinstance(split, Mapping), "M6 split contract is missing")
    _require(
        split.get("source_start_inclusive") == 1000
        and split.get("source_stop_exclusive") == 12087,
        "M6 source roster drift",
    )
    _require(split.get("mini_train_tasks") == 256, "M6 mini-train count drift")
    _require(split.get("mini_dev_tasks") == 200, "M6 mini-dev count drift")
    _require(split.get("formal_dev_tasks") == 500, "M6 formal-dev count drift")
    candidates = split.get("evaluation_task_candidates")
    _require(candidates == [1000, 1500, 2000], "M6 evaluation-size candidates drift")
    _require(split.get("selection_seed") == 20260812, "M6 split seed drift")
    _require(split.get("minimum_remaining_train_tasks") == 2000, "M6 remaining-train gate drift")
    _require(split.get("promotion_and_holdout_must_be_never_read") is True, "M6 fresh-test gate disabled")
    _require(split.get("normalized_instruction_overlap_allowed") is False, "M6 overlap gate disabled")

    mini = value.get("mini")
    _require(isinstance(mini, Mapping) and mini.get("development_only") is True, "M6 mini scope drift")
    _require(mini.get("raw_collection_K") == 8 and mini.get("maximum_raw_collection_K") == 16, "M6 mini collection drift")
    _require(mini.get("evaluation_K") == 4, "M6 mini evaluation K drift")
    _require(mini.get("maximum_rl_generated_action_tokens") == 50000, "M6 mini RL budget drift")
    _require(mini.get("rl_collection_groups_per_iteration") == 1, "M6 mini RL group budget drift")
    _require(mini.get("rl_collection_max_model_turns") == 6, "M6 mini RL model-turn cap drift")
    _require(mini.get("rl_collection_max_environment_steps") == 6, "M6 mini RL environment-step cap drift")
    _require(mini.get("rl_collection_action_token_cap_per_iteration") == 12288, "M6 mini RL iteration-token cap drift")
    _require(mini.get("maximum_rl_iterations") == 12, "M6 mini RL iteration-count drift")
    _require(mini.get("minimum_success_tasks") == 160, "M6 mini success-task gate drift")
    _require(mini.get("minimum_success_trajectories") == 320, "M6 mini trajectory gate drift")
    _require(mini.get("minimum_completion_label_tokens") == 20000, "M6 mini label minimum drift")
    _require(mini.get("maximum_completion_label_tokens") == 80000, "M6 mini label maximum drift")
    _require(mini.get("minimum_mixed_rl_iterations") == 5, "M6 mini mixed-iteration gate drift")
    _require(mini.get("minimum_optimizer_updates") == 2, "M6 mini optimizer-update gate drift")
    _require(mini.get("minimum_sft_minus_raw_pp") == 3.0, "M6 mini SFT delta drift")
    _require(mini.get("minimum_rl_minus_sft_pp") == 3.0, "M6 mini RL delta drift")
    _require(mini.get("minimum_bootstrap_positive_fraction") == 0.8, "M6 mini bootstrap gate drift")
    _require(mini.get("maximum_sft_search_exhaustion_increase_pp") == 5.0, "M6 mini search guardrail drift")
    _require(mini.get("maximum_rl_partial_match_increase_pp") == 5.0, "M6 mini partial-match guardrail drift")
    _require(mini.get("maximum_stage_schema_or_action_error_increase_pp") == 3.0, "M6 mini action guardrail drift")
    _require(mini.get("failed_dev_roster_policy") == "burn_and_replace", "M6 mini reuse policy drift")

    corpus = value.get("corpus")
    _require(isinstance(corpus, Mapping), "M6 corpus contract is missing")
    _require(corpus.get("strict_success_fraction") == 1.0, "M6 strict corpus gate drift")
    _require(corpus.get("replay_success_fraction") == 1.0, "M6 replay corpus gate drift")
    _require(corpus.get("maximum_hidden_field_count") == 0, "M6 hidden-field gate drift")
    _require(corpus.get("maximum_target_asin_search_label_count") == 0, "M6 target-ASIN gate drift")
    _require(corpus.get("minimum_public_query_token_fraction") == 0.9, "M6 query provenance gate drift")
    _require(corpus.get("maximum_mean_search_query_characters") == 48, "M6 mean-query gate drift")
    _require(corpus.get("maximum_p95_search_query_characters") == 96, "M6 p95-query gate drift")
    _require(corpus.get("minimum_unique_command_sequence_fraction") == 0.6, "M6 sequence-diversity gate drift")
    _require(corpus.get("minimum_recovery_trajectory_fraction") == 0.2, "M6 recovery gate drift")
    _require(corpus.get("maximum_action_family_fraction") == 0.35, "M6 action-balance gate drift")
    _require(corpus.get("maximum_trajectories_per_task") == 4, "M6 per-task trajectory cap drift")
    _require(corpus.get("minimum_formal_success_tasks") == 2000, "M6 formal success-task gate drift")
    _require(corpus.get("minimum_formal_trajectories") == 4000, "M6 formal trajectory minimum drift")
    _require(corpus.get("maximum_formal_trajectories") == 8000, "M6 formal trajectory maximum drift")
    _require(corpus.get("minimum_formal_completion_label_tokens") == 250000, "M6 formal label minimum drift")
    _require(corpus.get("maximum_formal_completion_label_tokens") == 600000, "M6 formal label maximum drift")

    sft = value.get("sft")
    _require(isinstance(sft, Mapping), "M6 SFT contract is missing")
    _require(sft.get("base_model") == "/data/share/model/Qwen3.5-4B", "M6 base model drift")
    _require(sft.get("prompt_contract") == "webshop_agent_v1_compact", "M6 prompt contract drift")
    _require(sft.get("maximum_sequence_tokens") == 8192, "M6 SFT context drift")
    _require(sft.get("effective_batch_size") == 10, "M6 SFT effective batch drift")
    _require(sft.get("microbatch_candidates") == [1, 2, 5], "M6 SFT microbatch candidates drift")
    _require(sft.get("minimum_vram_headroom") == 0.15, "M6 SFT VRAM headroom drift")
    _require(sft.get("chat_template_kwargs") == {"enable_thinking": False}, "M6 SFT chat template drift")
    _require(sft.get("maximum_epochs") == 1, "M6 SFT epoch drift")
    _require(sft.get("learning_rate_candidates") == [0.00002, 0.00005], "M6 SFT LR drift")
    _require(sft.get("raw_reference_kl_candidates") == [0.01, 0.03], "M6 SFT KL grid drift")
    _require(sft.get("mini_learning_rate") == 0.00002, "M6 mini SFT LR drift")
    _require(sft.get("mini_raw_reference_kl") == 0.03, "M6 mini SFT KL drift")
    _require(
        sft.get("raw_reference_kl_direction")
        == "KL(raw_policy||sft_policy)_sampled_on_raw_actions",
        "M6 SFT sampled KL direction drift",
    )
    _require(sft.get("success_imitation_fraction") == 0.9, "M6 SFT imitation fraction drift")
    _require(sft.get("retention_state_fraction") == 0.1, "M6 SFT retention fraction drift")
    _require(
        math.isclose(
            float(sft["success_imitation_fraction"]) + float(sft["retention_state_fraction"]),
            1.0,
            abs_tol=1e-12,
        ),
        "M6 SFT batch fractions do not sum to one",
    )
    _require(
        all(int(sft["effective_batch_size"]) % int(item) == 0 for item in sft["microbatch_candidates"]),
        "M6 SFT microbatch does not divide effective batch",
    )
    _require(
        sft.get("lora")
        == {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "M6 SFT LoRA contract drift",
    )

    rl = value.get("rl")
    _require(isinstance(rl, Mapping), "M6 RL contract is missing")
    _require(rl.get("method") == "strict_grpo_verifier_td", "M6 RL method drift")
    _require(rl.get("group_size") == 8, "M6 RL K drift")
    _require(rl.get("terminal_reward") == "1[official task_score >= 0.999]", "M6 terminal reward drift")
    _require(rl.get("verifier_td_lambda") == 0.5, "M6 verifier-TD weight drift")
    _require(rl.get("learning_rate") == 0.000001, "M6 RL learning-rate drift")
    _require(rl.get("policy_epochs") == 1, "M6 RL policy-epoch drift")
    _require(rl.get("telescoping_absolute_tolerance") == 0.00000001, "M6 RL telescoping tolerance drift")
    _require(rl.get("minimum_nonzero_credit_turn_fraction") == 0.3, "M6 RL credit-coverage gate drift")
    _require(rl.get("minimum_mixed_group_fraction") == 0.2, "M6 RL mixed-group gate drift")
    _require(rl.get("adaptive_kl_target_minimum") == 0.005, "M6 RL KL minimum drift")
    _require(rl.get("adaptive_kl_target_maximum") == 0.03, "M6 RL KL maximum drift")
    parity = rl.get("parity_contract")
    _require(isinstance(parity, Mapping), "M6 RL parity contract is missing")
    _require(
        parity
        == {
            "behavior_sampling_maximum_absolute_difference": 0.000001,
            "replay_mean_absolute_difference": 0.02,
            "replay_p95_absolute_difference": 0.08,
            "replay_p99_absolute_difference": 0.1,
            "replay_p999_absolute_difference": 0.5,
            "replay_initial_ratio_clip_fraction": 0.005,
            "mean_importance_ratio_absolute_deviation": 0.02,
        },
        "M6 RL parity thresholds drift",
    )
    _require(rl.get("formal_generated_action_token_cap_per_seed") == 500000, "M6 formal RL budget drift")
    _require(rl.get("formal_seeds") == [20260812, 20260813, 20260814], "M6 formal RL seeds drift")

    slurm = value.get("slurm")
    _require(isinstance(slurm, Mapping), "M6 Slurm contract is missing")
    _require(slurm.get("wall_time_per_job") == "24:00:00", "M6 wall-time drift")
    _require(slurm.get("maximum_parallel_gpu_jobs") == 4, "M6 parallel GPU limit drift")
    _require(slurm.get("training_per_job") == {"gpus": 1, "cpus": 8, "memory_gib": 32}, "M6 training resources drift")
    _require(slurm.get("evaluation_per_job") == {"gpus": 1, "cpus": 4, "memory_gib": 24}, "M6 evaluation resources drift")
    _require(
        slurm.get("environment_service_per_job") == {"gpus": 0, "cpus": 8, "memory_gib": 48},
        "M6 environment-service resources drift",
    )
    _require(slurm.get("deterministic_failure_successor_allowed") is False, "M6 deterministic retry gate disabled")
    _require(slurm.get("timeout_recovery") == "same_root_successor_only", "M6 timeout-recovery contract drift")
    return value


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "M6 protocol root must be an object")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "git_sha": repository_git_sha(),
        "payload": validate_protocol(payload),
    }


def _compact_indices_sha256(indices: Sequence[int]) -> str:
    values = [int(index) for index in indices]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ranked(candidates: Sequence[int], *, seed: int, namespace: str) -> list[int]:
    unique = sorted(set(candidates))
    _require(len(unique) == len(candidates), f"M6 candidate roster has duplicates: {namespace}")
    return sorted(
        unique,
        key=lambda index: (
            hashlib.sha256(f"{namespace}|{seed}|{index}".encode("utf-8")).digest(),
            index,
        ),
    )


def validate_exposure_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == EXPOSURE_SCHEMA, "M6 exposure registry schema drift")
    exposures = value.get("exposures")
    _require(isinstance(exposures, list), "M6 exposure registry rows are missing")
    indices: list[int] = []
    for row in exposures:
        _require(isinstance(row, Mapping), "M6 exposure registry row is malformed")
        index = row.get("goal_index")
        _require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < 12087, "M6 exposure goal index drift")
        _require(row.get("task_id") == task_id_for_goal_index(index), "M6 exposure task id drift")
        _require(row.get("ever_read") is True, "M6 registry may contain only exposed goals")
        reasons = row.get("reasons")
        evidence = row.get("evidence_paths")
        _require(isinstance(reasons, list) and reasons and all(isinstance(item, str) and item for item in reasons), "M6 exposure reason missing")
        _require(isinstance(evidence, list) and evidence and all(isinstance(item, str) and item for item in evidence), "M6 exposure evidence missing")
        indices.append(index)
    _require(indices == sorted(set(indices)), "M6 exposure registry is not unique and sorted")
    _require(value.get("exposed_goal_count") == len(indices), "M6 exposure count drift")
    _require(value.get("exposed_goal_index_sha256") == _compact_indices_sha256(indices), "M6 exposure hash drift")
    scanned = value.get("scanned_files")
    _require(isinstance(scanned, list) and scanned, "M6 exposure scan inventory is missing")
    _require(value.get("scanned_file_count") == len(scanned), "M6 exposure scan count drift")
    for item in scanned:
        _require(isinstance(item, Mapping), "M6 exposure scan row is malformed")
        _require(isinstance(item.get("path"), str) and item["path"], "M6 exposure scan path is missing")
        _require(SHA256_RE.fullmatch(str(item.get("sha256"))) is not None, "M6 exposure scan SHA drift")
        _require(isinstance(item.get("matched_goal_count"), int), "M6 exposure scan count is invalid")
    expected = dict(value)
    observed_hash = expected.pop("content_sha256", None)
    _require(observed_hash == sha256_json(expected), "M6 exposure registry self-hash drift")
    return value


def build_exposure_registry(
    *,
    evidence_files: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    """Extract task-specific M5 exposure from explicitly scoped evidence.

    The caller must pass only result-bearing corpus, preflight, online,
    frozen-evaluation and report files.  Merely loading ``goals.json`` or a
    split roster is intentionally not exposure and is therefore never scanned
    implicitly.
    """

    _require(bool(evidence_files), "M5 exposure evidence is empty")
    rows: dict[int, dict[str, set[str]]] = {}
    task_pattern = re.compile(r"\bwebshop_goal_([0-9]{5})\b")
    goal_index_pattern = re.compile(
        r"[\"']?goal[_ -]?index[\"']?\s*[:=]\s*[\"']?([0-9]{1,5})",
        re.IGNORECASE,
    )
    scanned_files: list[dict[str, Any]] = []
    allowed_reasons = {
        "sft_corpus",
        "online_preflight",
        "online_training",
        "frozen_evaluation",
        "trajectory_diagnostic",
        "technical_report",
    }
    for reason, raw_path in sorted(
        evidence_files,
        key=lambda item: (str(item[0]), str(Path(item[1]).expanduser().resolve())),
    ):
        _require(reason in allowed_reasons, f"unsupported M5 exposure reason: {reason}")
        path = Path(raw_path).expanduser().resolve()
        _require(path.is_file(), f"M5 exposure evidence is missing: {path}")
        _require(path.stat().st_size <= 512 * 1024 * 1024, f"M5 exposure evidence is unexpectedly large: {path}")
        text = path.read_text(encoding="utf-8", errors="strict")
        indices = sorted(
            {int(match.group(1)) for match in task_pattern.finditer(text)}
            | {int(match.group(1)) for match in goal_index_pattern.finditer(text)}
        )
        for index in indices:
            _require(0 <= index < 12087, f"M5 exposure task id is outside goal roster: {index}")
            item = rows.setdefault(index, {"reasons": set(), "evidence_paths": set()})
            item["reasons"].add(reason)
            item["evidence_paths"].add(str(path))
        scanned_files.append(
            {
                "reason": reason,
                "path": str(path),
                "sha256": sha256_file(path),
                "matched_goal_count": len(indices),
            }
        )
    indices = sorted(rows)
    registry = {
        "schema_version": EXPOSURE_SCHEMA,
        "study_id": STUDY_ID,
        "scope_contract": "task-specific M5 result evidence only; runtime goal loading and roster identity are not exposure",
        "scanned_files": scanned_files,
        "scanned_file_count": len(scanned_files),
        "exposures": [
            {
                "goal_index": index,
                "task_id": task_id_for_goal_index(index),
                "ever_read": True,
                "reasons": sorted(rows[index]["reasons"]),
                "evidence_paths": sorted(rows[index]["evidence_paths"]),
            }
            for index in indices
        ],
        "exposed_goal_count": len(indices),
        "exposed_goal_index_sha256": _compact_indices_sha256(indices),
    }
    registry["content_sha256"] = sha256_json(registry)
    return validate_exposure_registry(registry)


def build_split_lock(
    *,
    goals: Sequence[Mapping[str, Any]],
    exposure_registry: Mapping[str, Any],
    n_eval: int,
    power_report_content_sha256: str | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = load_protocol() if protocol is None else None
    contract = dict(protocol or loaded["payload"])
    protocol_sha256 = loaded["sha256"] if loaded is not None else sha256_json(contract)
    validate_protocol(contract)
    registry = validate_exposure_registry(exposure_registry)
    _require(len(goals) == 12087, "M6 goal roster count drift")
    for index, goal in enumerate(goals):
        _require(isinstance(goal, Mapping) and goal.get("goal_index") == index, f"M6 goal row drift: {index}")
        _require(isinstance(goal.get("instruction"), str) and goal["instruction"].strip(), f"M6 instruction missing: {index}")

    split = contract["split"]
    _require(n_eval in split["evaluation_task_candidates"], "M6 N_eval is not a frozen candidate")
    _require(
        power_report_content_sha256 is None
        or SHA256_RE.fullmatch(power_report_content_sha256) is not None,
        "M6 power-report binding drift",
    )
    eligible = list(eligible_goal_indices("train"))
    exposed = {int(row["goal_index"]) for row in registry["exposures"]}
    fresh = [index for index in eligible if index not in exposed]
    _require(len(fresh) >= 2 * n_eval, "M6 fresh roster cannot cover promotion and holdout")
    seed = int(split["selection_seed"])

    fresh_order = _ranked(fresh, seed=seed, namespace="m6-fresh-evaluation-v1")
    promotion = sorted(fresh_order[:n_eval])
    holdout = sorted(fresh_order[n_eval : 2 * n_eval])
    used = set(promotion) | set(holdout)
    remaining = [index for index in eligible if index not in used]
    development_order = _ranked(remaining, seed=seed, namespace="m6-development-v1")
    mini_dev_count = int(split["mini_dev_tasks"])
    formal_dev_count = int(split["formal_dev_tasks"])
    mini_dev = sorted(development_order[:mini_dev_count])
    formal_dev = sorted(development_order[mini_dev_count : mini_dev_count + formal_dev_count])
    train = sorted(set(remaining) - set(mini_dev) - set(formal_dev))
    _require(len(train) >= int(split["minimum_remaining_train_tasks"]), "M6 remaining train roster is too small")
    mini_train = sorted(
        _ranked(train, seed=seed, namespace="m6-mini-train-v1")[: int(split["mini_train_tasks"])]
    )

    roles = {
        "train": train,
        "mini_train": mini_train,
        "mini_dev": mini_dev,
        "formal_dev": formal_dev,
        "promotion": promotion,
        "holdout": holdout,
    }
    disjoint_roles = ("train", "mini_dev", "formal_dev", "promotion", "holdout")
    for left_index, left in enumerate(disjoint_roles):
        for right in disjoint_roles[left_index + 1 :]:
            _require(not set(roles[left]) & set(roles[right]), f"M6 role overlap: {left}/{right}")
    _require(set(mini_train) <= set(train), "M6 mini-train is not a train subset")
    normalized = {
        role: {normalized_instruction(str(goals[index]["instruction"])) for index in indices}
        for role, indices in roles.items()
        if role != "mini_train"
    }
    for left_index, left in enumerate(disjoint_roles):
        for right in disjoint_roles[left_index + 1 :]:
            _require(not normalized[left] & normalized[right], f"M6 normalized instruction overlap: {left}/{right}")
    _require(not (set(promotion) | set(holdout)) & exposed, "M6 exposed goal entered fresh evaluation")

    output = {
        "schema_version": SPLIT_SCHEMA,
        "study_id": STUDY_ID,
        "protocol_sha256": protocol_sha256,
        "goals_canonical_sha256": sha256_json(list(goals)),
        "selection_seed": seed,
        "n_eval": n_eval,
        "power_report_content_sha256": power_report_content_sha256,
        "source_eligible_train_count": len(eligible),
        "exposure_registry_content_sha256": registry["content_sha256"],
        "roles": {
            role: {
                "goal_indices": indices,
                "task_ids": [task_id_for_goal_index(index) for index in indices],
                "count": len(indices),
                "goal_index_sha256": _compact_indices_sha256(indices),
            }
            for role, indices in roles.items()
        },
        "checks": {
            "disjoint_primary_roles": True,
            "mini_train_subset_of_train": True,
            "normalized_instruction_overlap_count": 0,
            "exposed_goal_count_in_promotion": 0,
            "exposed_goal_count_in_holdout": 0,
        },
    }
    output["content_sha256"] = sha256_json(output)
    return output


def validate_split_lock(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == SPLIT_SCHEMA, "M6 split schema drift")
    _require(value.get("study_id") == STUDY_ID, "M6 split study drift")
    _require(value.get("n_eval") in (1000, 1500, 2000), "M6 split N_eval drift")
    power_hash = value.get("power_report_content_sha256")
    _require(
        power_hash is None or SHA256_RE.fullmatch(str(power_hash)) is not None,
        "M6 split power-report SHA drift",
    )
    _require(SHA256_RE.fullmatch(str(value.get("protocol_sha256"))) is not None, "M6 split protocol SHA drift")
    _require(SHA256_RE.fullmatch(str(value.get("goals_canonical_sha256"))) is not None, "M6 split goals SHA drift")
    _require(
        SHA256_RE.fullmatch(str(value.get("exposure_registry_content_sha256"))) is not None,
        "M6 split exposure SHA drift",
    )
    roles = value.get("roles")
    _require(isinstance(roles, Mapping), "M6 split roles are missing")
    expected_roles = {"train", "mini_train", "mini_dev", "formal_dev", "promotion", "holdout"}
    _require(set(roles) == expected_roles, "M6 split role set drift")
    rosters: dict[str, list[int]] = {}
    for role in expected_roles:
        item = roles[role]
        _require(isinstance(item, Mapping), f"M6 split role malformed: {role}")
        indices = item.get("goal_indices")
        _require(isinstance(indices, list) and indices == sorted(set(indices)), f"M6 split roster drift: {role}")
        _require(item.get("count") == len(indices), f"M6 split count drift: {role}")
        _require(item.get("goal_index_sha256") == _compact_indices_sha256(indices), f"M6 split hash drift: {role}")
        _require(item.get("task_ids") == [task_id_for_goal_index(index) for index in indices], f"M6 split task id drift: {role}")
        rosters[role] = indices
    _require(len(rosters["mini_train"]) == 256 and set(rosters["mini_train"]) <= set(rosters["train"]), "M6 mini-train role drift")
    _require(len(rosters["mini_dev"]) == 200, "M6 mini-dev role drift")
    _require(len(rosters["formal_dev"]) == 500, "M6 formal-dev role drift")
    _require(len(rosters["promotion"]) == len(rosters["holdout"]) == value["n_eval"], "M6 evaluation roster drift")
    primary = ("train", "mini_dev", "formal_dev", "promotion", "holdout")
    for left_index, left in enumerate(primary):
        for right in primary[left_index + 1 :]:
            _require(not set(rosters[left]) & set(rosters[right]), f"M6 role overlap: {left}/{right}")
    _require(value.get("checks") == {
        "disjoint_primary_roles": True,
        "mini_train_subset_of_train": True,
        "normalized_instruction_overlap_count": 0,
        "exposed_goal_count_in_promotion": 0,
        "exposed_goal_count_in_holdout": 0,
    }, "M6 split checks drift")
    expected = dict(value)
    observed_hash = expected.pop("content_sha256", None)
    _require(observed_hash == sha256_json(expected), "M6 split self-hash drift")
    return value


def _metric(metrics: Mapping[str, Any], field: str) -> float:
    value = _finite_number(metrics.get(field), field)
    _require(0.0 <= value <= 1.0, f"{field} must be a fraction")
    return value


def build_mini_gate_report(
    *,
    raw: Mapping[str, Any],
    sft: Mapping[str, Any],
    rl: Mapping[str, Any],
    bootstrap_positive_fraction_sft_minus_raw: float,
    bootstrap_positive_fraction_rl_minus_sft: float,
    corpus_audit: Mapping[str, Any],
    rl_audit: Mapping[str, Any],
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(protocol or load_protocol()["payload"])
    validate_protocol(contract)
    mini = contract["mini"]
    raw_success = _metric(raw, "strict_success_rate")
    sft_success = _metric(sft, "strict_success_rate")
    rl_success = _metric(rl, "strict_success_rate")
    sft_delta_pp = (sft_success - raw_success) * 100.0
    rl_delta_pp = (rl_success - sft_success) * 100.0
    checks = {
        "sft_minus_raw_minimum": sft_delta_pp >= float(mini["minimum_sft_minus_raw_pp"]),
        "rl_minus_sft_minimum": rl_delta_pp >= float(mini["minimum_rl_minus_sft_pp"]),
        "rl_above_raw": rl_success > raw_success,
        "sft_bootstrap_direction": _finite_number(bootstrap_positive_fraction_sft_minus_raw, "SFT bootstrap fraction") >= float(mini["minimum_bootstrap_positive_fraction"]),
        "rl_bootstrap_direction": _finite_number(bootstrap_positive_fraction_rl_minus_sft, "RL bootstrap fraction") >= float(mini["minimum_bootstrap_positive_fraction"]),
        "sft_search_exhaustion_guardrail": (
            _metric(sft, "search_exhaustion_rate") - _metric(raw, "search_exhaustion_rate")
        ) * 100.0 <= float(mini["maximum_sft_search_exhaustion_increase_pp"]),
        "rl_partial_match_guardrail": (
            _metric(rl, "partial_match_purchase_rate") - _metric(sft, "partial_match_purchase_rate")
        ) * 100.0 <= float(mini["maximum_rl_partial_match_increase_pp"]),
        "sft_schema_or_action_error_guardrail": max(
            (_metric(sft, "schema_error_rate") - _metric(raw, "schema_error_rate")) * 100.0,
            (_metric(sft, "action_error_rate") - _metric(raw, "action_error_rate")) * 100.0,
        ) <= float(mini["maximum_stage_schema_or_action_error_increase_pp"]),
        "rl_schema_or_action_error_guardrail": max(
            (_metric(rl, "schema_error_rate") - _metric(sft, "schema_error_rate")) * 100.0,
            (_metric(rl, "action_error_rate") - _metric(sft, "action_error_rate")) * 100.0,
        ) <= float(mini["maximum_stage_schema_or_action_error_increase_pp"]),
        "corpus_audit_passed": corpus_audit.get("passed") is True,
        "rl_audit_passed": rl_audit.get("passed") is True,
        "development_only": all(item.get("development_only") is True for item in (sft, rl)),
    }
    report = {
        "schema_version": MINI_GATE_SCHEMA,
        "study_id": STUDY_ID,
        "formal_training": False,
        "development_only": True,
        "passed": all(checks.values()),
        "decision": "READY_FOR_FULL_SFT_APPROVAL" if all(checks.values()) else "STOP_AND_BURN_MINI_DEV",
        "metrics": {"raw": dict(raw), "sft": dict(sft), "rl": dict(rl)},
        "deltas_pp": {"sft_minus_raw": sft_delta_pp, "rl_minus_sft": rl_delta_pp},
        "bootstrap_positive_fraction": {
            "sft_minus_raw": float(bootstrap_positive_fraction_sft_minus_raw),
            "rl_minus_sft": float(bootstrap_positive_fraction_rl_minus_sft),
        },
        "checks": checks,
        "corpus_audit_content_sha256": corpus_audit.get("content_sha256"),
        "rl_audit_content_sha256": rl_audit.get("content_sha256"),
        "failed_dev_roster_policy": mini["failed_dev_roster_policy"],
    }
    report["content_sha256"] = sha256_json(report)
    return report


def validate_mini_gate_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == MINI_GATE_SCHEMA, "M6 mini gate schema drift")
    _require(value.get("formal_training") is False and value.get("development_only") is True, "M6 mini scope drift")
    checks = value.get("checks")
    _require(isinstance(checks, Mapping) and checks, "M6 mini checks are missing")
    _require(value.get("passed") is all(bool(item) for item in checks.values()), "M6 mini gate decision drift")
    expected_decision = "READY_FOR_FULL_SFT_APPROVAL" if value["passed"] else "STOP_AND_BURN_MINI_DEV"
    _require(value.get("decision") == expected_decision, "M6 mini decision label drift")
    expected = dict(value)
    observed_hash = expected.pop("content_sha256", None)
    _require(observed_hash == sha256_json(expected), "M6 mini report self-hash drift")
    return value
