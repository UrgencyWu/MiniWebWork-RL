"""Closed-loop M6-mini metrics, bootstrap direction gate and RL audit."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .long_horizon_rl.contracts import sha256_json
from .m6_posttraining_protocol import (
    build_mini_gate_report,
    load_protocol,
    validate_protocol,
)
from .webshop_rl.verifier_td import validate_credit_assignment

EVAL_SCHEMA = "m6_mini_closed_loop_identity_v1"
RL_AUDIT_SCHEMA = "m6_mini_rl_audit_v1"
CHAIN_SCHEMA = "m6_mini_chain_report_v1"
SFT_GATE_SCHEMA = "m6_mini_sft_gate_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{field} must be finite",
    )
    return float(value)


def _trajectory_key(trajectory: Mapping[str, Any]) -> tuple[Any, ...]:
    rollout_seed = trajectory.get("rollout_seed")
    if rollout_seed is not None:
        _require(isinstance(rollout_seed, int) and not isinstance(rollout_seed, bool), "M6 rollout seed is invalid")
        return ("seed", rollout_seed)
    rollout_index = trajectory.get("rollout_index")
    _require(isinstance(rollout_index, int) and not isinstance(rollout_index, bool), "M6 rollout identity is missing")
    return ("index", rollout_index)


def summarize_closed_loop_identity(
    *,
    identity: str,
    groups: Sequence[Mapping[str, Any]],
    development_only: bool,
    evaluation_contract_sha256: str | None = None,
    source_bindings: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(protocol or load_protocol()["payload"])
    validate_protocol(contract)
    _require(isinstance(identity, str) and identity in {"raw", "mini_sft", "mini_rl"}, "M6 mini identity drift")
    _require(len(groups) == int(contract["split"]["mini_dev_tasks"]), "M6 mini-dev task count drift")
    task_rows: dict[str, dict[str, Any]] = {}
    schema_errors = 0
    action_errors = 0
    search_exhaustions = 0
    partial_match_purchases = 0
    trajectory_count = 0
    total_tokens = 0
    total_steps = 0
    for group in groups:
        _require(isinstance(group, Mapping), "M6 evaluation group is malformed")
        task_id = group.get("task_id")
        _require(isinstance(task_id, str) and task_id and task_id not in task_rows, "M6 evaluation task identity drift")
        trajectories = group.get("trajectories")
        _require(
            isinstance(trajectories, list)
            and len(trajectories) == int(contract["mini"]["evaluation_K"]),
            "M6 mini evaluation must be K=4",
        )
        keys = [_trajectory_key(trajectory) for trajectory in trajectories]
        _require(len(set(keys)) == len(keys), "M6 evaluation rollout identity duplicate")
        successes: list[float] = []
        dense_scores: list[float] = []
        for trajectory in trajectories:
            success = trajectory.get("success")
            _require(isinstance(success, bool), "M6 evaluation success is invalid")
            dense = _finite(trajectory.get("task_score", trajectory.get("reward")), "M6 task score")
            _require(0.0 <= dense <= 1.0 and success == (dense >= 0.999), "M6 strict success/task score disagreement")
            successes.append(float(success))
            dense_scores.append(dense)
            turns = trajectory.get("turns")
            _require(isinstance(turns, list) and turns, "M6 evaluation trajectory has no turns")
            has_schema_error = any(turn.get("schema_valid") is not True for turn in turns)
            has_action_error = any(
                isinstance(turn.get("action_result"), Mapping)
                and bool(turn["action_result"].get("error_code"))
                for turn in turns
            )
            last_observation = turns[-1].get("post_action_observation", turns[-1].get("observation"))
            last_page = str(last_observation.get("page_type", "")) if isinstance(last_observation, Mapping) else ""
            termination = str(trajectory.get("termination_reason", ""))
            is_search_exhaustion = termination in {"max_environment_steps", "max_model_turns"} and last_page in {
                "home",
                "search_results",
            }
            is_partial_purchase = termination == "purchase" and not success and dense > 0.0
            schema_errors += int(has_schema_error)
            action_errors += int(has_action_error)
            search_exhaustions += int(is_search_exhaustion)
            partial_match_purchases += int(is_partial_purchase)
            trajectory_count += 1
            total_tokens += int(trajectory.get("generated_action_tokens", 0))
            total_steps += int(trajectory.get("environment_steps", 0))
        task_rows[task_id] = {
            "task_id": task_id,
            "strict_success_rate": statistics.fmean(successes),
            "dense_score": statistics.fmean(dense_scores),
            "rollout_keys": [list(item) for item in sorted(keys)],
        }
    report = {
        "schema_version": EVAL_SCHEMA,
        "study_id": contract["study_id"],
        "identity": identity,
        "development_only": development_only,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "source_bindings": dict(source_bindings or {}),
        "task_count": len(task_rows),
        "trajectory_count": trajectory_count,
        "K": int(contract["mini"]["evaluation_K"]),
        "strict_success_rate": statistics.fmean(row["strict_success_rate"] for row in task_rows.values()),
        "dense_score": statistics.fmean(row["dense_score"] for row in task_rows.values()),
        "search_exhaustion_rate": search_exhaustions / trajectory_count,
        "partial_match_purchase_rate": partial_match_purchases / trajectory_count,
        "schema_error_rate": schema_errors / trajectory_count,
        "action_error_rate": action_errors / trajectory_count,
        "mean_generated_action_tokens": total_tokens / trajectory_count,
        "mean_environment_steps": total_steps / trajectory_count,
        "task_rows": task_rows,
        "task_roster_sha256": sha256_json(list(task_rows)),
        "rollout_key_roster_sha256": sha256_json(
            {task_id: row["rollout_keys"] for task_id, row in task_rows.items()}
        ),
    }
    report["content_sha256"] = sha256_json(report)
    return report


def validate_closed_loop_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == EVAL_SCHEMA, "M6 mini evaluation schema drift")
    _require(value.get("identity") in {"raw", "mini_sft", "mini_rl"}, "M6 mini evaluation identity drift")
    _require(value.get("task_count") == 200 and value.get("trajectory_count") == 800, "M6 mini evaluation size drift")
    _require(value.get("K") == 4, "M6 mini evaluation K drift")
    evaluation_contract = value.get("evaluation_contract_sha256")
    _require(
        isinstance(evaluation_contract, str)
        and len(evaluation_contract) == 64,
        "M6 mini evaluation contract hash drift",
    )
    rows = value.get("task_rows")
    _require(isinstance(rows, Mapping) and len(rows) == 200, "M6 mini task rows drift")
    _require(value.get("task_roster_sha256") == sha256_json(list(rows)), "M6 mini task roster hash drift")
    _require(
        value.get("rollout_key_roster_sha256")
        == sha256_json({task_id: row["rollout_keys"] for task_id, row in rows.items()}),
        "M6 mini rollout-key roster hash drift",
    )
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 mini evaluation self-hash drift")
    return value


def _paired_differences(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> list[float]:
    _require(first.get("schema_version") == second.get("schema_version") == EVAL_SCHEMA, "M6 paired eval schema drift")
    _require(
        first.get("evaluation_contract_sha256") == second.get("evaluation_contract_sha256"),
        "M6 paired evaluation contract drift",
    )
    first_rows = first.get("task_rows")
    second_rows = second.get("task_rows")
    _require(isinstance(first_rows, Mapping) and isinstance(second_rows, Mapping), "M6 paired task rows are missing")
    _require(tuple(first_rows) == tuple(second_rows), "M6 paired task roster drift")
    differences: list[float] = []
    for task_id in first_rows:
        _require(first_rows[task_id]["rollout_keys"] == second_rows[task_id]["rollout_keys"], "M6 paired rollout seeds drift")
        differences.append(
            float(second_rows[task_id]["strict_success_rate"])
            - float(first_rows[task_id]["strict_success_rate"])
        )
    return differences


def bootstrap_positive_fraction(
    differences: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 20260812,
) -> float:
    _require(bool(differences) and samples >= 10_000, "M6 mini bootstrap requires tasks and >=10,000 draws")
    values = [_finite(value, "M6 paired task difference") for value in differences]
    rng = random.Random(seed)
    positive = 0
    for _ in range(samples):
        mean = sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        positive += int(mean > 0.0)
    return positive / samples


def build_mini_sft_gate(
    *,
    raw: Mapping[str, Any],
    sft: Mapping[str, Any],
    corpus_audit: Mapping[str, Any],
    bootstrap_samples: int = 10_000,
    seed: int = 20260812,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stop before online RL unless mini-SFT improves the paired Raw policy."""

    contract = dict(protocol or load_protocol()["payload"])
    validate_protocol(contract)
    raw = validate_closed_loop_identity(raw)
    sft = validate_closed_loop_identity(sft)
    from .webshop_rl.m6_corpus import validate_conditional_learnability_audit

    corpus = validate_conditional_learnability_audit(corpus_audit)
    differences = _paired_differences(raw, sft)
    positive = bootstrap_positive_fraction(differences, samples=bootstrap_samples, seed=seed)
    delta_pp = (float(sft["strict_success_rate"]) - float(raw["strict_success_rate"])) * 100.0
    mini = contract["mini"]
    checks = {
        "development_only": sft.get("development_only") is True,
        "corpus_audit_passed": corpus.get("passed") is True,
        "sft_minus_raw_minimum": delta_pp >= float(mini["minimum_sft_minus_raw_pp"]),
        "bootstrap_direction": positive >= float(mini["minimum_bootstrap_positive_fraction"]),
        "search_exhaustion_guardrail": (
            float(sft["search_exhaustion_rate"]) - float(raw["search_exhaustion_rate"])
        ) * 100.0 <= float(mini["maximum_sft_search_exhaustion_increase_pp"]),
        "schema_or_action_error_guardrail": max(
            (float(sft["schema_error_rate"]) - float(raw["schema_error_rate"])) * 100.0,
            (float(sft["action_error_rate"]) - float(raw["action_error_rate"])) * 100.0,
        ) <= float(mini["maximum_stage_schema_or_action_error_increase_pp"]),
    }
    report = {
        "schema_version": SFT_GATE_SCHEMA,
        "study_id": contract["study_id"],
        "development_only": True,
        "formal_training": False,
        "passed": all(checks.values()),
        "decision": "ALLOW_MINI_RL" if all(checks.values()) else "STOP_AND_BURN_MINI_DEV",
        "sft_minus_raw_pp": delta_pp,
        "bootstrap_positive_fraction": positive,
        "bootstrap_samples": bootstrap_samples,
        "checks": checks,
        "raw_content_sha256": raw["content_sha256"],
        "sft_content_sha256": sft["content_sha256"],
        "corpus_audit_content_sha256": corpus["content_sha256"],
        "paired_task_difference_sha256": sha256_json(differences),
    }
    report["content_sha256"] = sha256_json(report)
    return report


def validate_mini_sft_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == SFT_GATE_SCHEMA, "M6 mini SFT gate schema drift")
    checks = value.get("checks")
    _require(isinstance(checks, Mapping) and checks, "M6 mini SFT gate checks are missing")
    _require(value.get("passed") is all(bool(item) for item in checks.values()), "M6 mini SFT gate decision drift")
    expected_decision = "ALLOW_MINI_RL" if value["passed"] else "STOP_AND_BURN_MINI_DEV"
    _require(value.get("decision") == expected_decision, "M6 mini SFT gate label drift")
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 mini SFT gate self-hash drift")
    return value


def audit_mini_rl(
    *,
    learner_report: Mapping[str, Any],
    credit_assignments: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(protocol or load_protocol()["payload"])
    validate_protocol(contract)
    mini = contract["mini"]
    rl = contract["rl"]
    _require(bool(credit_assignments), "M6 mini RL credit evidence is empty")
    validated_credit = [validate_credit_assignment(item) for item in credit_assignments]
    expected_learner = dict(learner_report)
    observed_learner_hash = expected_learner.pop("content_sha256", None)
    _require(observed_learner_hash == sha256_json(expected_learner), "M6 mini learner self-hash drift")
    iterations = learner_report.get("iterations")
    _require(isinstance(iterations, list), "M6 mini learner iterations are missing")
    finite_iterations = bool(iterations) and all(isinstance(item, Mapping) for item in iterations) and all(
        math.isfinite(float(item.get("loss")))
        and math.isfinite(float(item.get("gradient_norm")))
        and math.isfinite(float(item.get("observed_kl")))
        for item in iterations
        if isinstance(item, Mapping)
    )
    mixed_iteration_count = sum(
        bool(item.get("mixed_strict_reward_signal")) for item in iterations if isinstance(item, Mapping)
    )
    turn_count = sum(int(item["metrics"]["turn_count"]) for item in validated_credit)
    nonzero_turns = sum(int(item["metrics"]["nonzero_td_turn_count"]) for item in validated_credit)
    mixed_groups = sum(bool(item["metrics"]["mixed_strict_reward_signal"]) for item in validated_credit)
    checks = {
        "development_only": learner_report.get("development_only") is True,
        "method": learner_report.get("method") == rl["method"],
        "group_size": learner_report.get("group_size") == int(rl["group_size"]),
        "generated_action_token_cap": int(learner_report.get("generated_action_tokens", -1)) <= int(mini["maximum_rl_generated_action_tokens"]),
        "minimum_mixed_iterations": mixed_iteration_count >= int(mini["minimum_mixed_rl_iterations"]),
        "minimum_optimizer_updates": int(learner_report.get("optimizer_updates", -1)) >= int(mini["minimum_optimizer_updates"]),
        "parameters_changed": learner_report.get("parameters_changed") is True,
        "lineage_chain_complete": learner_report.get("lineage_chain_complete") is True,
        "parameter_hash_changed": (
            isinstance(learner_report.get("parameter_sha256_before"), str)
            and isinstance(learner_report.get("parameter_sha256_after"), str)
            and learner_report.get("parameter_sha256_before") != learner_report.get("parameter_sha256_after")
        ),
        "finite_loss_gradient_kl": bool(iterations) and finite_iterations,
        # Iteration zero starts from the frozen SFT reference, so its pre-step
        # KL is mathematically zero.  The safety gate is the upper bound plus a
        # correctly directed adaptive-controller response; requiring every
        # iteration to exceed the lower target would make a valid first update
        # impossible by construction.
        "observed_kl_upper_bound": bool(iterations) and all(
            0.0 <= float(item["observed_kl"]) <= float(rl["adaptive_kl_target_maximum"])
            for item in iterations
        ),
        "adaptive_kl_controller_response": bool(iterations) and all(
            (
                float(item["adaptive_kl_coefficient_after"])
                <= float(item["adaptive_kl_coefficient_before"])
                if float(item["observed_kl"]) < float(rl["adaptive_kl_target_minimum"])
                else float(item["adaptive_kl_coefficient_after"])
                >= float(item["adaptive_kl_coefficient_before"])
                if float(item["observed_kl"]) > float(rl["adaptive_kl_target_maximum"])
                else float(item["adaptive_kl_coefficient_after"])
                == float(item["adaptive_kl_coefficient_before"])
            )
            for item in iterations
        ),
        "mixed_group_fraction": mixed_groups / len(validated_credit) >= float(rl["minimum_mixed_group_fraction"]),
        "nonzero_credit_turn_fraction": nonzero_turns / turn_count >= float(rl["minimum_nonzero_credit_turn_fraction"]),
        "telescoping": all(
            float(item["metrics"]["maximum_absolute_telescoping_error"])
            <= float(rl["telescoping_absolute_tolerance"])
            for item in validated_credit
        ),
        "credit_conservation": all(
            float(item["metrics"]["maximum_absolute_credit_conservation_error"])
            <= float(rl["telescoping_absolute_tolerance"])
            for item in validated_credit
        ),
        "partial_match_negative_terminal_credit": all(
            item["metrics"]["positive_progress_failure_terminal_cancellation_complete"] is True
            for item in validated_credit
        ),
    }
    report = {
        "schema_version": RL_AUDIT_SCHEMA,
        "study_id": contract["study_id"],
        "development_only": True,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "iteration_count": len(iterations),
            "mixed_iteration_count": mixed_iteration_count,
            "optimizer_updates": int(learner_report.get("optimizer_updates", 0)),
            "generated_action_tokens": int(learner_report.get("generated_action_tokens", 0)),
            "credit_group_count": len(validated_credit),
            "mixed_group_fraction": mixed_groups / len(validated_credit),
            "turn_count": turn_count,
            "nonzero_credit_turn_fraction": nonzero_turns / turn_count,
        },
        "learner_report_content_sha256": learner_report.get("content_sha256"),
        "credit_assignment_content_sha256": [item["content_sha256"] for item in validated_credit],
    }
    report["content_sha256"] = sha256_json(report)
    return report


def validate_mini_rl_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == RL_AUDIT_SCHEMA, "M6 mini RL audit schema drift")
    checks = value.get("checks")
    _require(isinstance(checks, Mapping) and checks, "M6 mini RL audit checks are missing")
    _require(value.get("passed") is all(bool(item) for item in checks.values()), "M6 mini RL audit decision drift")
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 mini RL audit self-hash drift")
    return value


def build_mini_chain_report(
    *,
    raw: Mapping[str, Any],
    sft: Mapping[str, Any],
    rl: Mapping[str, Any],
    corpus_audit: Mapping[str, Any],
    rl_audit: Mapping[str, Any],
    bootstrap_samples: int = 10_000,
    seed: int = 20260812,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(protocol or load_protocol()["payload"])
    raw = validate_closed_loop_identity(raw)
    sft = validate_closed_loop_identity(sft)
    rl = validate_closed_loop_identity(rl)
    from .webshop_rl.m6_corpus import validate_conditional_learnability_audit

    corpus_audit = validate_conditional_learnability_audit(corpus_audit)
    rl_audit = validate_mini_rl_audit(rl_audit)
    sft_differences = _paired_differences(raw, sft)
    rl_differences = _paired_differences(sft, rl)
    sft_positive = bootstrap_positive_fraction(sft_differences, samples=bootstrap_samples, seed=seed)
    rl_positive = bootstrap_positive_fraction(rl_differences, samples=bootstrap_samples, seed=seed + 1)
    gate = build_mini_gate_report(
        raw=raw,
        sft=sft,
        rl=rl,
        bootstrap_positive_fraction_sft_minus_raw=sft_positive,
        bootstrap_positive_fraction_rl_minus_sft=rl_positive,
        corpus_audit=corpus_audit,
        rl_audit=rl_audit,
        protocol=contract,
    )
    report = {
        "schema_version": CHAIN_SCHEMA,
        "study_id": contract["study_id"],
        "formal_training": False,
        "development_only": True,
        "passed": gate["passed"],
        "decision": gate["decision"],
        "identity_content_sha256": {
            "raw": raw.get("content_sha256"),
            "mini_sft": sft.get("content_sha256"),
            "mini_rl": rl.get("content_sha256"),
        },
        "paired_task_difference_sha256": {
            "sft_minus_raw": sha256_json(sft_differences),
            "rl_minus_sft": sha256_json(rl_differences),
        },
        "bootstrap_samples": bootstrap_samples,
        "mini_gate": gate,
        "mini_checkpoints_reusable_for_formal_training": False,
        "burn_mini_dev_on_failure": not gate["passed"],
    }
    report["content_sha256"] = sha256_json(report)
    return report
