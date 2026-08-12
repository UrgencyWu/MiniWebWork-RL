"""Strict K=8 GRPO credit with detached verifier-TD shaping for M6.

The verifier potential is evidence produced from the public WebShop state.  It
is never shown to the policy and is treated as a detached scalar by the
learner.  The last transition is always forced to the strict official terminal
reward, so dense rewards telescope exactly to the binary task reward.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

from ..long_horizon_rl.contracts import sha256_json
from .credit import (
    canonical_credit_state,
    public_state_anchor_signature,
    standardized_advantages,
)

BASELINE_METHOD = "multi_turn_grpo"
ANCHOR_METHOD = "anchor_gigpo"
METHODS = (BASELINE_METHOD, ANCHOR_METHOD)
# Backward-compatible default for callers that only validate rollout evidence.
METHOD = ANCHOR_METHOD
FORMULA_VERSION_BY_METHOD = {
    BASELINE_METHOD: "m6_strict_grpo_uniform_turn_credit_v1",
    ANCHOR_METHOD: "m6_strict_grpo_verifier_td_v1",
}
FORMULA_VERSION = FORMULA_VERSION_BY_METHOD[ANCHOR_METHOD]
GROUP_SIZE = 8
VERIFIER_TD_LAMBDA = 0.5
TELESCOPING_TOLERANCE = 1e-8
MAXIMUM_NONTERMINAL_POTENTIAL = 0.9
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
RESULT_LINE_RE = re.compile(r"^\s*\[\d+\]\s+B[0-9A-Z]{9}\s*\|", re.IGNORECASE)
PRICE_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
SELECTED_OPTION_RE = re.compile(
    r"^\s*-\s*[^\n:]+\(selected:\s*([^\)]+)\)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
PUBLIC_PROGRESS_SCHEMA = "m6_public_verifier_progress_v1"


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


def strict_terminal_reward(task_score: Any) -> float:
    """Return the only M6 terminal reward: official score >= 0.999."""

    score = _finite(task_score, "official task score")
    _require(0.0 <= score <= 1.0, "official task score must be in [0, 1]")
    return float(score >= 0.999)


def public_stage_potential(
    *,
    page_type: str,
    candidate_attribute_match_fraction: float = 0.0,
    item_attribute_match_fraction: float = 0.0,
    selected_option_fraction: float = 0.0,
) -> float:
    """Map verifier matches from a public state to the frozen M6 potential.

    The caller owns extraction of public attribute matches.  This function is
    deliberately small and fail-closed: it accepts only normalized fractions
    and known public page stages.  Terminal pages are not accepted because the
    terminal boundary is set from the official task score instead.
    """

    candidate_match = _finite(
        candidate_attribute_match_fraction,
        "candidate attribute match fraction",
    )
    item_match = _finite(item_attribute_match_fraction, "item attribute match fraction")
    options = _finite(selected_option_fraction, "selected option fraction")
    _require(0.0 <= candidate_match <= 1.0, "candidate match fraction must be in [0, 1]")
    _require(0.0 <= item_match <= 1.0, "item match fraction must be in [0, 1]")
    _require(0.0 <= options <= 1.0, "selected option fraction must be in [0, 1]")
    normalized_page = str(page_type).strip().casefold()
    if normalized_page in {"home", "search"}:
        value = 0.0
    elif normalized_page in {"search_results", "results"}:
        value = 0.15 * candidate_match
    elif normalized_page in {"item", "product", "subpage", "item_subpage"}:
        value = 0.15 * candidate_match + 0.55 * item_match
    elif normalized_page in {"item_options", "options", "product_options", "item_with_options"}:
        value = 0.15 * candidate_match + 0.55 * item_match + 0.20 * options
    else:
        raise ValueError(f"unsupported public page type for verifier potential: {page_type}")
    return min(MAXIMUM_NONTERMINAL_POTENTIAL, max(0.0, value))


def _normalize_text(value: Any) -> str:
    return " ".join(match.group(0).casefold() for match in TEXT_TOKEN_RE.finditer(str(value or "")))


def _flatten_goal_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        values = [item for nested in value.values() for item in _flatten_goal_values(nested)]
    elif isinstance(value, (list, tuple)):
        values = [item for nested in value for item in _flatten_goal_values(nested)]
    elif isinstance(value, bool) or value is None:
        values = []
    else:
        normalized = _normalize_text(value)
        values = [normalized] if normalized else []
    return values


def _unique(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        normalized = _normalize_text(value)
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _goal_product_constraints(goal: Mapping[str, Any]) -> list[str]:
    """Return verifier-only product constraints, deliberately excluding name/ASIN."""

    attributes = goal.get("attributes") or goal.get("instruction_attributes") or []
    type_values = [
        str(goal.get("query") or ""),
        str(goal.get("product_category") or "").split("›")[-1],
        str(goal.get("category") or ""),
    ]
    return _unique([*_flatten_goal_values(attributes), *type_values])


def _goal_options(goal: Mapping[str, Any]) -> list[str]:
    return _unique(_flatten_goal_values(goal.get("goal_options") or []))


def _matches(public_text: str, desired: str) -> bool:
    public = _normalize_text(public_text)
    target = _normalize_text(desired)
    if not public or not target:
        return False
    if target in public:
        return True
    public_tokens = set(public.split())
    target_tokens = set(target.split())
    return bool(target_tokens) and len(public_tokens & target_tokens) / len(target_tokens) > 0.85


def _price_match(public_text: str, goal: Mapping[str, Any]) -> tuple[int, int]:
    upper = goal.get("price_upper")
    if not isinstance(upper, (int, float)) or isinstance(upper, bool) or not math.isfinite(float(upper)):
        return (0, 0)
    prices = [float(match.group(1)) for match in PRICE_RE.finditer(public_text)]
    return (int(bool(prices) and min(prices) <= float(upper)), 1)


def _match_fraction(public_text: str, goal: Mapping[str, Any]) -> tuple[float, int, int]:
    constraints = _goal_product_constraints(goal)
    matched = sum(_matches(public_text, value) for value in constraints)
    price_matched, price_total = _price_match(public_text, goal)
    numerator = matched + price_matched
    denominator = len(constraints) + price_total
    return (numerator / denominator if denominator else 0.0, numerator, denominator)


def _product_text(visible_text: str) -> str:
    """Remove repeated task/control lines before matching product evidence."""

    retained = []
    for line in str(visible_text or "").splitlines():
        normalized = line.strip().casefold()
        if normalized.startswith("instruction:") or normalized.startswith("clickable controls:"):
            continue
        retained.append(line)
    return "\n".join(retained)


def _option_fraction(visible_text: str, goal: Mapping[str, Any]) -> tuple[float, int, int]:
    desired = _goal_options(goal)
    if not desired:
        # No option constraint means there is no option-progress event to
        # reward.  Price/type/attribute progress is already represented by
        # the product match components.
        return (0.0, 0, 0)
    selected = [
        _normalize_text(match.group(1))
        for match in SELECTED_OPTION_RE.finditer(str(visible_text or ""))
        if _normalize_text(match.group(1)) not in {"", "not selected"}
    ]
    matched = sum(any(_matches(value, target) for value in selected) for target in desired)
    return (matched / len(desired), matched, len(desired))


def score_public_observation(
    *,
    goal: Mapping[str, Any],
    observation: Mapping[str, Any],
    terminal_boundary_reward: float | None = None,
) -> dict[str, Any]:
    """Score only content present in one post-action public observation.

    Goal metadata is available only to this detached verifier.  The evidence
    object stores counts and hashes, not goal strings, and is never passed to
    prompt construction.
    """

    state = canonical_credit_state(observation)
    instruction = _normalize_text(goal.get("instruction"))
    _require(instruction and _normalize_text(state["instruction"]) == instruction, "M6 verifier goal/observation mismatch")
    visible_text = str(state["visible_text"])
    page_type = str(state["page_type"]).casefold()
    candidate_fraction = item_fraction = selected_fraction = 0.0
    candidate_matches = candidate_total = item_matches = item_total = 0
    selected_matches = selected_total = 0

    if terminal_boundary_reward is not None:
        terminal = _finite(terminal_boundary_reward, "M6 terminal boundary reward")
        _require(terminal in (0.0, 1.0), "M6 terminal boundary reward must be strict binary")
        potential = terminal
        boundary = "strict_episode_boundary"
    elif page_type in {"home", "search"}:
        potential = 0.0
        boundary = "nonterminal_public_state"
    elif page_type in {"search_results", "results"}:
        candidates = [line for line in visible_text.splitlines() if RESULT_LINE_RE.match(line)]
        candidate_scores = [_match_fraction(line, goal) for line in candidates]
        if candidate_scores:
            candidate_fraction, candidate_matches, candidate_total = max(
                candidate_scores,
                key=lambda item: (item[0], item[1], -item[2]),
            )
        potential = public_stage_potential(
            page_type="search_results",
            candidate_attribute_match_fraction=candidate_fraction,
        )
        boundary = "nonterminal_public_state"
    elif page_type in {"item", "product", "subpage", "item_subpage"}:
        product_text = _product_text(visible_text)
        item_fraction, item_matches, item_total = _match_fraction(product_text, goal)
        candidate_fraction = item_fraction
        candidate_matches, candidate_total = item_matches, item_total
        selected_fraction, selected_matches, selected_total = _option_fraction(product_text, goal)
        potential = public_stage_potential(
            page_type="item_with_options" if page_type in {"item", "product"} else "subpage",
            candidate_attribute_match_fraction=candidate_fraction,
            item_attribute_match_fraction=item_fraction,
            # Options contribute only when their selected values are actually
            # visible on the item page, never from hidden environment state.
            selected_option_fraction=selected_fraction if page_type in {"item", "product"} else 0.0,
        )
        boundary = "nonterminal_public_state"
    elif page_type == "done":
        raise ValueError("M6 done observation requires strict terminal boundary reward")
    else:
        raise ValueError(f"unsupported public page type for M6 verifier: {page_type}")

    evidence = {
        "schema_version": PUBLIC_PROGRESS_SCHEMA,
        "task_id": state["task_id"],
        "page_type": page_type,
        "boundary": boundary,
        "policy_visible_input_only": True,
        "gradient_attached": False,
        "public_state_sha256": public_state_anchor_signature(observation),
        "goal_constraint_identity_sha256": sha256_json(
            {
                "product_constraints": _goal_product_constraints(goal),
                "goal_options": _goal_options(goal),
                "price_upper": goal.get("price_upper"),
            }
        ),
        "candidate_match_count": candidate_matches,
        "candidate_constraint_count": candidate_total,
        "candidate_attribute_match_fraction": candidate_fraction,
        "item_match_count": item_matches,
        "item_constraint_count": item_total,
        "item_attribute_match_fraction": item_fraction,
        "selected_option_match_count": selected_matches,
        "selected_option_count": selected_total,
        "selected_option_fraction": selected_fraction,
        "potential": potential,
    }
    evidence["content_sha256"] = sha256_json(evidence)
    return evidence


def annotate_episode_with_verifier(
    episode: Mapping[str, Any],
    *,
    goal: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach detached post-transition potential evidence to a full episode."""

    _require(episode.get("rollout_valid") is True, "cannot annotate infrastructure-invalid M6 episode")
    turns = episode.get("turns")
    _require(isinstance(turns, list) and turns, "M6 episode has no policy turns")
    terminal = strict_terminal_reward(episode.get("task_score", episode.get("reward")))
    output = deepcopy(dict(episode))
    output_turns = output["turns"]
    for index, turn in enumerate(output_turns):
        post = turn.get("post_action_observation")
        _require(isinstance(post, Mapping), f"M6 turn {index + 1} lacks post-action public state")
        # Model-output failures do not execute an environment action.  They
        # remain charged policy turns, but transition shaping must be exactly
        # zero rather than re-scoring the unchanged state and inventing
        # progress.  The final boundary still equals strict outcome, preserving
        # telescoping over every generated turn used by the learner.
        if (
            turn.get("schema_valid") is not True
            or not isinstance(turn.get("action"), Mapping)
            or not isinstance(turn.get("action_result"), Mapping)
        ):
            previous = 0.0 if index == 0 else float(output_turns[index - 1]["verifier_progress_after_action"])
            if index + 1 == len(output_turns):
                previous = terminal
            evidence = {
                "schema_version": PUBLIC_PROGRESS_SCHEMA,
                "task_id": str(post.get("task_id", "")),
                "page_type": str(post.get("page_type", "")).casefold(),
                "boundary": "strict_episode_boundary" if index + 1 == len(output_turns) else "no_environment_transition",
                "policy_visible_input_only": True,
                "gradient_attached": False,
                "public_state_sha256": public_state_anchor_signature(post),
                "goal_constraint_identity_sha256": sha256_json(
                    {
                        "product_constraints": _goal_product_constraints(goal),
                        "goal_options": _goal_options(goal),
                        "price_upper": goal.get("price_upper"),
                    }
                ),
                "candidate_match_count": 0,
                "candidate_constraint_count": 0,
                "candidate_attribute_match_fraction": 0.0,
                "item_match_count": 0,
                "item_constraint_count": 0,
                "item_attribute_match_fraction": 0.0,
                "selected_option_match_count": 0,
                "selected_option_count": 0,
                "selected_option_fraction": 0.0,
                "potential": previous,
            }
            evidence["content_sha256"] = sha256_json(evidence)
        else:
            evidence = score_public_observation(
                goal=goal,
                observation=post,
                terminal_boundary_reward=terminal if index + 1 == len(output_turns) else None,
            )
        turn["verifier_public_state_sha256"] = evidence["public_state_sha256"]
        turn["verifier_progress_after_action"] = evidence["potential"]
        turn["verifier_progress_evidence"] = evidence
    # Recompute now so malformed/non-telescoping annotations fail immediately.
    trajectory_td_credit(
        task_score=float(episode.get("task_score", episode.get("reward"))),
        post_transition_potentials=[turn["verifier_progress_after_action"] for turn in output_turns],
    )
    return output


def trajectory_td_credit(
    *,
    task_score: float,
    post_transition_potentials: Sequence[float],
    tolerance: float = TELESCOPING_TOLERANCE,
) -> dict[str, Any]:
    """Compute telescoping rewards and zero-sum TD deviations for one rollout.

    ``post_transition_potentials`` has one value per generated policy turn;
    invalid outputs keep the preceding potential because no action ran. Values
    before the last turn must be public-state verifier potentials in
    ``[0, 0.9]``.  The
    final value must equal the strict official terminal reward; this makes an
    accidental dense-score or partial-match boundary a hard failure.
    """

    _require(post_transition_potentials, "M6 verifier-TD trajectory has no policy turns")
    _require(math.isfinite(tolerance) and tolerance > 0, "M6 telescoping tolerance is invalid")
    supplied = tuple(
        _finite(value, f"post-transition potential {index}")
        for index, value in enumerate(post_transition_potentials)
    )
    terminal = strict_terminal_reward(task_score)
    _require(
        all(0.0 <= value <= MAXIMUM_NONTERMINAL_POTENTIAL for value in supplied[:-1]),
        "M6 non-terminal potential left [0, 0.9]",
    )
    _require(abs(supplied[-1] - terminal) <= tolerance, "M6 terminal potential is not strict reward")
    state_potentials = [0.0, *supplied]
    dense_rewards = [
        state_potentials[index + 1] - state_potentials[index]
        for index in range(len(supplied))
    ]
    telescope_error = sum(dense_rewards) - terminal
    _require(abs(telescope_error) <= tolerance, "M6 verifier-TD telescoping invariant failed")
    baseline = terminal / len(supplied)
    deviations = [reward - baseline for reward in dense_rewards]
    conservation_error = sum(deviations)
    _require(abs(conservation_error) <= tolerance, "M6 verifier-TD zero-sum invariant failed")
    return {
        "strict_terminal_reward": terminal,
        "state_potentials": state_potentials,
        "dense_rewards": dense_rewards,
        "uniform_terminal_baseline": baseline,
        "td_deviations": deviations,
        "telescoping_error": telescope_error,
        "credit_conservation_error": conservation_error,
    }


def _validate_group(trajectories: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    _require(len(trajectories) == GROUP_SIZE, "M6 credit assignment requires exactly K=8 trajectories")
    task_ids: set[str] = set()
    validated: list[Mapping[str, Any]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        _require(isinstance(trajectory, Mapping), f"M6 trajectory {trajectory_index} is malformed")
        task_id = trajectory.get("task_id")
        _require(isinstance(task_id, str) and task_id, f"M6 trajectory {trajectory_index} lacks task_id")
        task_ids.add(task_id)
        score = trajectory.get("task_score", trajectory.get("reward"))
        terminal = strict_terminal_reward(score)
        success = trajectory.get("success")
        _require(isinstance(success, bool) and success is bool(terminal), "M6 success/task-score disagreement")
        turns = trajectory.get("turns")
        _require(isinstance(turns, list) and turns, f"M6 trajectory {trajectory_index} has no turns")
        for turn_index, turn in enumerate(turns):
            _require(isinstance(turn, Mapping), f"M6 trajectory {trajectory_index} turn {turn_index} is malformed")
            anchor = turn.get("verifier_public_state_sha256")
            _require(
                isinstance(anchor, str) and SHA256_RE.fullmatch(anchor) is not None,
                f"M6 trajectory {trajectory_index} turn {turn_index} lacks verifier public-state evidence",
            )
            _finite(
                turn.get("verifier_progress_after_action"),
                f"M6 trajectory {trajectory_index} turn {turn_index} verifier progress",
            )
        validated.append(trajectory)
    _require(len(task_ids) == 1, "M6 credit group crosses task identities")
    return tuple(validated)


def assign_group_credit(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    method: str = ANCHOR_METHOD,
    verifier_td_lambda: float | None = None,
    tolerance: float = TELESCOPING_TOLERANCE,
) -> dict[str, Any]:
    """Assign strict GRPO macro credit plus zero-sum within-trajectory credit."""

    _require(method in METHODS, "unsupported M6 mini RL method")
    expected_weight = 0.0 if method == BASELINE_METHOD else VERIFIER_TD_LAMBDA
    weight = expected_weight if verifier_td_lambda is None else _finite(verifier_td_lambda, "M6 verifier-TD lambda")
    _require(weight == expected_weight, "M6 verifier-TD lambda/method drift")
    validated = _validate_group(trajectories)
    strict_rewards = [
        strict_terminal_reward(trajectory.get("task_score", trajectory.get("reward")))
        for trajectory in validated
    ]
    macro_advantages = standardized_advantages(strict_rewards)
    trajectory_reports: list[dict[str, Any]] = []
    turn_credit: list[list[dict[str, Any]]] = []
    for trajectory_index, trajectory in enumerate(validated):
        turns = trajectory["turns"]
        td = trajectory_td_credit(
            task_score=float(trajectory.get("task_score", trajectory.get("reward"))),
            post_transition_potentials=[turn["verifier_progress_after_action"] for turn in turns],
            tolerance=tolerance,
        )
        trajectory_reports.append(td)
        turn_credit.append(
            [
                {
                    "turn_index": turn_index + 1,
                    "verifier_public_state_sha256": turn["verifier_public_state_sha256"],
                    "strict_macro_advantage": macro_advantages[trajectory_index],
                    "dense_reward": td["dense_rewards"][turn_index],
                    "td_deviation": td["td_deviations"][turn_index],
                    "turn_advantage": (
                        macro_advantages[trajectory_index]
                        + weight * td["td_deviations"][turn_index]
                    ),
                }
                for turn_index, turn in enumerate(turns)
            ]
        )

    flat_turns = [item for trajectory in turn_credit for item in trajectory]
    failure_reports = [
        report for reward, report in zip(strict_rewards, trajectory_reports) if reward == 0.0
    ]
    failed_with_positive_preterminal = sum(
        any(float(value) > tolerance for value in item["state_potentials"][1:-1])
        for item in failure_reports
    )
    failed_with_negative_terminal = sum(
        bool(item["dense_rewards"] and item["dense_rewards"][-1] < -tolerance)
        for item in failure_reports
    )
    report = {
        "schema_version": "m6_verifier_td_credit_assignment_v1",
        "formula_version": FORMULA_VERSION_BY_METHOD[method],
        "method": method,
        "task_id": validated[0]["task_id"],
        "K": GROUP_SIZE,
        "verifier_td_lambda": weight,
        "telescoping_tolerance": tolerance,
        "terminal_reward_contract": "1[official task_score >= 0.999]",
        "potential_contract": {
            "initial": 0.0,
            "nonterminal_minimum": 0.0,
            "nonterminal_maximum": MAXIMUM_NONTERMINAL_POTENTIAL,
            "terminal": "strict_terminal_reward",
            "policy_visible": False,
            "gradient_attached": False,
        },
        "strict_rewards": strict_rewards,
        "macro_advantages": list(macro_advantages),
        "trajectory_td": trajectory_reports,
        "turn_credit": turn_credit,
        "metrics": {
            "mixed_strict_reward_signal": len(set(strict_rewards)) == 2,
            "strict_success_fraction": sum(strict_rewards) / GROUP_SIZE,
            "turn_count": len(flat_turns),
            "nonzero_td_turn_count": sum(
                abs(float(item["td_deviation"])) > tolerance for item in flat_turns
            ),
            "nonzero_td_turn_fraction": sum(
                abs(float(item["td_deviation"])) > tolerance for item in flat_turns
            ) / len(flat_turns),
            "nonzero_optimizer_turn_count": sum(
                abs(float(item["turn_advantage"])) > tolerance for item in flat_turns
            ),
            "nonzero_optimizer_turn_fraction": sum(
                abs(float(item["turn_advantage"])) > tolerance for item in flat_turns
            ) / len(flat_turns),
            "maximum_absolute_telescoping_error": max(
                abs(float(item["telescoping_error"])) for item in trajectory_reports
            ),
            "maximum_absolute_credit_conservation_error": max(
                abs(float(item["credit_conservation_error"])) for item in trajectory_reports
            ),
            "failed_trajectory_count": len(failure_reports),
            "failed_trajectory_with_positive_preterminal_potential_count": failed_with_positive_preterminal,
            "failed_trajectory_with_negative_terminal_credit_count": failed_with_negative_terminal,
            "positive_progress_failure_terminal_cancellation_complete": (
                failed_with_positive_preterminal == failed_with_negative_terminal
            ),
        },
        "normalization_contract": "population standardization over strict terminal rewards within same-task K8",
    }
    report["content_sha256"] = sha256_json(report)
    return report


def validate_credit_assignment(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == "m6_verifier_td_credit_assignment_v1", "M6 credit schema drift")
    method = value.get("method")
    _require(method in METHODS and value.get("K") == GROUP_SIZE, "M6 credit method/K drift")
    _require(value.get("formula_version") == FORMULA_VERSION_BY_METHOD[method], "M6 credit formula drift")
    _require(
        value.get("verifier_td_lambda") == (0.0 if method == BASELINE_METHOD else VERIFIER_TD_LAMBDA),
        "M6 credit lambda/method drift",
    )
    metrics = value.get("metrics")
    _require(isinstance(metrics, Mapping), "M6 credit metrics are missing")
    _require(
        _finite(metrics.get("maximum_absolute_telescoping_error"), "M6 telescoping metric")
        <= TELESCOPING_TOLERANCE,
        "M6 credit report violates telescoping",
    )
    _require(
        _finite(metrics.get("maximum_absolute_credit_conservation_error"), "M6 conservation metric")
        <= TELESCOPING_TOLERANCE,
        "M6 credit report violates conservation",
    )
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 credit self-hash drift")
    return value
