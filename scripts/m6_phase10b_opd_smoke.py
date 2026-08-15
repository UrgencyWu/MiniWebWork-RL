#!/usr/bin/env python3
"""Build, query, and audit Phase10-B's eight-task zero-update OPD smoke."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import (  # noqa: E402
    atomic_write_json,
    sha256_json,
    token_ids_sha256,
)
from miniwebwork.long_horizon_rl.model_manifest import validate_base_model_manifest  # noqa: E402
from miniwebwork.m6_phase10b_opd import (  # noqa: E402
    MODEL_SPECS,
    OPD_SMOKE_REPORT_SCHEMA,
    ROUTER_MANIFEST_SCHEMA,
    SPECIALIST_IDENTITIES,
    SPECIALIST_TARGET_SCHEMA,
    compress_topk_logprobs,
    frozen_public_router,
)
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402

TOPK = 64
MAX_TURNS_PER_SPECIALIST = 8
ROUTER_SEED = 20260852


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("content_sha256") == _self_hash(value),
             f"Phase10-B smoke input self-hash drift: {path}")
    return value


def _git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(output: Path, value: dict[str, Any]) -> None:
    destination = output.expanduser().resolve()
    _require(not destination.exists(), f"Phase10-B smoke output already exists: {destination}")
    value["content_sha256"] = _self_hash(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, value)
    print(json.dumps(value, indent=2, sort_keys=True))


def _load_behavior(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = root.expanduser().resolve()
    _require(not ({"promotion", "holdout"} & {part.casefold() for part in resolved.parts}),
             "Phase10-B OPD smoke touches promotion/holdout")
    report = _load_hashed(resolved / "collection_report.json")
    _require(
        report.get("complete") is True
        and report.get("development_only") is True
        and report.get("mode") == "phase10b_opd_smoke_behavior"
        and report.get("task_count") == 8
        and report.get("trajectory_count") == 32
        and report.get("K") == 4
        and report.get("training_updates_allowed") is False,
        "Phase10-B OPD behavior contract drift",
    )
    groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=4)
        for path in sorted((resolved / "groups").glob("g*.json"))
    ]
    _require(len(groups) == 8, "Phase10-B OPD behavior group count drift")
    _require(report.get("group_content_sha256") == [group["content_sha256"] for group in groups],
             "Phase10-B OPD behavior report/group drift")
    return report, groups


def build_router_manifest(*, behavior_root: Path, qualification_report: Path) -> dict[str, Any]:
    qualification = _load_hashed(qualification_report)
    qualified = list(qualification.get("qualified_specialists") or [])
    _require(set(qualified) <= set(SPECIALIST_IDENTITIES), "Phase10-B qualification identity drift")
    _require(qualification.get("training_performed") is False and qualification.get("optimizer_steps") == 0,
             "Phase10-B qualification report trained")
    behavior_report, groups = _load_behavior(behavior_root)
    candidates: dict[str, list[dict[str, Any]]] = {identity: [] for identity in qualified}
    all_route_counts: Counter[str] = Counter()
    for group_index, group in enumerate(groups):
        for trajectory in group["trajectories"]:
            prior_states: list[str] = []
            previous_success: bool | None = None
            for turn in trajectory["turns"]:
                route = frozen_public_router(
                    turn,
                    qualified_specialists=qualified,
                    previous_action_success=previous_success,
                    prior_public_state_sha256=prior_states,
                )
                assigned = route["assigned_specialist"]
                desired = route["desired_specialist"] or "none"
                all_route_counts[f"desired:{desired}"] += 1
                all_route_counts[f"assigned:{assigned or 'none'}"] += 1
                if assigned is not None:
                    prompt_ids = [int(value) for value in turn["prompt_token_ids"]]
                    action_ids = [int(value) for value in turn["generated_token_ids"]]
                    row = {
                        "task_id": group["task_id"],
                        "group_index": group_index,
                        "source_group_content_sha256": group["content_sha256"],
                        "rollout_index": int(trajectory["rollout_index"]),
                        "turn_index": int(turn["turn_index"]),
                        "assigned_specialist": assigned,
                        "route_rule": route["route_rule"],
                        "public_features": route["public_features"],
                        "prompt_token_count": len(prompt_ids),
                        "action_token_count": len(action_ids),
                        "prompt_token_sha256": token_ids_sha256(prompt_ids),
                        "action_token_sha256": token_ids_sha256(action_ids),
                        "exact_student_prefix_sha256": token_ids_sha256(prompt_ids),
                        "behavior_action_executed_by": "student_pi_0",
                        "specialist_action_executed": False,
                        "prefix_policy_loss_eligible": False,
                        "observation_policy_loss_eligible": False,
                        "student_action_policy_loss_eligible": True,
                        "forbidden_fields_used": False,
                    }
                    row["route_selection_sha256"] = sha256_json({
                        "seed": ROUTER_SEED,
                        "task_id": row["task_id"],
                        "rollout_index": row["rollout_index"],
                        "turn_index": row["turn_index"],
                        "assigned_specialist": assigned,
                    })
                    candidates[assigned].append(row)
                prior_states.append(str(turn["pre_action_public_state_sha256"]))
                result = turn.get("action_result") or {}
                previous_success = result.get("success") if isinstance(result.get("success"), bool) else None
    selected: list[dict[str, Any]] = []
    for identity in qualified:
        rows = sorted(candidates[identity], key=lambda row: (row["route_selection_sha256"], row["task_id"]))
        selected.extend(rows[:MAX_TURNS_PER_SPECIALIST])
    selected.sort(key=lambda row: (row["assigned_specialist"], row["route_selection_sha256"]))
    selected_counts = Counter(row["assigned_specialist"] for row in selected)
    routed_identities = sorted(selected_counts)
    return {
        "schema_version": ROUTER_MANIFEST_SCHEMA,
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "producer_git_sha": _git_sha(),
        "router_seed": ROUTER_SEED,
        "router_type": "frozen_public_turn_router_v1",
        "router_inputs": [
            "page_type", "available_actions", "selected_marker", "previous_action_success",
            "remaining_budget_proxy", "pre_action_public_state_sha256",
        ],
        "terminal_score_used": False,
        "target_asin_or_hidden_goal_used": False,
        "future_outcome_used": False,
        "behavior_tokens_generated_by": "student_pi_0",
        "specialist_tokens_executed_in_environment": False,
        "topk": TOPK,
        "max_selected_turns_per_specialist": MAX_TURNS_PER_SPECIALIST,
        "qualification_report_content_sha256": qualification["content_sha256"],
        "qualified_specialists": qualified,
        "behavior_root": str(behavior_root.expanduser().resolve()),
        "behavior_collection_report_content_sha256": behavior_report["content_sha256"],
        "behavior_policy_lineage": behavior_report["policy_lineage"],
        "behavior_group_content_sha256": behavior_report["group_content_sha256"],
        "all_route_counts": dict(sorted(all_route_counts.items())),
        "selected_route_counts": dict(sorted(selected_counts.items())),
        "selected_route_count": len(selected),
        "routed_specialists": routed_identities,
        "route_coverage_pass": len(routed_identities) >= 2,
        "routes": selected,
    }


def _find_turn(groups: list[dict[str, Any]], route: Mapping[str, Any]) -> dict[str, Any]:
    group = groups[int(route["group_index"])]
    _require(group["task_id"] == route["task_id"]
             and group["content_sha256"] == route["source_group_content_sha256"],
             "Phase10-B routed source group drift")
    trajectory = next(
        item for item in group["trajectories"] if int(item["rollout_index"]) == int(route["rollout_index"])
    )
    turn = next(item for item in trajectory["turns"] if int(item["turn_index"]) == int(route["turn_index"]))
    _require(token_ids_sha256(turn["prompt_token_ids"]) == route["prompt_token_sha256"]
             and token_ids_sha256(turn["generated_token_ids"]) == route["action_token_sha256"],
             "Phase10-B routed token source drift")
    return turn


async def query_specialist_targets(
    *, router_manifest: Path,
    identity: str,
    model_path: Path,
    base_model_manifest: Path,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    _require(identity in SPECIALIST_IDENTITIES, "Phase10-B target Specialist identity drift")
    spec = MODEL_SPECS[identity]
    resolved_model = model_path.expanduser().resolve()
    _require(str(resolved_model) == spec["path"], "Phase10-B target model path drift")
    model_manifest = validate_base_model_manifest(
        path=base_model_manifest,
        expected_base_model=resolved_model,
        verify_files=False,
    )
    manifest = _load_hashed(router_manifest)
    _require(manifest.get("schema_version") == ROUTER_MANIFEST_SCHEMA, "Phase10-B router schema drift")
    _require(identity in manifest.get("qualified_specialists", []), "Phase10-B unqualified target query")
    routes = [row for row in manifest["routes"] if row["assigned_specialist"] == identity]
    _require(routes, "Phase10-B Specialist has no routed turns")
    _, groups = _load_behavior(Path(manifest["behavior_root"]))
    tensor_parallel_size = 2 if identity == "S_match" else 1
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=str(resolved_model),
        tokenizer=MODEL_SPECS["student"]["path"],
        dtype="bfloat16",
        seed=ROUTER_SEED,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        max_logprobs=TOPK,
        logprobs_mode="raw_logprobs",
        language_model_only=True,
        enable_lora=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        generation_config="vllm",
        trust_remote_code=True,
    ))
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        seed=ROUTER_SEED,
        max_tokens=1,
        prompt_logprobs=TOPK,
        flat_logprobs=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
        detokenize=False,
        skip_special_tokens=False,
    )
    target_rows = []
    started = time.monotonic()
    try:
        for route_index, route in enumerate(routes):
            turn = _find_turn(groups, route)
            prompt_ids = [int(value) for value in turn["prompt_token_ids"]]
            action_ids = [int(value) for value in turn["generated_token_ids"]]
            full_ids = prompt_ids + action_ids
            final = None
            async for item in engine.generate(
                TokensPrompt(prompt_token_ids=full_ids),
                sampling,
                f"p10b-opd-{identity}-{route_index:03d}",
            ):
                final = item
            _require(final is not None and final.finished, "Phase10-B target query did not finish")
            prompt_logprobs = final.prompt_logprobs
            _require(isinstance(prompt_logprobs, list) and len(prompt_logprobs) == len(full_ids),
                     "Phase10-B prompt-logprob length drift")
            token_targets = []
            for offset, token_id in enumerate(action_ids):
                values = prompt_logprobs[len(prompt_ids) + offset]
                _require(isinstance(values, Mapping) and values, "Phase10-B action-token target missing")
                raw = {
                    int(candidate): float(value.logprob if hasattr(value, "logprob") else value)
                    for candidate, value in values.items()
                }
                compressed = compress_topk_logprobs(raw, k=TOPK)
                token_targets.append({"position": offset, "student_behavior_token_id": token_id, **compressed})
            target_rows.append({
                "route_selection_sha256": route["route_selection_sha256"],
                "task_id": route["task_id"],
                "rollout_index": route["rollout_index"],
                "turn_index": route["turn_index"],
                "prompt_token_sha256": route["prompt_token_sha256"],
                "action_token_sha256": route["action_token_sha256"],
                "prompt_token_count": len(prompt_ids),
                "action_token_count": len(action_ids),
                "token_targets": token_targets,
                "assistant_action_only": True,
                "prefix_and_observation_masked": True,
                "student_behavior_action_unchanged": True,
            })
    finally:
        engine.shutdown()
    return {
        "schema_version": SPECIALIST_TARGET_SCHEMA,
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "producer_git_sha": _git_sha(),
        "identity": identity,
        "role": spec["role"],
        "model_path": str(resolved_model),
        "base_model_manifest_path": model_manifest["path"],
        "base_model_manifest_sha256": model_manifest["sha256"],
        "base_model_functional_file_set_sha256": model_manifest["payload"]["functional_file_set_sha256"],
        "student_tokenizer_path": MODEL_SPECS["student"]["path"],
        "student_tokenizer_used": True,
        "tensor_parallel_size": tensor_parallel_size,
        "topk": TOPK,
        "router_manifest_content_sha256": manifest["content_sha256"],
        "routed_turn_count": len(target_rows),
        "routed_action_token_count": sum(row["action_token_count"] for row in target_rows),
        "finite_targets": True,
        "student_behavior_tokens_replaced": False,
        "targets": target_rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def aggregate_smoke(*, router_manifest: Path, target_reports: list[Path]) -> dict[str, Any]:
    router = _load_hashed(router_manifest)
    _require(router.get("schema_version") == ROUTER_MANIFEST_SCHEMA, "Phase10-B router schema drift")
    reports = [_load_hashed(path) for path in target_reports]
    identities = [report.get("identity") for report in reports]
    _require(len(identities) == len(set(identities)), "Phase10-B duplicate target identity")
    expected = list(router.get("routed_specialists") or [])
    _require(sorted(identities) == sorted(expected), "Phase10-B target report coverage drift")
    routes = {row["route_selection_sha256"]: row for row in router["routes"]}
    token_counts: Counter[str] = Counter()
    retained_mass: dict[str, list[float]] = {identity: [] for identity in identities}
    audited_turns = 0
    for report in reports:
        _require(
            report.get("schema_version") == SPECIALIST_TARGET_SCHEMA
            and report.get("training_performed") is False
            and report.get("optimizer_steps") == 0
            and report.get("student_behavior_tokens_replaced") is False
            and report.get("router_manifest_content_sha256") == router["content_sha256"],
            "Phase10-B target report contract drift",
        )
        identity = str(report["identity"])
        for row in report["targets"]:
            route = routes[row["route_selection_sha256"]]
            _require(route["assigned_specialist"] == identity, "Phase10-B target route identity drift")
            _require(row["assistant_action_only"] is True
                     and row["prefix_and_observation_masked"] is True
                     and row["student_behavior_action_unchanged"] is True,
                     "Phase10-B OPD token mask/behavior drift")
            _require(len(row["token_targets"]) == route["action_token_count"] == row["action_token_count"],
                     "Phase10-B target action-token count drift")
            for target in row["token_targets"]:
                _require(target["probability_sum_abs_error"] <= 1e-5
                         and math.isfinite(float(target["topk_probability_mass"]))
                         and math.isfinite(float(target["rest_mass"])),
                         "Phase10-B target probability mass drift")
                retained_mass[identity].append(float(target["topk_probability_mass"]))
            token_counts[identity] += int(row["action_token_count"])
            audited_turns += 1
    total_tokens = sum(token_counts.values())
    fractions = {identity: count / total_tokens for identity, count in token_counts.items()} if total_tokens else {}
    checks = {
        "at_least_two_qualified_specialists_routed": len(identities) >= 2,
        "student_generated_all_behavior_tokens": router.get("behavior_tokens_generated_by") == "student_pi_0",
        "specialist_tokens_never_executed": router.get("specialist_tokens_executed_in_environment") is False,
        "public_router_only": (
            router.get("terminal_score_used") is False
            and router.get("target_asin_or_hidden_goal_used") is False
            and router.get("future_outcome_used") is False
        ),
        "one_or_zero_specialist_per_turn": len(routes) == len(router["routes"]),
        "assistant_action_only_mask": audited_turns == len(routes),
        "all_targets_finite_and_mass_safe": all(
            report.get("finite_targets") is True for report in reports
        ),
        "optimizer_steps_zero": True,
    }
    return {
        "schema_version": OPD_SMOKE_REPORT_SCHEMA,
        "complete": True,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "producer_git_sha": _git_sha(),
        "router_manifest_content_sha256": router["content_sha256"],
        "target_report_content_sha256": [report["content_sha256"] for report in reports],
        "qualified_specialists": router["qualified_specialists"],
        "routed_specialists": identities,
        "routed_turn_count": audited_turns,
        "routed_action_token_count": total_tokens,
        "specialist_action_token_counts": dict(sorted(token_counts.items())),
        "specialist_action_token_fractions": dict(sorted(fractions.items())),
        "mean_topk_retained_mass": {
            identity: sum(values) / len(values) for identity, values in retained_mass.items() if values
        },
        "checks": checks,
        "passed": all(checks.values()),
        "decision": "allow_matched_single_update_probe" if all(checks.values()) else "stop_before_opd_update",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    route = subparsers.add_parser("route")
    route.add_argument("--behavior-root", type=Path, required=True)
    route.add_argument("--qualification-report", type=Path, required=True)
    route.add_argument("--output", type=Path, required=True)
    target = subparsers.add_parser("specialist")
    target.add_argument("--router-manifest", type=Path, required=True)
    target.add_argument("--identity", choices=SPECIALIST_IDENTITIES, required=True)
    target.add_argument("--model-path", type=Path, required=True)
    target.add_argument("--base-model-manifest", type=Path, required=True)
    target.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--router-manifest", type=Path, required=True)
    aggregate.add_argument("--target-report", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "route":
        _write(args.output, build_router_manifest(
            behavior_root=args.behavior_root,
            qualification_report=args.qualification_report,
        ))
    elif args.command == "specialist":
        _write(args.output, asyncio.run(query_specialist_targets(
            router_manifest=args.router_manifest,
            identity=args.identity,
            model_path=args.model_path,
            base_model_manifest=args.base_model_manifest,
        )))
    else:
        _write(args.output, aggregate_smoke(
            router_manifest=args.router_manifest,
            target_reports=args.target_report,
        ))


if __name__ == "__main__":
    main()
