"""Frozen M5 WebShop evaluation identities, authorization and artifact checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..long_horizon_rl.adapter_view import validate_vllm_adapter_view
from ..long_horizon_rl.contracts import SHA256_PATTERN, directory_sha256, sha256_file, sha256_json
from ..long_horizon_rl.model_manifest import validate_base_model_manifest
from ..m5_webshop_protocol import eligible_goal_indices, task_id_for_goal_index
from .formal_training import SEEDS, self_hash, validate_run_report, validate_self_hashed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_PLAN_PATH = PROJECT_ROOT / "data" / "m5_webshop_frozen_eval_plan_v1.json"
EVAL_ROOT = PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "formal" / "frozen_test"
PLAN_SCHEMA = "m5_webshop_frozen_eval_plan_v1"
AUTHORIZATION_SCHEMA = "m5_webshop_frozen_eval_authorization_v1"
ATTEMPT_SCHEMA = "m5_webshop_frozen_eval_attempt_v1"
RUN_REPORT_SCHEMA = "m5_webshop_frozen_eval_run_v1"
IDENTITIES = (
    "raw_base_model",
    "shared_verified_sft",
    "multi_turn_grpo_seed_20260801",
    "multi_turn_grpo_seed_20260802",
    "multi_turn_grpo_seed_20260803",
    "anchor_gigpo_seed_20260801",
    "anchor_gigpo_seed_20260802",
    "anchor_gigpo_seed_20260803",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"M5 evaluation JSON root is not an object: {path}")
    return payload


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def frozen_test_task_ids() -> tuple[str, ...]:
    indices = eligible_goal_indices("test")
    _require(indices == tuple(range(500)), "M5 frozen test roster drift")
    return tuple(task_id_for_goal_index(index) for index in indices)


def evaluation_run_root(identity: str, root: Path = EVAL_ROOT) -> Path:
    _require(identity in IDENTITIES, "M5 frozen evaluation identity drift")
    return Path(root).expanduser().resolve() / identity


def validate_eval_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(payload)
    _require(plan.get("schema_version") == PLAN_SCHEMA, "M5 evaluation plan schema drift")
    _require(plan.get("study_id") == "m5_webshop_credit_assignment_v1", "M5 evaluation study drift")
    _require(plan.get("producer_training_git_sha") == "b9b5221777427b0d7decaf7b12c6572ac9a1d3f3", "M5 training producer drift")
    _require(plan.get("protocol_sha256") == "a49881137e90a7bc6b2747e36a7e129ea37c52924d9ab965ea549e8be98a8293", "M5 evaluation protocol hash drift")
    _require(plan.get("formal_training_plan_sha256") == "51fbe550ed8e3dd6064ac5f98a71fd08e3be4f7f9606adf3cb111d1479ed985b", "M5 formal training plan hash drift")
    test = plan.get("test")
    _require(isinstance(test, Mapping), "M5 evaluation test contract missing")
    _require(
        test.get("split") == "test"
        and test.get("task_count") == 500
        and test.get("goal_index_start_inclusive") == 0
        and test.get("goal_index_stop_exclusive") == 500
        and test.get("rollouts_per_task") == 4
        and test.get("trajectory_count_per_identity") == 2000
        and test.get("total_trajectory_count") == 16000,
        "M5 frozen test matrix drift",
    )
    _require(
        test.get("evaluation_seed") == 20260812
        and test.get("shared_prefix_turns") == 0
        and test.get("maximum_environment_steps") == 15
        and test.get("maximum_model_turns") == 18
        and test.get("maximum_new_tokens_per_turn") == 128
        and test.get("sampling") == {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
        "M5 evaluation sampling drift",
    )
    _require(_sha(test.get("goals_sha256")), "M5 frozen goals hash drift")
    _require(
        test.get("stratification")
        == {
            "category": "string goal.category, otherwise unknown",
            "constraint_count": "count non-empty goal.attributes plus non-empty goal.goal_options entries plus one finite goal.price_upper",
        },
        "M5 evaluation stratification drift",
    )
    inference = plan.get("inference")
    _require(
        isinstance(inference, Mapping)
        and inference.get("maximum_sequence_tokens") == 8192
        and inference.get("maximum_sequences") == 32
        and inference.get("concurrent_k4_groups") == 8
        and inference.get("dtype") == "bfloat16"
        and inference.get("enforce_eager") is True
        and math.isclose(float(inference.get("gpu_memory_utilization", -1)), 0.5),
        "M5 evaluation inference drift",
    )
    _require(
        inference.get("prompt_path") == "prompts/webshop_agent_v1_compact.txt"
        and inference.get("prompt_sha256") == "c33178c5c9a4e3f9cf683e6a1debb1e2f762a3cd9f27338292c8e8fa34f712aa"
        and inference.get("chat_template_kwargs") == {"enable_thinking": False},
        "M5 evaluation prompt/chat-template drift",
    )
    base_model = plan.get("base_model")
    _require(
        isinstance(base_model, Mapping)
        and base_model.get("path") == "/data/share/model/Qwen3.5-4B"
        and base_model.get("manifest_sha256") == "290ecd9ec4eaa1f5ac6927b10e9cb4c600d22aec78a6d743baa8a01d72c1b7a3"
        and base_model.get("functional_file_set_sha256") == "6b2cdb9cf894cec7eb1dcef2a57682a9d73e22f2a81a58ef3b1f32854f031b85",
        "M5 evaluation base-model identity drift",
    )
    resources = plan.get("resources")
    _require(
        isinstance(resources, Mapping)
        and resources.get("wall_time_per_allocation") == "24:00:00"
        and resources.get("gpus_per_identity") == 1
        and resources.get("cpus_per_identity") == 6
        and resources.get("memory_gib_per_identity") == 24
        and resources.get("maximum_parallel_identities") == 8,
        "M5 evaluation resource drift",
    )
    success = plan.get("success_contract")
    _require(
        isinstance(success, Mapping)
        and success.get("exact_task_count") == 500
        and success.get("exact_trajectory_count") == 2000
        and success.get("minimum_infrastructure_valid_attempt_fraction") == 0.98
        and success.get("nonempty_gpu_telemetry") is True
        and success.get("training_updates_allowed") is False,
        "M5 evaluation success contract drift",
    )
    identities = plan.get("identities")
    _require(isinstance(identities, list) and len(identities) == 8, "M5 evaluation identity matrix missing")
    _require(tuple(item.get("identity") for item in identities) == IDENTITIES, "M5 evaluation identity order drift")
    expected_methods = {
        "raw_base_model": ("raw_base_model", None),
        "shared_verified_sft": ("shared_verified_sft", None),
        **{f"multi_turn_grpo_seed_{seed}": ("multi_turn_grpo", seed) for seed in SEEDS},
        **{f"anchor_gigpo_seed_{seed}": ("anchor_gigpo", seed) for seed in SEEDS},
    }
    for item in identities:
        identity = item["identity"]
        _require((item.get("method"), item.get("training_seed")) == expected_methods[identity], f"M5 {identity} method/seed drift")
        if identity == "raw_base_model":
            _require(
                item.get("kind") == "raw"
                and all(
                    item.get(field) is None
                    for field in (
                        "adapter_path",
                        "adapter_sha256",
                        "rollout_adapter_path",
                        "rollout_adapter_sha256",
                        "adapter_semantic_sha256",
                        "training_report_path",
                        "training_report_file_sha256",
                        "training_report_content_sha256",
                    )
                ),
                "M5 raw identity drift",
            )
            continue
        _require(item.get("kind") == "adapter", f"M5 adapter identity kind drift: {identity}")
        for field in ("adapter_sha256", "rollout_adapter_sha256", "adapter_semantic_sha256"):
            _require(_sha(item.get(field)), f"M5 {identity} {field} drift")
        _require(isinstance(item.get("adapter_path"), str) and isinstance(item.get("rollout_adapter_path"), str), f"M5 {identity} path drift")
        if identity == "shared_verified_sft":
            _require(item.get("training_seed") is None and item.get("training_report_path") is None, "M5 SFT identity drift")
        else:
            _require(item.get("training_seed") in SEEDS, f"M5 {identity} seed drift")
            _require(_sha(item.get("training_report_file_sha256")) and _sha(item.get("training_report_content_sha256")), f"M5 {identity} training report drift")
    recovery = plan.get("recovery")
    _require(
        isinstance(recovery, Mapping)
        and recovery.get("same_root_resume") is True
        and recovery.get("maximum_infrastructure_attempts_per_group") == 4
        and recovery.get("deterministic_failure") == "stop without successor",
        "M5 evaluation recovery drift",
    )
    _require(plan.get("output_root_template") == "outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/{identity}", "M5 evaluation output-root drift")
    return plan


def load_eval_plan(path: Path = EVAL_PLAN_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = validate_eval_plan(_json(resolved))
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}


def identity_spec(plan: Mapping[str, Any], identity: str) -> dict[str, Any]:
    _require(identity in IDENTITIES, "M5 frozen evaluation identity drift")
    return next(dict(item) for item in plan["identities"] if item["identity"] == identity)


def _validate_static_inference(
    plan: Mapping[str, Any],
    *,
    verify_base_model_files: bool,
) -> dict[str, Any]:
    _require(sha256_file(PROJECT_ROOT / "data" / "m5_webshop_study_v1.json") == plan["protocol_sha256"], "M5 protocol bytes drift")
    _require(sha256_file(PROJECT_ROOT / "data" / "m5_webshop_formal_plan_v1.json") == plan["formal_training_plan_sha256"], "M5 training plan bytes drift")
    prompt = PROJECT_ROOT / plan["inference"]["prompt_path"]
    _require(sha256_file(prompt) == plan["inference"]["prompt_sha256"], "M5 inference prompt bytes drift")
    model = validate_base_model_manifest(verify_files=verify_base_model_files)
    _require(model["sha256"] == plan["base_model"]["manifest_sha256"], "M5 base-model manifest drift")
    _require(model["payload"]["functional_file_set_sha256"] == plan["base_model"]["functional_file_set_sha256"], "M5 base-model bytes drift")
    return model


def validate_inference_identity(
    plan: Mapping[str, Any],
    *,
    identity: str,
    verify_base_model_files: bool,
) -> dict[str, Any]:
    model = _validate_static_inference(plan, verify_base_model_files=verify_base_model_files)
    item = identity_spec(plan, identity)
    if item["kind"] == "raw":
        return {"kind": "raw", "base_model_manifest_sha256": model["sha256"]}
    adapter = PROJECT_ROOT / item["adapter_path"]
    view = PROJECT_ROOT / item["rollout_adapter_path"]
    base_model = Path(plan["base_model"]["path"])
    _require(directory_sha256(adapter) == item["adapter_sha256"], f"M5 {identity} adapter hash drift")
    audit = validate_vllm_adapter_view(source_adapter=adapter, view_directory=view, base_model=base_model)
    _require(audit["view_directory_sha256"] == item["rollout_adapter_sha256"], f"M5 {identity} rollout-view hash drift")
    _require(audit["semantic_tensor_sha256"] == item["adapter_semantic_sha256"], f"M5 {identity} semantic hash drift")
    if item["training_report_path"] is not None:
        report_path = PROJECT_ROOT / item["training_report_path"]
        _require(sha256_file(report_path) == item["training_report_file_sha256"], f"M5 {identity} training-report file drift")
        report = validate_run_report(_json(report_path))
        _require(report["content_sha256"] == item["training_report_content_sha256"], f"M5 {identity} training-report content drift")
        _require(report["git_sha"] == plan["producer_training_git_sha"], f"M5 {identity} training producer drift")
        _require(report["method"] == item["method"] and report["seed"] == item["training_seed"], f"M5 {identity} training identity drift")
        _require(report["final_adapter_sha256"] == item["adapter_sha256"], f"M5 {identity} final adapter/report drift")
    return {
        "kind": "adapter",
        "adapter_sha256": item["adapter_sha256"],
        "rollout_adapter_sha256": item["rollout_adapter_sha256"],
        "adapter_semantic_sha256": item["adapter_semantic_sha256"],
    }


def validate_all_inference_identities(
    plan: Mapping[str, Any],
    *,
    verify_base_model_files: bool = True,
) -> dict[str, Any]:
    """Validate every frozen inference identity before the test is opened."""

    output: dict[str, Any] = {}
    for item in plan["identities"]:
        identity = item["identity"]
        output[identity] = validate_inference_identity(
            plan,
            identity=identity,
            verify_base_model_files=verify_base_model_files and identity == IDENTITIES[0],
        )
    return output


def validate_eval_authorization(
    payload: Mapping[str, Any],
    *,
    consumer_git_sha: str,
    plan_sha256: str,
) -> dict[str, Any]:
    authorization = validate_self_hashed(payload, schema=AUTHORIZATION_SCHEMA)
    _require(authorization.get("frozen_evaluation_submission_allowed") is True, "M5 frozen evaluation authorization is closed")
    _require(authorization.get("approval_scope") == "eight_frozen_evaluation_identities_only", "M5 evaluation authorization scope drift")
    _require(authorization.get("consumer_git_sha") == consumer_git_sha, "M5 evaluation consumer Git drift")
    _require(authorization.get("producer_training_git_sha") == "b9b5221777427b0d7decaf7b12c6572ac9a1d3f3", "M5 evaluation training producer drift")
    _require(authorization.get("eval_plan_sha256") == plan_sha256, "M5 evaluation authorization/plan drift")
    _require(tuple(authorization.get("identities", ())) == IDENTITIES, "M5 authorized evaluation identity matrix drift")
    _require(authorization.get("logical_job_count") == 8, "M5 authorized evaluation job count drift")
    return authorization


def load_eval_authorization(
    path: Path,
    *,
    consumer_git_sha: str,
    plan_sha256: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = validate_eval_authorization(_json(resolved), consumer_git_sha=consumer_git_sha, plan_sha256=plan_sha256)
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}


def summarize_groups(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(groups) == 500, "M5 frozen evaluation group count drift")
    trajectories = [item for group in groups for item in group["trajectories"]]
    _require(len(trajectories) == 2000, "M5 frozen evaluation trajectory count drift")
    scores = [float(item["reward"]) for item in trajectories]
    successes = [bool(item["success"]) for item in trajectories]
    turns = [turn for item in trajectories for turn in item["turns"]]
    invalid = sum(not bool(turn.get("schema_valid")) for turn in turns)
    action_errors = sum(
        bool((turn.get("action_result") or {}).get("error_code"))
        for item in trajectories
        for turn in item.get("evaluation_turn_summary", [])
    )
    task_success = [sum(bool(item["success"]) for item in group["trajectories"]) / 4 for group in groups]
    task_scores = [sum(float(item["reward"]) for item in group["trajectories"]) / 4 for group in groups]
    category: dict[str, dict[str, float]] = {}
    constraints: dict[str, dict[str, float]] = {}
    for group in groups:
        metadata = group.get("task_metadata")
        _require(isinstance(metadata, Mapping), "M5 evaluation task metadata missing")
        for bucket, target in (
            (str(metadata.get("category", "unknown")), category),
            (str(metadata.get("constraint_count", 0)), constraints),
        ):
            row = target.setdefault(bucket, {"task_count": 0, "trajectory_count": 0, "success_count": 0, "dense_score_sum": 0.0})
            row["task_count"] += 1
            row["trajectory_count"] += 4
            row["success_count"] += sum(bool(item["success"]) for item in group["trajectories"])
            row["dense_score_sum"] += sum(float(item["reward"]) for item in group["trajectories"])

    def finish(rows: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
        return {
            key: {
                "task_count": int(value["task_count"]),
                "trajectory_count": int(value["trajectory_count"]),
                "success_count": int(value["success_count"]),
                "success_rate": value["success_count"] / value["trajectory_count"],
                "mean_dense_task_score": value["dense_score_sum"] / value["trajectory_count"],
            }
            for key, value in sorted(rows.items())
        }

    return {
        "task_count": 500,
        "trajectory_count": 2000,
        "success_count": sum(successes),
        "trajectory_success_rate": sum(successes) / 2000,
        "task_macro_success_rate": sum(task_success) / 500,
        "mean_dense_task_score": sum(scores) / 2000,
        "task_macro_dense_score": sum(task_scores) / 500,
        "nonzero_dense_score_fraction": sum(score > 0 for score in scores) / 2000,
        "generated_action_tokens": sum(int(item["generated_action_tokens"]) for item in trajectories),
        "model_turns": len(turns),
        "environment_steps": sum(int(item["environment_steps"]) for item in trajectories),
        "schema_invalid_turn_count": invalid,
        "schema_invalid_turn_fraction": invalid / len(turns),
        "public_action_error_count": action_errors,
        "public_action_error_fraction": action_errors / max(1, sum(int(item["environment_steps"]) for item in trajectories)),
        "by_category": finish(category),
        "by_constraint_count": finish(constraints),
    }


def validate_eval_report(
    payload: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    report = validate_self_hashed(payload, schema=RUN_REPORT_SCHEMA)
    _require(report.get("complete") is True and report.get("passed") is True, "M5 frozen evaluation is incomplete")
    _require(report.get("formal_evaluation") is True and report.get("training_updates_allowed") is False, "M5 evaluation/training boundary drift")
    _require(report.get("identity") in IDENTITIES, "M5 evaluation report identity drift")
    summary = report.get("metrics")
    _require(isinstance(summary, Mapping) and summary.get("task_count") == 500 and summary.get("trajectory_count") == 2000, "M5 evaluation report matrix drift")
    _require(report.get("optimizer_state_loaded") is False, "M5 evaluation report loaded optimizer state")
    _require(_sha(report.get("protocol_sha256")) and _sha(report.get("eval_plan_sha256")) and _sha(report.get("authorization_sha256")), "M5 evaluation report lineage drift")
    groups = report.get("group_content_sha256")
    _require(isinstance(groups, Mapping) and len(groups) == 500, "M5 evaluation report group inventory drift")
    _require(tuple(groups) == tuple(f"e{index:04d}" for index in range(500)), "M5 evaluation report group order drift")
    _require(all(_sha(value) for value in groups.values()), "M5 evaluation report group hash drift")
    gates = report.get("gates")
    _require(isinstance(gates, Mapping) and gates and all(value is True for value in gates.values()), "M5 evaluation report gates failed")
    _require(report.get("unmet_gates") == [], "M5 evaluation report has unmet gates")
    if root is not None:
        resolved = Path(root).expanduser().resolve()
        _require(resolved == evaluation_run_root(report["identity"]), "M5 evaluation report root drift")
        invocation = resolved / "invocation.json"
        _require(invocation.is_file() and sha256_file(invocation) == report.get("invocation_file_sha256"), "M5 evaluation invocation file drift")
        for group_id, expected_sha in groups.items():
            group_path = resolved / "groups" / f"{group_id}.json"
            _require(group_path.is_file(), f"M5 evaluation group file missing: {group_id}")
            group = _json(group_path)
            _require(group.get("content_sha256") == expected_sha and self_hash(group) == expected_sha, f"M5 evaluation group content drift: {group_id}")
    return report
