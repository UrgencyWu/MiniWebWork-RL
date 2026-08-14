#!/usr/bin/env python3
"""Compare binary strict and strict-dominant failure-quality gradients."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m6_phase2_buy_readiness import (  # noqa: E402
    EQUAL_BLOCK_FORMULA,
    _readiness_from_evidence,
    _self_hash as readiness_self_hash,
)
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.credit import standardized_advantages  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import (  # noqa: E402
    REFERENCE_ADAPTER_NAME,
    _chunks,
    _forward_logprobs,
    _move_batch,
    strict_grpo_kl_loss,
    validate_committed_group,
)
from miniwebwork.webshop_rl.verifier_td import strict_terminal_reward  # noqa: E402

FAILURE_REWARD_SCALE = 0.1
PREMATURE_PURCHASE_PENALTY = 0.25
HORIZON_PENALTY = 0.1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def _active_adapter_name(model: Any) -> str:
    active = getattr(model, "active_adapters", None)
    active = active() if callable(active) else active
    if isinstance(active, (list, tuple)):
        _require(len(active) == 1, "M6 Phase2 reward probe requires one active adapter")
        return str(active[0])
    value = getattr(model, "active_adapter", None)
    value = value() if callable(value) else value
    _require(isinstance(value, str) and value, "M6 Phase2 reward probe active adapter drift")
    return value


def _gradient_cosine(left: Any, right: Any, torch: Any, *, chunk_size: int = 1_000_000) -> float:
    _require(left.shape == right.shape and left.ndim == 1, "M6 Phase2 reward gradient shape drift")
    numerator = left_square = right_square = 0.0
    for start in range(0, int(left.numel()), chunk_size):
        left_chunk = left[start : start + chunk_size].double()
        right_chunk = right[start : start + chunk_size].double()
        numerator += float(torch.dot(left_chunk, right_chunk))
        left_square += float(torch.dot(left_chunk, left_chunk))
        right_square += float(torch.dot(right_chunk, right_chunk))
    _require(left_square > 0.0 and right_square > 0.0, "M6 Phase2 reward probe found zero gradient")
    value = numerator / math.sqrt(left_square * right_square)
    _require(math.isfinite(value) and -1.0 - 1e-12 <= value <= 1.0 + 1e-12, "M6 Phase2 reward cosine invalid")
    return min(1.0, max(-1.0, value))


def _clip_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def trajectory_rewards(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    strict = strict_terminal_reward(trajectory.get("task_score", trajectory.get("reward")))
    turns = trajectory.get("turns")
    _require(isinstance(turns, list) and turns, "M6 Phase2 reward trajectory has no turns")
    readiness = [
        _readiness_from_evidence(
            turn["verifier_progress_evidence"],
            formula_version=EQUAL_BLOCK_FORMULA,
        )
        for turn in turns[:-1]
    ]
    before_terminal = readiness[-1] if readiness else 0.0
    maximum = max(readiness, default=0.0)
    termination = str(trajectory.get("termination_reason", ""))
    if strict:
        failure_class = "strict_success"
        quality = 0.0
    elif termination == "purchase":
        failure_class = "partial_purchase" if float(trajectory.get("task_score", 0.0)) > 0 else "zero_match_purchase"
        quality = -_clip_unit((1.0 - before_terminal) + PREMATURE_PURCHASE_PENALTY)
    elif termination in {"max_environment_steps", "max_model_turns"}:
        failure_class = "horizon_exhaustion"
        quality = -_clip_unit((1.0 - maximum) + HORIZON_PENALTY)
    else:
        failure_class = "schema_or_action_failure"
        quality = -1.0
    shaped = strict + (1.0 - strict) * FAILURE_REWARD_SCALE * quality
    _require(-1.0 <= quality <= 0.0, "M6 Phase2 failure quality left [-1, 0]")
    _require(shaped == 1.0 if strict else -FAILURE_REWARD_SCALE <= shaped <= 0.0, "M6 Phase2 strict-dominant reward drift")
    return {
        "strict_reward": strict,
        "failure_quality": quality,
        "strict_dominant_reward": shaped,
        "failure_class": failure_class,
        "final_preterminal_readiness": before_terminal,
        "maximum_preterminal_readiness": maximum,
    }


@dataclass(frozen=True)
class Phase2RewardExample:
    group_id: str
    trajectory_id: str
    trajectory_index: int
    turn_index: int
    turns_in_trajectory: int
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    sampling_logprobs: tuple[float, ...]
    advantage: float
    token_loss_weight: float

    @property
    def completion_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def forward_tokens(self) -> int:
        return len(self.prompt_token_ids) + self.completion_tokens


def panel_examples(
    groups: Sequence[Mapping[str, Any]],
    *,
    reward_key: str,
) -> tuple[tuple[Phase2RewardExample, ...], dict[str, Any]]:
    _require(len(groups) == 4, "M6 Phase2 reward panel must contain four K4 task groups")
    examples: list[Phase2RewardExample] = []
    group_rows = []
    for group in groups:
        validated = validate_committed_group(group, require_k=4)
        rewards = [trajectory_rewards(item) for item in validated["trajectories"]]
        selected_rewards = [float(item[reward_key]) for item in rewards]
        advantages = standardized_advantages(selected_rewards)
        strict_indices = [index for index, item in enumerate(rewards) if item["strict_reward"] == 1.0]
        failure_indices = [index for index, item in enumerate(rewards) if item["strict_reward"] == 0.0]
        _require(strict_indices and failure_indices, "M6 Phase2 reward panel requires mixed groups")
        _require(
            min(advantages[index] for index in strict_indices) > max(advantages[index] for index in failure_indices),
            "M6 Phase2 strict advantage did not dominate failure",
        )
        _require(
            all(advantages[index] > 0.0 for index in strict_indices)
            and all(advantages[index] < 0.0 for index in failure_indices),
            "M6 Phase2 strict/failure macro sign flipped",
        )
        group_rows.append({
            "group_id": validated["group_id"],
            "task_id": validated["task_id"],
            "strict_count": len(strict_indices),
            "rewards": selected_rewards,
            "advantages": list(advantages),
            "trajectory_quality": rewards,
        })
        for trajectory_index, (trajectory, advantage) in enumerate(zip(validated["trajectories"], advantages)):
            for turn in trajectory["turns"]:
                completion_tokens = len(turn["generated_token_ids"])
                examples.append(
                    Phase2RewardExample(
                        group_id=validated["group_id"],
                        trajectory_id=trajectory["trajectory_id"],
                        trajectory_index=trajectory_index,
                        turn_index=turn["turn_index"],
                        turns_in_trajectory=len(trajectory["turns"]),
                        prompt_token_ids=tuple(turn["prompt_token_ids"]),
                        generated_token_ids=tuple(turn["generated_token_ids"]),
                        behavior_logprobs=tuple(turn["behavior_logprobs"]),
                        sampling_logprobs=tuple(turn["sampling_logprobs"]),
                        advantage=float(advantage),
                        token_loss_weight=0.25 / (4 * len(trajectory["turns"]) * completion_tokens),
                    )
                )
    total_weight = sum(item.token_loss_weight * item.completion_tokens for item in examples)
    _require(math.isclose(total_weight, 1.0, abs_tol=1e-12), "M6 Phase2 reward panel weight drift")
    return tuple(examples), {"groups": group_rows, "hierarchical_weight_sum": total_weight}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=4, choices=(1, 2, 4, 8))
    args = parser.parse_args()

    import torch
    from miniwebwork.long_horizon_rl.learner import collate_turn_training_examples, load_trainable_policy_model

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M6 Phase2 reward probe requires one GPU")
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "M6 Phase2 reward probe report already exists")
    readiness_report = json.loads(args.readiness_report.expanduser().resolve().read_text(encoding="utf-8"))
    _require(readiness_report.get("content_sha256") == readiness_self_hash(readiness_report), "M6 Phase2 readiness report self-hash drift")
    _require(readiness_report.get("formula_version") == EQUAL_BLOCK_FORMULA, "M6 Phase2 readiness formula drift")
    _require(readiness_report.get("decision", {}).get("process_reward_calibration_passed") is True, "M6 Phase2 readiness gate did not pass")
    _require(readiness_report.get("training_performed") is False, "M6 Phase2 readiness report performed training")

    candidates = []
    for path in sorted(args.groups_dir.expanduser().resolve().glob("g*.json")):
        group = validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=4)
        strict = [strict_terminal_reward(item["task_score"]) for item in group["trajectories"]]
        if len(set(strict)) > 1:
            candidates.append((path, group))
    _require(len(candidates) >= 8, "M6 Phase2 reward probe requires eight mixed full-horizon groups")
    selected = candidates[:8]
    _require(len({group["task_id"] for _, group in selected}) == 8, "M6 Phase2 reward source tasks duplicated")
    adapter = args.sft_adapter.expanduser().resolve()
    adapter_sha256 = directory_sha256(adapter)
    _require(all(group["adapter_sha256"] == adapter_sha256 for _, group in selected), "M6 Phase2 reward group/SFT drift")
    _require(all(group["training_updates_allowed"] is False for _, group in selected), "M6 Phase2 reward source permits updates")

    torch.manual_seed(20260822)
    torch.cuda.manual_seed_all(20260822)
    model, tokenizer = load_trainable_policy_model(base_model=args.base_model.expanduser().resolve(), adapter_path=adapter)
    current = _active_adapter_name(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(parameters, "M6 Phase2 reward probe found no trainable parameters")
    model.load_adapter(str(adapter), adapter_name=REFERENCE_ADAPTER_NAME, is_trainable=False)
    for name, parameter in model.named_parameters():
        if REFERENCE_ADAPTER_NAME in name:
            parameter.requires_grad_(False)
    dropout_modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    _require(dropout_modules, "M6 Phase2 reward probe found no dropout modules")
    device = torch.device("cuda:0")

    def gradient(examples: Sequence[Any]) -> tuple[Any, dict[str, Any]]:
        model.zero_grad(set_to_none=True)
        loss_sum = kl_sum = clip_sum = 0.0
        token_count = 0
        for chunk in _chunks(tuple(examples), args.microbatch_size):
            batch = _move_batch(collate_turn_training_examples(chunk, pad_token_id=tokenizer.pad_token_id), device)
            model.set_adapter(REFERENCE_ADAPTER_NAME)
            model.eval()
            with torch.inference_mode():
                reference = _forward_logprobs(model, batch)
            model.set_adapter(current)
            model.train()
            for module in dropout_modules:
                module.eval()
            replay = _forward_logprobs(model, batch)
            result = strict_grpo_kl_loss(replay, reference, batch, clip_epsilon=0.2, kl_coefficient=0.03)
            result["loss"].backward()
            count = int(result["token_count"])
            loss_sum += float(result["loss"].detach().cpu())
            kl_sum += float(result["mean_token_reference_kl"].detach().cpu()) * count
            clip_sum += float(result["clip_fraction"].detach().cpu()) * count
            token_count += count
        vector = torch.cat([
            parameter.grad.detach().float().cpu().reshape(-1)
            if parameter.grad is not None
            else torch.zeros_like(parameter.detach(), dtype=torch.float32, device="cpu").reshape(-1)
            for parameter in parameters
        ])
        norm = float(torch.linalg.vector_norm(vector))
        values = (loss_sum, kl_sum / token_count, clip_sum / token_count, norm)
        _require(all(math.isfinite(value) for value in values), "M6 Phase2 reward gradient is non-finite")
        return vector, {
            "loss": loss_sum,
            "mean_token_reference_kl": kl_sum / token_count,
            "clip_fraction": clip_sum / token_count,
            "gradient_norm": norm,
            "token_count": token_count,
            "finite": True,
        }

    panel_reports = []
    for panel_index in range(2):
        groups = [group for _, group in selected[panel_index * 4 : (panel_index + 1) * 4]]
        binary_examples, binary_credit = panel_examples(groups, reward_key="strict_reward")
        quality_examples, quality_credit = panel_examples(groups, reward_key="strict_dominant_reward")
        binary_vector, binary_metrics = gradient(binary_examples)
        quality_vector, quality_metrics = gradient(quality_examples)
        secondary = quality_vector - binary_vector
        secondary_norm = float(torch.linalg.vector_norm(secondary))
        ratio = secondary_norm / float(binary_metrics["gradient_norm"])
        cosine = _gradient_cosine(binary_vector, quality_vector, torch)
        panel_reports.append({
            "panel_index": panel_index,
            "source_task_ids": [group["task_id"] for group in groups],
            "binary": binary_metrics,
            "strict_dominant_quality": quality_metrics,
            "binary_credit": binary_credit,
            "strict_dominant_credit": quality_credit,
            "gradient_cosine": cosine,
            "secondary_gradient_norm": secondary_norm,
            "secondary_to_primary_gradient_norm_ratio": ratio,
            "gradient_band_passed": 0.90 <= cosine < 0.98,
            "secondary_norm_band_passed": 0.10 <= ratio <= 0.25,
        })

    decision = {
        "strict_reward_is_one": all(
            item["strict_reward"] == 1.0
            for panel in panel_reports
            for group in panel["strict_dominant_credit"]["groups"]
            for item in group["trajectory_quality"]
            if item["failure_class"] == "strict_success"
        ),
        "all_failure_rewards_nonpositive": all(
            -FAILURE_REWARD_SCALE <= item["strict_dominant_reward"] <= 0.0
            for panel in panel_reports
            for group in panel["strict_dominant_credit"]["groups"]
            for item in group["trajectory_quality"]
            if item["failure_class"] != "strict_success"
        ),
        "strict_failure_macro_sign_preserved": True,
        "all_finite": all(panel["binary"]["finite"] and panel["strict_dominant_quality"]["finite"] for panel in panel_reports),
        "all_gradient_cosines_in_target_band": all(panel["gradient_band_passed"] for panel in panel_reports),
        "all_secondary_norms_in_target_band": all(panel["secondary_norm_band_passed"] for panel in panel_reports),
        "maximum_reference_kl": max(
            max(panel["binary"]["mean_token_reference_kl"], panel["strict_dominant_quality"]["mean_token_reference_kl"])
            for panel in panel_reports
        ),
        "maximum_clip_fraction": max(
            max(panel["binary"]["clip_fraction"], panel["strict_dominant_quality"]["clip_fraction"])
            for panel in panel_reports
        ),
    }
    decision["p2_same_batch_reward_probe_passed"] = all(
        decision[key]
        for key in (
            "strict_reward_is_one",
            "all_failure_rewards_nonpositive",
            "strict_failure_macro_sign_preserved",
            "all_finite",
            "all_gradient_cosines_in_target_band",
            "all_secondary_norms_in_target_band",
        )
    ) and decision["maximum_reference_kl"] <= 0.01 and decision["maximum_clip_fraction"] <= 0.10
    report = {
        "schema_version": "m6_phase2_same_batch_reward_probe_v1",
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "formula": {
            "readiness": EQUAL_BLOCK_FORMULA,
            "failure_reward_scale": FAILURE_REWARD_SCALE,
            "premature_purchase_penalty": PREMATURE_PURCHASE_PENALTY,
            "horizon_penalty": HORIZON_PENALTY,
            "official_dense_task_score_used_as_reward": False,
        },
        "readiness_report_content_sha256": readiness_report["content_sha256"],
        "source_groups_dir": str(args.groups_dir.expanduser().resolve()),
        "source_group_content_sha256": [group["content_sha256"] for _, group in selected],
        "source_task_ids": [group["task_id"] for _, group in selected],
        "sft_adapter_sha256": adapter_sha256,
        "dropout": 0.0,
        "group_size": 4,
        "task_groups_per_gradient": 4,
        "panels": panel_reports,
        "decision": decision,
        "interpretation_guardrail": "This is a same-batch gradient counterfactual without optimizer steps; it cannot establish online or unseen-task performance.",
    }
    report["content_sha256"] = _self_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
