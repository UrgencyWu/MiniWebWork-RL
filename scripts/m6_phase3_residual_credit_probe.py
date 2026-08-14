#!/usr/bin/env python3
"""Probe strict-first residual failure credit on the frozen Phase2 batch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import m6_phase2_reward_probe as phase2  # noqa: E402
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

RESIDUAL_ADVANTAGE_SCALE = 0.2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def strict_first_residual_advantages(
    strict_rewards: Sequence[float],
    failure_qualities: Sequence[float],
    *,
    residual_scale: float = RESIDUAL_ADVANTAGE_SCALE,
) -> dict[str, tuple[float, ...]]:
    """Add bounded, failure-only credit after binary group standardization."""
    _require(len(strict_rewards) == len(failure_qualities) == 4, "Phase3 residual credit requires K4")
    _require(0.0 < residual_scale <= 0.2, "Phase3 residual scale drift")
    strict = tuple(float(value) for value in strict_rewards)
    quality = tuple(float(value) for value in failure_qualities)
    _require(all(value in (0.0, 1.0) for value in strict), "Phase3 strict reward is not binary")
    strict_indices = [index for index, value in enumerate(strict) if value == 1.0]
    failure_indices = [index for index, value in enumerate(strict) if value == 0.0]
    _require(strict_indices and failure_indices, "Phase3 residual credit requires a mixed group")
    _require(all(-1.0 <= quality[index] <= 0.0 for index in failure_indices), "Phase3 failure quality left [-1,0]")

    binary = tuple(float(value) for value in standardized_advantages(strict))
    failure_mean = sum(quality[index] for index in failure_indices) / len(failure_indices)
    centered = [0.0] * len(strict)
    for index in failure_indices:
        centered[index] = quality[index] - failure_mean
    maximum = max((abs(centered[index]) for index in failure_indices), default=0.0)
    residual = tuple(
        0.0
        if index in strict_indices or maximum == 0.0
        else residual_scale * centered[index] / maximum
        for index in range(len(strict))
    )
    candidate = tuple(binary[index] + residual[index] for index in range(len(strict)))

    _require(all(residual[index] == 0.0 for index in strict_indices), "Phase3 changed strict credit")
    _require(abs(sum(residual[index] for index in failure_indices)) <= 1e-12, "Phase3 failure residual is not zero mean")
    _require(max(abs(value) for value in residual) <= residual_scale + 1e-12, "Phase3 residual exceeded bound")
    _require(
        all(candidate[index] == binary[index] for index in strict_indices),
        "Phase3 strict standardized advantage changed",
    )
    _require(
        min(candidate[index] for index in strict_indices) > max(candidate[index] for index in failure_indices),
        "Phase3 strict advantage did not dominate failure",
    )
    _require(
        all(candidate[index] > 0.0 for index in strict_indices)
        and all(candidate[index] < 0.0 for index in failure_indices),
        "Phase3 strict/failure advantage sign flipped",
    )
    for left in failure_indices:
        for right in failure_indices:
            if quality[left] > quality[right]:
                _require(residual[left] > residual[right], "Phase3 failure quality order was not preserved")
    return {
        "binary": binary,
        "residual": residual,
        "candidate": candidate,
    }


def panel_examples(
    groups: Sequence[Mapping[str, Any]],
    *,
    use_residual: bool,
) -> tuple[tuple[phase2.Phase2RewardExample, ...], dict[str, Any]]:
    _require(len(groups) == 4, "Phase3 panel must contain four K4 task groups")
    examples: list[phase2.Phase2RewardExample] = []
    group_rows = []
    for group in groups:
        validated = validate_committed_group(group, require_k=4)
        rewards = [phase2.trajectory_rewards(item) for item in validated["trajectories"]]
        credit = strict_first_residual_advantages(
            [item["strict_reward"] for item in rewards],
            [item["failure_quality"] for item in rewards],
        )
        advantages = credit["candidate"] if use_residual else credit["binary"]
        group_rows.append({
            "group_id": validated["group_id"],
            "task_id": validated["task_id"],
            "strict_count": sum(item["strict_reward"] == 1.0 for item in rewards),
            "trajectory_quality": rewards,
            "binary_advantages": list(credit["binary"]),
            "failure_quality_residual": list(credit["residual"]),
            "selected_advantages": list(advantages),
        })
        for trajectory_index, (trajectory, advantage) in enumerate(zip(validated["trajectories"], advantages)):
            for turn in trajectory["turns"]:
                completion_tokens = len(turn["generated_token_ids"])
                examples.append(
                    phase2.Phase2RewardExample(
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
    _require(math.isclose(total_weight, 1.0, abs_tol=1e-12), "Phase3 panel weight drift")
    return tuple(examples), {"groups": group_rows, "hierarchical_weight_sum": total_weight}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--p2-report", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=4, choices=(1, 2, 4, 8))
    args = parser.parse_args()

    import torch
    from miniwebwork.long_horizon_rl.learner import collate_turn_training_examples, load_trainable_policy_model

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Phase3 residual probe requires one GPU")
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase3 residual probe report already exists")

    p2_report = json.loads(args.p2_report.expanduser().resolve().read_text(encoding="utf-8"))
    _require(p2_report.get("content_sha256") == phase2._self_hash(p2_report), "Phase3 P2 report self-hash drift")
    _require(p2_report.get("training_performed") is False and p2_report.get("optimizer_steps") == 0, "Phase3 P2 source trained")
    _require(p2_report.get("decision", {}).get("p2_same_batch_reward_probe_passed") is False, "Phase3 requires the frozen P2 negative result")
    _require(
        all(panel["gradient_cosine"] >= 0.98 for panel in p2_report["panels"])
        and all(panel["secondary_to_primary_gradient_norm_ratio"] < 0.10 for panel in p2_report["panels"]),
        "Phase3 P2 source was not rejected for gradient redundancy",
    )

    source_hashes = list(p2_report["source_group_content_sha256"])
    source_tasks = list(p2_report["source_task_ids"])
    _require(len(source_hashes) == len(source_tasks) == 8, "Phase3 P2 source cardinality drift")
    candidates: dict[str, Mapping[str, Any]] = {}
    for path in sorted(args.groups_dir.expanduser().resolve().glob("g*.json")):
        group = validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=4)
        candidates[group["content_sha256"]] = group
    _require(all(value in candidates for value in source_hashes), "Phase3 source group hash missing")
    selected = [candidates[value] for value in source_hashes]
    _require([group["task_id"] for group in selected] == source_tasks, "Phase3 source task order drift")

    adapter = args.sft_adapter.expanduser().resolve()
    adapter_sha256 = directory_sha256(adapter)
    _require(adapter_sha256 == p2_report["sft_adapter_sha256"], "Phase3 SFT/P2 adapter drift")
    _require(all(group["adapter_sha256"] == adapter_sha256 for group in selected), "Phase3 source group/SFT drift")
    _require(all(group["training_updates_allowed"] is False for group in selected), "Phase3 source permits updates")

    torch.manual_seed(20260823)
    torch.cuda.manual_seed_all(20260823)
    model, tokenizer = load_trainable_policy_model(base_model=args.base_model.expanduser().resolve(), adapter_path=adapter)
    current = phase2._active_adapter_name(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(parameters, "Phase3 residual probe found no trainable parameters")
    model.load_adapter(str(adapter), adapter_name=REFERENCE_ADAPTER_NAME, is_trainable=False)
    for name, parameter in model.named_parameters():
        if REFERENCE_ADAPTER_NAME in name:
            parameter.requires_grad_(False)
    dropout_modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    _require(dropout_modules, "Phase3 residual probe found no dropout modules")
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
        _require(all(math.isfinite(value) for value in values), "Phase3 residual gradient is non-finite")
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
        groups = selected[panel_index * 4 : (panel_index + 1) * 4]
        binary_examples, binary_credit = panel_examples(groups, use_residual=False)
        candidate_examples, candidate_credit = panel_examples(groups, use_residual=True)
        binary_vector, binary_metrics = gradient(binary_examples)
        candidate_vector, candidate_metrics = gradient(candidate_examples)
        secondary = candidate_vector - binary_vector
        secondary_norm = float(torch.linalg.vector_norm(secondary))
        ratio = secondary_norm / float(binary_metrics["gradient_norm"])
        cosine = phase2._gradient_cosine(binary_vector, candidate_vector, torch)
        panel_reports.append({
            "panel_index": panel_index,
            "source_task_ids": [group["task_id"] for group in groups],
            "binary": binary_metrics,
            "strict_first_residual_quality": candidate_metrics,
            "binary_credit": binary_credit,
            "strict_first_residual_credit": candidate_credit,
            "gradient_cosine": cosine,
            "secondary_gradient_norm": secondary_norm,
            "secondary_to_primary_gradient_norm_ratio": ratio,
            "gradient_band_passed": 0.90 <= cosine < 0.98,
            "secondary_norm_band_passed": 0.10 <= ratio <= 0.25,
        })

    group_rows = [
        group
        for panel in panel_reports
        for group in panel["strict_first_residual_credit"]["groups"]
    ]
    decision = {
        "strict_advantages_exactly_unchanged": all(
            all(
                residual == 0.0 and binary == selected_advantage
                for strict_reward, residual, binary, selected_advantage in zip(
                    [item["strict_reward"] for item in group["trajectory_quality"]],
                    group["failure_quality_residual"],
                    group["binary_advantages"],
                    group["selected_advantages"],
                )
                if strict_reward == 1.0
            )
            for group in group_rows
        ),
        "failure_residuals_zero_mean": all(abs(sum(group["failure_quality_residual"])) <= 1e-12 for group in group_rows),
        "residuals_bounded": all(
            max(abs(value) for value in group["failure_quality_residual"]) <= RESIDUAL_ADVANTAGE_SCALE + 1e-12
            for group in group_rows
        ),
        "strict_failure_sign_and_order_preserved": all(
            min(
                advantage
                for item, advantage in zip(group["trajectory_quality"], group["selected_advantages"])
                if item["strict_reward"] == 1.0
            )
            > max(
                advantage
                for item, advantage in zip(group["trajectory_quality"], group["selected_advantages"])
                if item["strict_reward"] == 0.0
            )
            and all(
                advantage > 0.0 if item["strict_reward"] == 1.0 else advantage < 0.0
                for item, advantage in zip(group["trajectory_quality"], group["selected_advantages"])
            )
            for group in group_rows
        ),
        "all_finite": all(
            panel["binary"]["finite"] and panel["strict_first_residual_quality"]["finite"]
            for panel in panel_reports
        ),
        "all_gradient_cosines_in_target_band": all(panel["gradient_band_passed"] for panel in panel_reports),
        "all_secondary_norms_in_target_band": all(panel["secondary_norm_band_passed"] for panel in panel_reports),
        "maximum_reference_kl": max(
            max(panel["binary"]["mean_token_reference_kl"], panel["strict_first_residual_quality"]["mean_token_reference_kl"])
            for panel in panel_reports
        ),
        "maximum_clip_fraction": max(
            max(panel["binary"]["clip_fraction"], panel["strict_first_residual_quality"]["clip_fraction"])
            for panel in panel_reports
        ),
    }
    decision["phase3_residual_credit_probe_passed"] = all(
        decision[key]
        for key in (
            "strict_advantages_exactly_unchanged",
            "failure_residuals_zero_mean",
            "residuals_bounded",
            "strict_failure_sign_and_order_preserved",
            "all_finite",
            "all_gradient_cosines_in_target_band",
            "all_secondary_norms_in_target_band",
        )
    ) and decision["maximum_reference_kl"] <= 0.01 and decision["maximum_clip_fraction"] <= 0.10

    report = {
        "schema_version": "m6_phase3_strict_first_residual_credit_probe_v1",
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "formula": {
            "macro_advantage": "standardized_binary_strict_reward",
            "failure_quality": "phase2_equal_item_option_blocks_v2",
            "residual_insertion": "after_per_group_macro_standardization",
            "failure_residual_centering": "failure_only_zero_mean",
            "failure_residual_max_abs": RESIDUAL_ADVANTAGE_SCALE,
            "strict_residual": 0.0,
            "official_dense_task_score_used_as_reward": False,
        },
        "p2_negative_report_content_sha256": p2_report["content_sha256"],
        "source_groups_dir": str(args.groups_dir.expanduser().resolve()),
        "source_group_content_sha256": source_hashes,
        "source_task_ids": source_tasks,
        "sft_adapter_sha256": adapter_sha256,
        "dropout": 0.0,
        "group_size": 4,
        "task_groups_per_gradient": 4,
        "panels": panel_reports,
        "decision": decision,
        "interpretation_guardrail": "This is a same-batch, zero-update mechanics probe; passing does not establish online or unseen-task improvement.",
    }
    report["content_sha256"] = _self_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
