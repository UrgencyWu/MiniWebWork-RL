#!/usr/bin/env python3
"""Compare K8 single-task and K4-by-four-task gradients on the same trajectories."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.credit import standardized_advantages  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import (  # noqa: E402
    REFERENCE_ADAPTER_NAME,
    _chunks,
    _forward_logprobs,
    _move_batch,
    _optimizer_examples,
    prepare_group_training_examples,
    strict_grpo_kl_loss,
    validate_committed_group,
)
from miniwebwork.webshop_rl.verifier_td import BASELINE_METHOD, strict_terminal_reward  # noqa: E402


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
        _require(len(active) == 1, "M6 Phase2 batch probe requires one active adapter")
        return str(active[0])
    value = getattr(model, "active_adapter", None)
    value = value() if callable(value) else value
    _require(isinstance(value, str) and value, "M6 Phase2 batch probe active adapter drift")
    return value


def _gradient_cosine(left: Any, right: Any, torch: Any, *, chunk_size: int = 1_000_000) -> float:
    _require(left.shape == right.shape and left.ndim == 1, "M6 Phase2 batch gradient shape drift")
    numerator = left_square = right_square = 0.0
    for start in range(0, int(left.numel()), chunk_size):
        left_chunk = left[start : start + chunk_size].double()
        right_chunk = right[start : start + chunk_size].double()
        numerator += float(torch.dot(left_chunk, right_chunk))
        left_square += float(torch.dot(left_chunk, left_chunk))
        right_square += float(torch.dot(right_chunk, right_chunk))
    _require(left_square > 0.0 and right_square > 0.0, "M6 Phase2 batch probe found a zero gradient")
    value = numerator / math.sqrt(left_square * right_square)
    _require(math.isfinite(value) and -1.0 - 1e-12 <= value <= 1.0 + 1e-12, "M6 Phase2 batch cosine invalid")
    return min(1.0, max(-1.0, value))


@dataclass(frozen=True)
class Phase2TurnExample:
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


def _k4_examples(
    group: Mapping[str, Any],
    trajectory_indices: Sequence[int],
    *,
    group_scale: float,
) -> tuple[tuple[Phase2TurnExample, ...], bool]:
    validated = validate_committed_group(group, require_k=8)
    _require(tuple(trajectory_indices) in {(0, 1, 2, 3), (4, 5, 6, 7)}, "M6 Phase2 K4 half drift")
    trajectories = [validated["trajectories"][index] for index in trajectory_indices]
    rewards = [strict_terminal_reward(item["task_score"]) for item in trajectories]
    advantages = standardized_advantages(rewards)
    examples: list[Phase2TurnExample] = []
    for local_index, (trajectory, advantage) in enumerate(zip(trajectories, advantages)):
        turns = trajectory["turns"]
        for turn in turns:
            completion_tokens = len(turn["generated_token_ids"])
            examples.append(
                Phase2TurnExample(
                    group_id=f"{validated['group_id']}.k4.{trajectory_indices[0]}",
                    trajectory_id=trajectory["trajectory_id"],
                    trajectory_index=local_index,
                    turn_index=turn["turn_index"],
                    turns_in_trajectory=len(turns),
                    prompt_token_ids=tuple(turn["prompt_token_ids"]),
                    generated_token_ids=tuple(turn["generated_token_ids"]),
                    behavior_logprobs=tuple(turn["behavior_logprobs"]),
                    sampling_logprobs=tuple(turn["sampling_logprobs"]),
                    advantage=float(advantage),
                    token_loss_weight=group_scale / (4 * len(turns) * completion_tokens),
                )
            )
    weight = sum(item.token_loss_weight * item.completion_tokens for item in examples)
    _require(math.isclose(weight, group_scale, abs_tol=1e-12), "M6 Phase2 K4 group weight drift")
    return tuple(examples), len(set(rewards)) > 1


def _variation(rows: Sequence[Mapping[str, Any]], vectors: Sequence[Any], torch: Any) -> dict[str, Any]:
    _require(len(rows) == len(vectors) >= 2, "M6 Phase2 batch arm is incomplete")
    pairwise = [
        _gradient_cosine(vectors[left], vectors[right], torch)
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    norms = [float(row["gradient_norm"]) for row in rows]
    mean_norm = statistics.fmean(norms)
    return {
        "gradient_count": len(vectors),
        "mean_pairwise_gradient_cosine": statistics.fmean(pairwise),
        "minimum_pairwise_gradient_cosine": min(pairwise),
        "gradient_angular_dispersion": 1.0 - statistics.fmean(pairwise),
        "gradient_norm_coefficient_of_variation": statistics.pstdev(norms) / mean_norm if mean_norm else 0.0,
        "maximum_clip_fraction": max(float(row["clip_fraction"]) for row in rows),
        "maximum_reference_kl": max(float(row["mean_token_reference_kl"]) for row in rows),
        "all_finite": all(bool(row["finite"]) for row in rows),
        "rows": list(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=4, choices=(1, 2, 4, 8))
    args = parser.parse_args()

    import torch
    from miniwebwork.long_horizon_rl.learner import collate_turn_training_examples, load_trainable_policy_model

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M6 Phase2 batch probe requires one GPU")
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "M6 Phase2 batch report already exists")
    candidates = []
    for path in sorted(args.groups_dir.expanduser().resolve().glob("g*.json")):
        group = validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=8)
        if len({bool(item["success"]) for item in group["trajectories"]}) > 1:
            candidates.append((path, group))
    _require(len(candidates) >= 8, "M6 Phase2 batch probe requires eight mixed SFT K8 groups")
    selected = candidates[:8]
    _require(len({group["task_id"] for _, group in selected}) == 8, "M6 Phase2 batch source tasks are duplicated")
    adapter = args.sft_adapter.expanduser().resolve()
    adapter_sha256 = directory_sha256(adapter)
    _require(all(group["adapter_sha256"] == adapter_sha256 for _, group in selected), "M6 Phase2 batch group/SFT drift")
    _require(all(group["training_updates_allowed"] is False for _, group in selected), "M6 Phase2 batch source is update-authorized")

    torch.manual_seed(20260821)
    torch.cuda.manual_seed_all(20260821)
    model, tokenizer = load_trainable_policy_model(base_model=args.base_model.expanduser().resolve(), adapter_path=adapter)
    current = _active_adapter_name(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(parameters, "M6 Phase2 batch probe found no trainable parameters")
    model.load_adapter(str(adapter), adapter_name=REFERENCE_ADAPTER_NAME, is_trainable=False)
    for name, parameter in model.named_parameters():
        if REFERENCE_ADAPTER_NAME in name:
            parameter.requires_grad_(False)
    dropout_modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    _require(dropout_modules, "M6 Phase2 batch probe found no dropout modules")
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
        return vector, {
            "loss": loss_sum,
            "mean_token_reference_kl": kl_sum / token_count,
            "clip_fraction": clip_sum / token_count,
            "gradient_norm": norm,
            "token_count": token_count,
            "finite": all(math.isfinite(value) for value in values),
        }

    k8_vectors = []
    k8_rows = []
    for path, group in selected:
        prepared = prepare_group_training_examples(group, method=BASELINE_METHOD)
        vector, row = gradient(_optimizer_examples(prepared))
        row.update({"task_ids": [group["task_id"]], "source_group": str(path), "trajectory_count": 8})
        k8_vectors.append(vector)
        k8_rows.append(row)

    panels = [
        (selected[:4], (0, 1, 2, 3)),
        (selected[:4], (4, 5, 6, 7)),
        (selected[4:], (0, 1, 2, 3)),
        (selected[4:], (4, 5, 6, 7)),
    ]
    k4_vectors = []
    k4_rows = []
    mixed_k4_groups = 0
    for panel_groups, indices in panels:
        examples = []
        task_ids = []
        panel_mixed = 0
        for _, group in panel_groups:
            group_examples, mixed = _k4_examples(group, indices, group_scale=0.25)
            examples.extend(group_examples)
            task_ids.append(group["task_id"])
            panel_mixed += int(mixed)
        mixed_k4_groups += panel_mixed
        vector, row = gradient(examples)
        row.update({"task_ids": task_ids, "trajectory_count": 16, "mixed_k4_group_count": panel_mixed})
        k4_vectors.append(vector)
        k4_rows.append(row)

    k8 = _variation(k8_rows, k8_vectors, torch)
    k4 = _variation(k4_rows, k4_vectors, torch)
    decision = {
        "same_total_source_trajectory_count": 64,
        "k8_single_task_unique_tasks_per_gradient": 1,
        "k4_four_task_unique_tasks_per_gradient": 4,
        "k4_mixed_group_fraction": mixed_k4_groups / 16,
        "direction_variance_reduced": float(k4["gradient_angular_dispersion"]) < float(k8["gradient_angular_dispersion"]),
        "norm_variance_reduced": float(k4["gradient_norm_coefficient_of_variation"]) < float(k8["gradient_norm_coefficient_of_variation"]),
        "all_finite": bool(k8["all_finite"] and k4["all_finite"]),
        "safety_passed": max(float(k8["maximum_reference_kl"]), float(k4["maximum_reference_kl"])) <= 0.01
        and max(float(k8["maximum_clip_fraction"]), float(k4["maximum_clip_fraction"])) <= 0.10,
    }
    decision["k4_four_task_batch_recommended"] = all(
        decision[key]
        for key in ("direction_variance_reduced", "norm_variance_reduced", "all_finite", "safety_passed")
    )
    report = {
        "schema_version": "m6_phase2_batch_probe_v1",
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "engineering_precheck_only": True,
        "source_groups_dir": str(args.groups_dir.expanduser().resolve()),
        "source_group_content_sha256": [group["content_sha256"] for _, group in selected],
        "source_task_ids": [group["task_id"] for _, group in selected],
        "sft_adapter_sha256": adapter_sha256,
        "dropout": 0.0,
        "evidence_budget": {
            "k8_single_task": "8 gradients x 1 task x K8 = 64 trajectories",
            "k4_four_task": "4 gradients x 4 tasks x K4 = 64 trajectories",
            "same_trajectory_multiset": True,
        },
        "k8_single_task": k8,
        "k4_four_task": k4,
        "decision": decision,
        "interpretation_guardrail": "This reuses seen-task SFT K8 artifacts and may select mixed source groups; it can validate gradient mechanics, not unseen-task generalization.",
    }
    report["content_sha256"] = _self_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
