#!/usr/bin/env python3
"""Run matched one-update teacher/rehearsal Phase10 corrective-SFT probes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_json  # noqa: E402
from miniwebwork.long_horizon_rl.sft_trainer import (  # noqa: E402
    CompletionOnlyCollator,
    _forward,
    _move_batch,
    completion_only_cross_entropy,
)
from miniwebwork.m6_posttraining_protocol import load_protocol  # noqa: E402
from miniwebwork.webshop_rl.m6_sft_training import (  # noqa: E402
    M6SFTConfig,
    load_recovery,
    load_trainable_lora_model,
    save_recovery,
    tokenize_sft_row,
)
from miniwebwork.webshop_rl.phase10_corrective import (  # noqa: E402
    parameter_displacement,
    parameter_snapshot,
    parameter_tensor_sha256,
    row_loss_coefficients,
    sampled_fixed_action_kl,
    validate_source_weights,
)

SEED = 20260842
FIXED_KL_LIMIT = 0.01
RETENTION_NLL_INCREASE_LIMIT = 0.10
RELATIVE_PARAMETER_DISPLACEMENT_LIMIT = 0.01


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("content_sha256") == _self_hash(value), "Phase10 probe input hash drift")
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _selected_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    tail_labels = labels[:, -logits.shape[1] :]
    shifted_labels = tail_labels[:, 1:]
    shifted_logits = logits[:, :-1, :].float()
    mask = shifted_labels != -100
    gathered = torch.log_softmax(shifted_logits, dim=-1).gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return [float(value) for value in gathered[mask].detach().cpu()]


def _evaluate(
    model: Any,
    examples: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    collator: CompletionOnlyCollator,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    model.eval()
    totals: dict[tuple[str, str, str, str], list[float]] = {}
    logprobs: dict[str, list[float]] = {"old_sft": [], "current_student": []}
    with torch.inference_mode():
        for row, example in zip(rows, examples):
            batch = _move_batch(collator([example]), device)
            output = _forward(model, batch)
            loss = completion_only_cross_entropy(output.logits, batch["labels"])
            key = (str(row["source"]), str(row["task_id"]), str(row["state_id"]), str(row["path_id"]))
            slot = totals.setdefault(key, [0.0, 0.0])
            slot[0] += float(loss.total.cpu())
            slot[1] += loss.token_count
            if row["source"] in logprobs:
                logprobs[str(row["source"])].extend(_selected_logprobs(output.logits, batch["labels"]))
    per_source: dict[str, list[float]] = {}
    nested: dict[str, dict[str, dict[str, list[float]]]] = {}
    for (source, task, state, _path), (total, tokens) in totals.items():
        nested.setdefault(source, {}).setdefault(task, {}).setdefault(state, []).append(total / tokens)
    for source, tasks in nested.items():
        task_values = []
        for states in tasks.values():
            task_values.append(sum(sum(paths) / len(paths) for paths in states.values()) / len(states))
        per_source.setdefault(source, []).append(sum(task_values) / len(task_values))
    return {source: values[0] for source, values in per_source.items()}, logprobs


def _run_arm(
    *,
    arm: str,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    config: M6SFTConfig,
    base_model: Path,
    pi0_adapter: Path,
    output_root: Path,
    input_sha: str,
    source_weights: Mapping[str, float],
) -> dict[str, Any]:
    _set_seed(SEED)
    examples = [tokenize_sft_row(row, tokenizer, config) for row in rows]
    label_tokens = [example.completion_label_tokens for example in examples]
    _require(all(all(label == -100 for label in example.labels[: example.prompt_tokens]) for example in examples),
             "Phase10 prefix labels are not fully masked")
    coefficients = row_loss_coefficients(rows, label_tokens, source_weights)
    schedule = [{"source": row["source"], "task": row["task_id"], "state": row["state_id"], "path": row["path_id"],
                 "sample_id": row["sample_id"], "coefficient": coefficient}
                for row, coefficient in zip(rows, coefficients)]
    schedule_sha = sha256_json(schedule)
    input_identity = {"probe_inputs": input_sha, "arm": arm}
    arm_root = output_root / arm
    _require(not (arm_root / "probe_report.json").exists(), f"Phase10 {arm} probe already completed")
    arm_root.mkdir(parents=True, exist_ok=True)
    model = load_trainable_lora_model(base_model, config, resume_adapter=pi0_adapter)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)
    device = torch.device("cuda:0")
    initial_sha = parameter_tensor_sha256(model)
    initial = parameter_snapshot(model)
    pre_nll, pre_logprobs = _evaluate(model, examples, rows, collator, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    contribution = {source: 0.0 for source in source_weights}
    for row, example, coefficient in zip(rows, examples, coefficients):
        batch = _move_batch(collator([example]), device)
        loss = completion_only_cross_entropy(_forward(model, batch).logits, batch["labels"])
        weighted = loss.total * coefficient
        weighted.backward()
        contribution[str(row["source"])] += float(weighted.detach().cpu())
    gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    _require(bool(torch.isfinite(gradient)) and float(gradient) > 0, "Phase10 gradient is zero or non-finite")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output_sha = parameter_tensor_sha256(model)
    displacement = parameter_displacement(initial, model)
    post_nll, post_logprobs = _evaluate(model, examples, rows, collator, device)
    fixed_kl = {
        source: sampled_fixed_action_kl(post_logprobs[source], pre_logprobs[source])
        for source in ("old_sft", "current_student")
    }
    cumulative = {"source_cursor": len(rows), "task_cursor": len(rows), "optimizer_updates": 1}
    save_recovery(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        output_dir=arm_root,
        completed_updates=1,
        schedule_sha256=schedule_sha,
        input_sha256=input_identity,
        cumulative_metrics=cumulative,
    )
    recovery = load_recovery(arm_root, schedule_sha, input_identity)
    _require(recovery is not None and int(recovery["completed_updates"]) == 1, "Phase10 recovery reload failed")
    checkpoint = torch.load(recovery["optimizer_rng"], map_location="cpu", weights_only=False)
    expected_python = random.random()
    expected_torch = float(torch.rand(1))
    random.setstate(checkpoint["python_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    rng_roundtrip = random.random() == expected_python and float(torch.rand(1)) == expected_torch
    gates = {
        "all_prefix_labels_masked": True,
        "action_label_tokens_positive": sum(label_tokens) > 0,
        "finite_nonzero_gradient": math.isfinite(float(gradient)) and float(gradient) > 0,
        "real_parameter_update": output_sha != initial_sha and displacement["changed_tensor_count"] > 0,
        "new_source_nll_decreased": post_nll["new"] < pre_nll["new"],
        "fixed_state_kl_safe": max(fixed_kl.values()) <= FIXED_KL_LIMIT,
        "retention_nll_safe": all(post_nll[source] - pre_nll[source] <= RETENTION_NLL_INCREASE_LIMIT
                                  for source in ("old_sft", "current_student")),
        "parameter_displacement_safe": 0 < displacement["relative_l2"] <= RELATIVE_PARAMETER_DISPLACEMENT_LIMIT,
        "recovery_roundtrip": rng_roundtrip and recovery["cumulative_metrics"] == cumulative,
    }
    report = {
        "schema_version": "m6_phase10_single_update_arm_report_v1",
        "complete": True,
        "development_only": True,
        "formal_checkpoint_reusable": False,
        "arm": arm,
        "optimizer_updates": 1,
        "learning_rate": config.learning_rate,
        "source_weights": dict(source_weights),
        "loss_hierarchy": "source_task_state_path_action_token",
        "schedule": schedule,
        "schedule_sha256": schedule_sha,
        "input_sha256": input_identity,
        "source_action_rows": {source: sum(row["source"] == source for row in rows) for source in source_weights},
        "source_labeled_tokens": {source: sum(tokens for row, tokens in zip(rows, label_tokens) if row["source"] == source)
                                  for source in source_weights},
        "weighted_loss_contribution": contribution,
        "pre_nll": pre_nll,
        "post_nll": post_nll,
        "fixed_state_k3_kl": fixed_kl,
        "gradient_norm": float(gradient),
        "initial_parameter_sha256": initial_sha,
        "output_parameter_sha256": output_sha,
        "parameter_displacement": displacement,
        "pi0_adapter_sha256": directory_sha256(pi0_adapter),
        "recovery_generation": str(recovery["generation"]),
        "gates": gates,
        "passed": all(gates.values()),
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(arm_root / "probe_report.json", report)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--pi0-adapter", type=Path, required=True)
    parser.add_argument("--student-tokenizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Phase10 single-update probe requires exactly one GPU")
    inputs = _load_hashed(args.inputs.expanduser().resolve())
    _require(inputs.get("schema_version") in {
        "m6_phase10_single_update_probe_inputs_v1", "m6_phase10_single_update_probe_inputs_v2"
    }, "Phase10 probe input schema drift")
    revised_control = inputs.get("control_mode") == "loss_mass_matched_continued_sft"
    source_weights = validate_source_weights(inputs["source_weights"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.student_tokenizer.expanduser().resolve()), trust_remote_code=True, local_files_only=True)
    config = M6SFTConfig.from_protocol(load_protocol()["payload"], seed=SEED)
    _require(math.isclose(config.learning_rate, float(inputs["learning_rate"]), abs_tol=1e-15), "Phase10 probe LR drift")
    output_root = args.output_root.expanduser().resolve()
    _require(not (output_root / "single_update_probe_report.json").exists(), "Phase10 combined probe report exists")
    started = time.monotonic()
    reports = {}
    for arm in ("teacher", "rehearsal"):
        rows = [row for source in ("new", "old_sft", "current_student") for row in inputs["arms"][arm][source]]
        reports[arm] = _run_arm(
            arm=arm,
            rows=rows,
            tokenizer=tokenizer,
            config=config,
            base_model=args.base_model.expanduser().resolve(),
            pi0_adapter=args.pi0_adapter.expanduser().resolve(),
            output_root=output_root,
            input_sha=inputs["content_sha256"],
            source_weights=source_weights,
        )
    cross_arm = {
        "identical_initial_parameters": reports["teacher"]["initial_parameter_sha256"] == reports["rehearsal"]["initial_parameter_sha256"],
        "new_source_task_mass_matched": inputs["source_audit"]["teacher"]["new"]["task_count"]
                                        == inputs["source_audit"]["rehearsal"]["new"]["task_count"],
        "new_source_loss_mass_matched": math.isclose(source_weights["new"], 0.60, abs_tol=1e-12),
        "identical_retention_inputs": inputs["arms"]["teacher"]["old_sft"] == inputs["arms"]["rehearsal"]["old_sft"]
                                      and inputs["arms"]["teacher"]["current_student"] == inputs["arms"]["rehearsal"]["current_student"],
    }
    if not revised_control:
        cross_arm.update({
            "new_source_action_rows_matched": reports["teacher"]["source_action_rows"]["new"] == reports["rehearsal"]["source_action_rows"]["new"],
            "new_source_labeled_tokens_matched": reports["teacher"]["source_labeled_tokens"]["new"] == reports["rehearsal"]["source_labeled_tokens"]["new"],
        })
    combined = {
        "schema_version": "m6_phase10_single_update_probe_report_v1",
        "complete": True,
        "development_only": True,
        "formal_checkpoint_reusable": False,
        "inputs_content_sha256": inputs["content_sha256"],
        "control_mode": inputs.get("control_mode", "exact_complete_path"),
        "attribution_boundary": inputs.get("attribution_boundary", "exact_compute_matched_control"),
        "pure_teacher_action_effect_claim_allowed": inputs.get("pure_teacher_action_effect_claim_allowed", True),
        "thresholds": {
            "fixed_state_k3_kl_max": FIXED_KL_LIMIT,
            "retention_nll_increase_max": RETENTION_NLL_INCREASE_LIMIT,
            "relative_parameter_displacement_max": RELATIVE_PARAMETER_DISPLACEMENT_LIMIT,
        },
        "arms": {arm: {"report": str(output_root / arm / "probe_report.json"), "content_sha256": report["content_sha256"], "passed": report["passed"]}
                 for arm, report in reports.items()},
        "cross_arm_gates": cross_arm,
        "passed": all(report["passed"] for report in reports.values()) and all(cross_arm.values()),
        "elapsed_seconds": time.monotonic() - started,
    }
    combined["content_sha256"] = _self_hash(combined)
    atomic_write_json(output_root / "single_update_probe_report.json", combined)
    print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
