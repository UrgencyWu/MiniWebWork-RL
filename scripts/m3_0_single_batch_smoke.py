#!/usr/bin/env python3
"""M3.0B-1 single-batch multi-turn LoRA optimizer smoke.

The script consumes one strict schema-v3.3 rollout artifact, independently
revalidates its behavior distribution and token evidence, replays every real
browser turn under its stored prompt, performs exactly one LoRA-only optimizer
step, saves a disposable adapter, and reloads it for a finite forward check.

This is an optimizer-contract test, not a performance-improvement experiment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if not SRC_DIR.is_dir():
    raise RuntimeError(f"Invalid source directory: {SRC_DIR}")
sys.path.insert(0, str(SRC_DIR))

from miniwebwork.model_agent.model_backend import extract_generated_token_logprobs
from miniwebwork.rl.batch import build_replay_group
from miniwebwork.rl.streaming import clipped_single_trajectory_loss
from miniwebwork.rollout import RolloutRecord, RolloutStep

DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "m3_0b1_smoke"


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise ValueError(f"Directory contains no files: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(_file_sha256(file).encode("ascii"))
    return digest.hexdigest()


def _disable_allocator_warmup() -> None:
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


def _record_from_dict(value: dict[str, Any]) -> RolloutRecord:
    step_fields = {field.name for field in fields(RolloutStep)}
    record_fields = {field.name for field in fields(RolloutRecord)}
    steps = [
        RolloutStep(**{key: item[key] for key in step_fields if key in item})
        for item in value.get("steps", [])
    ]
    payload = {key: value[key] for key in record_fields if key in value and key != "steps"}
    payload["steps"] = steps
    record = RolloutRecord(**payload)
    record.validate()
    return record


def _select_group(
    artifact: dict[str, Any],
    task_id: str | None,
) -> tuple[dict[str, Any], list[RolloutRecord]]:
    groups = artifact.get("groups")
    records = artifact.get("records")
    if not isinstance(groups, list) or not isinstance(records, list):
        raise ValueError("artifact must contain groups and records lists")

    candidates = [
        group
        for group in groups
        if bool(group.get("valid_for_grpo_update"))
        and (task_id is None or group.get("task_id") == task_id)
    ]
    if not candidates:
        requested = f" for task {task_id}" if task_id else ""
        raise ValueError(f"artifact contains no valid_for_grpo_update group{requested}")
    selected = sorted(candidates, key=lambda group: str(group.get("task_id")))[0]
    if int(selected.get("infrastructure_errors", 0)) != 0:
        raise ValueError("single-batch smoke requires a group with zero infrastructure errors")

    selected_records = [
        _record_from_dict(record)
        for record in records
        if record.get("task_id") == selected.get("task_id")
    ]
    if not selected_records:
        raise ValueError("selected group has no trajectory records")
    return selected, selected_records


def _load_trainable_policy(base_model_path: str, adapter_dir: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    _disable_allocator_warmup()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=True,
        torch_dtype=torch.bfloat16,
    )
    model.enable_adapter_layers()
    model.config.use_cache = False
    model = model.to("cuda:0")
    # Evaluation mode disables dropout while preserving gradients for LoRA.
    model.eval()
    torch.cuda.synchronize()

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("PEFT model contains no trainable parameters")
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError(
            "single-batch smoke permits LoRA parameters only; unexpected trainable "
            f"parameters: {unexpected[:10]}"
        )
    return model, trainable


def _turn_logprobs(model, prompt_ids: tuple[int, ...], completion_ids: tuple[int, ...]) -> torch.Tensor:
    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion token IDs must be non-empty")
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device="cuda:0")
    completion = torch.tensor(completion_ids, dtype=torch.long, device="cuda:0")
    full = torch.cat([prompt, completion]).unsqueeze(0)
    attention = torch.ones_like(full)
    output = model(input_ids=full, attention_mask=attention, use_cache=False)
    return extract_generated_token_logprobs(
        output.logits,
        prompt_length=len(prompt_ids),
        generated_ids=completion,
    )


def _audit_old_current(model, replay_group) -> dict[str, float | int]:
    maximum = 0.0
    token_count = 0
    turn_count = 0
    with torch.inference_mode():
        for trajectory in replay_group.trajectories:
            for turn in trajectory.turns:
                current = _turn_logprobs(
                    model,
                    turn.prompt_token_ids,
                    turn.completion_token_ids,
                )
                old = torch.tensor(
                    turn.old_policy_logprobs,
                    dtype=current.dtype,
                    device=current.device,
                )
                maximum = max(maximum, float((current - old).abs().max().item()))
                token_count += int(current.numel())
                turn_count += 1
                del current, old
    return {
        "max_abs_difference": maximum,
        "token_count": token_count,
        "turn_count": turn_count,
    }


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64, device="cuda:0")
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("trainable gradient contains NaN or Inf")
        total += gradient.double().pow(2).sum()
    return float(total.sqrt().item())


def _train_one_batch(
    model,
    trainable,
    replay_group,
    *,
    learning_rate: float,
    clip_epsilon: float,
    gradient_clip: float,
) -> dict[str, Any]:
    parameters = [parameter for _, parameter in trainable]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    before = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in trainable
    }

    trajectory_count = len(replay_group.trajectories)
    weighted_loss = 0.0
    weighted_clip_fraction = 0.0
    weighted_approximate_kl = 0.0
    weighted_mean_ratio = 0.0
    total_tokens = 0

    for trajectory in replay_group.trajectories:
        trajectory_tokens = trajectory.action_token_count
        if trajectory_tokens <= 0:
            raise ValueError("trajectory contains no action token")
        for turn in trajectory.turns:
            current = _turn_logprobs(
                model,
                turn.prompt_token_ids,
                turn.completion_token_ids,
            )
            old = torch.tensor(
                turn.old_policy_logprobs,
                dtype=current.dtype,
                device=current.device,
            )
            segment = clipped_single_trajectory_loss(
                current,
                old,
                trajectory.advantage,
                clip_epsilon=clip_epsilon,
            )
            trajectory_weight = segment.token_count / trajectory_tokens
            batch_weight = trajectory_weight / trajectory_count
            (segment.loss * batch_weight).backward()

            weighted_loss += float(segment.loss.detach().cpu()) * batch_weight
            weighted_clip_fraction += float(segment.clip_fraction.detach().cpu()) * batch_weight
            weighted_approximate_kl += float(segment.approximate_kl.detach().cpu()) * batch_weight
            weighted_mean_ratio += float(segment.mean_ratio.detach().cpu()) * batch_weight
            total_tokens += segment.token_count
            del current, old, segment

    nonzero_gradient_parameters = sum(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in parameters
    )
    if nonzero_gradient_parameters == 0:
        raise RuntimeError("all LoRA gradients are zero")
    pre_clip_norm = _gradient_norm(parameters)
    if not math.isfinite(pre_clip_norm) or pre_clip_norm <= 0:
        raise FloatingPointError(f"invalid pre-clip gradient norm: {pre_clip_norm}")
    clipped_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    clipped_norm = float(clipped_norm_tensor.detach().cpu())
    post_clip_norm = _gradient_norm(parameters)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    maximum_parameter_delta = 0.0
    changed_parameters = 0
    for name, parameter in trainable:
        current_cpu = parameter.detach().float().cpu()
        delta = float((current_cpu - before[name]).abs().max().item())
        if not math.isfinite(delta):
            raise FloatingPointError(f"non-finite parameter delta for {name}")
        maximum_parameter_delta = max(maximum_parameter_delta, delta)
        changed_parameters += int(delta > 0)
    if changed_parameters == 0 or maximum_parameter_delta <= 0:
        raise RuntimeError("optimizer step did not change any LoRA parameter")

    return {
        "trajectory_count": trajectory_count,
        "action_token_count": total_tokens,
        "mean_policy_loss": weighted_loss,
        "clip_fraction": weighted_clip_fraction,
        "approximate_kl": weighted_approximate_kl,
        "mean_ratio": weighted_mean_ratio,
        "trainable_parameter_tensors": len(parameters),
        "nonzero_gradient_parameter_tensors": nonzero_gradient_parameters,
        "pre_clip_gradient_norm": pre_clip_norm,
        "clip_grad_norm_return": clipped_norm,
        "post_clip_gradient_norm": post_clip_norm,
        "changed_parameter_tensors": changed_parameters,
        "max_parameter_abs_delta": maximum_parameter_delta,
    }


def _reload_forward(
    base_model_path: str,
    adapter_dir: Path,
    prompt_ids: tuple[int, ...],
    completion_ids: tuple[int, ...],
) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    model.eval()
    with torch.inference_mode():
        values = _turn_logprobs(model, prompt_ids, completion_ids)
    if not torch.isfinite(values).all():
        raise FloatingPointError("reloaded checkpoint produced non-finite logprobs")
    result = {
        "token_count": int(values.numel()),
        "logprob_min": float(values.min().cpu()),
        "logprob_max": float(values.max().cpu()),
    }
    del values, model, base
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="M3.0B-1 single-batch LoRA smoke")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--old-current-tolerance", type=float, default=5e-2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("single-batch smoke requires one CUDA device")
    for name, value in (
        ("learning-rate", args.learning_rate),
        ("gradient-clip", args.gradient_clip),
        ("old-current-tolerance", args.old_current_tolerance),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < args.clip_epsilon < 1:
        raise ValueError("clip-epsilon must be in (0, 1)")

    artifact_path = args.artifact.expanduser().resolve()
    adapter_dir = args.adapter.expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not artifact.get("complete"):
        raise ValueError("rollout artifact must have complete=true")
    if artifact.get("schema_version") != "3.3":
        raise ValueError("single-batch smoke requires rollout schema_version=3.3")
    actual_adapter_hash = _directory_sha256(adapter_dir)
    if artifact.get("adapter_sha256") != actual_adapter_hash:
        raise ValueError("artifact adapter hash does not match --adapter")

    selected_group, selected_records = _select_group(artifact, args.task_id)
    replay_group = build_replay_group(
        selected_records,
        logprob_match_tolerance=args.old_current_tolerance,
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / (
            f"{artifact.get('policy', 'policy')}_{replay_group.task_id}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "updated_adapter"
    report_path = output_dir / "single_batch_smoke_report.json"

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": "m3_0b1_single_batch_smoke",
        "complete": False,
        "passed": False,
        "source_artifact": str(artifact_path),
        "source_artifact_sha256": _file_sha256(artifact_path),
        "source_adapter": str(adapter_dir),
        "source_adapter_sha256": actual_adapter_hash,
        "base_model": args.base_model,
        "policy": artifact.get("policy"),
        "task_id": replay_group.task_id,
        "sampling_distribution": {
            "temperature": replay_group.temperature,
            "top_p": replay_group.top_p,
            "top_k": replay_group.top_k,
        },
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "clip_epsilon": args.clip_epsilon,
            "gradient_clip": args.gradient_clip,
            "old_current_tolerance": args.old_current_tolerance,
            "optimizer_updates": 1,
            "kl_beta": 0.0,
        },
        "selected_group": selected_group,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    model = None
    trainable = None
    started = time.time()
    try:
        model, trainable = _load_trainable_policy(args.base_model, adapter_dir)
        audit = _audit_old_current(model, replay_group)
        report["pre_update_replay_audit"] = audit
        if audit["max_abs_difference"] > args.old_current_tolerance:
            raise ValueError(
                "old/current logprob mismatch before update: "
                f"{audit['max_abs_difference']} > {args.old_current_tolerance}"
            )

        report["optimizer"] = _train_one_batch(
            model,
            trainable,
            replay_group,
            learning_rate=args.learning_rate,
            clip_epsilon=args.clip_epsilon,
            gradient_clip=args.gradient_clip,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        report["checkpoint_path"] = str(checkpoint_dir)
        report["checkpoint_adapter_sha256"] = _directory_sha256(checkpoint_dir)

        first_turn = replay_group.trajectories[0].turns[0]
        del model, trainable
        model = trainable = None
        gc.collect()
        torch.cuda.empty_cache()
        report["reload_forward"] = _reload_forward(
            args.base_model,
            checkpoint_dir,
            first_turn.prompt_token_ids,
            first_turn.completion_token_ids,
        )
        report["passed"] = True
        report["complete"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if model is not None:
            del model
        trainable = None
        gc.collect()
        torch.cuda.empty_cache()
        report["elapsed_s"] = round(time.time() - started, 3)
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic_json_write(report_path, report)
        print(f"Report: {report_path}", flush=True)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
