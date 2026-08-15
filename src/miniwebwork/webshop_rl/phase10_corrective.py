"""Phase10 source-balanced corrective SFT primitives.

This module deliberately does not reuse the M6 mini-SFT update loop.  Phase10
uses explicit source -> task -> state -> path -> action-token normalization and
replay-success CE retention; Raw/disable-adapter KL is forbidden.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch


SOURCE_WEIGHTS = {"new": 0.60, "old_sft": 0.25, "current_student": 0.15}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_source_weights(weights: Mapping[str, float]) -> dict[str, float]:
    observed = {str(key): float(value) for key, value in weights.items()}
    _require(set(observed) == set(SOURCE_WEIGHTS), "Phase10 source identities drift")
    _require(all(math.isclose(observed[key], value, abs_tol=1e-12) for key, value in SOURCE_WEIGHTS.items()),
             "Phase10 source weights drift")
    _require(math.isclose(sum(observed.values()), 1.0, abs_tol=1e-12), "Phase10 source weights do not sum to one")
    return observed


def row_loss_coefficients(
    rows: Sequence[Mapping[str, Any]],
    label_tokens: Sequence[int],
    source_weights: Mapping[str, float],
) -> tuple[float, ...]:
    """Return coefficients that implement the frozen hierarchy exactly.

    Multiplying each row's token-sum CE by its coefficient yields
    ``sum_source w_s mean_task mean_state mean_path mean_token CE``.
    """

    weights = validate_source_weights(source_weights)
    _require(len(rows) == len(label_tokens) and bool(rows), "Phase10 rows/token counts drift")
    sources: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for index, (row, tokens) in enumerate(zip(rows, label_tokens)):
        source = str(row.get("source") or "")
        task = str(row.get("task_id") or "")
        state = str(row.get("state_id") or "")
        path = str(row.get("path_id") or row.get("trajectory_id") or "")
        _require(source in weights and task and state and path and int(tokens) > 0, "Phase10 row hierarchy is incomplete")
        sources[source][task][state][path].append(index)
    _require(set(sources) == set(weights), "Phase10 update lacks one or more frozen sources")
    coefficients = [0.0] * len(rows)
    for source, tasks in sources.items():
        for task, states in tasks.items():
            for state, paths in states.items():
                for path, indices in paths.items():
                    path_tokens = sum(int(label_tokens[index]) for index in indices)
                    coefficient = weights[source] / len(tasks) / len(states) / len(paths) / path_tokens
                    for index in indices:
                        coefficients[index] = coefficient
    _require(all(value > 0 and math.isfinite(value) for value in coefficients), "Phase10 row coefficient is invalid")
    return tuple(coefficients)


def parameter_tensor_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        count += 1
    _require(count > 0, "Phase10 model has no trainable tensors")
    return digest.hexdigest()


def parameter_snapshot(model: Any) -> dict[str, torch.Tensor]:
    values = {name: parameter.detach().float().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    _require(values, "Phase10 model has no trainable tensors")
    return values


def parameter_displacement(before: Mapping[str, torch.Tensor], model: Any) -> dict[str, float]:
    squared_before = squared_delta = 0.0
    changed = 0
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    _require(set(current) == set(before), "Phase10 trainable parameter identity drift")
    for name, original in before.items():
        value = current[name].detach().float().cpu()
        delta = value - original
        squared_before += float(original.square().sum())
        squared_delta += float(delta.square().sum())
        changed += int(bool(torch.any(delta != 0)))
    absolute = math.sqrt(squared_delta)
    relative = absolute / max(math.sqrt(squared_before), 1e-30)
    return {"absolute_l2": absolute, "relative_l2": relative, "changed_tensor_count": changed}


def sampled_fixed_action_kl(post_logprobs: Sequence[float], pre_logprobs: Sequence[float]) -> float:
    _require(len(post_logprobs) == len(pre_logprobs) and bool(post_logprobs), "Phase10 fixed-state KL input drift")
    values = []
    for post, pre in zip(post_logprobs, pre_logprobs):
        ratio = float(post) - float(pre)
        values.append(math.exp(ratio) - ratio - 1.0)
    result = sum(values) / len(values)
    _require(math.isfinite(result) and result >= 0, "Phase10 fixed-state KL is invalid")
    return result
