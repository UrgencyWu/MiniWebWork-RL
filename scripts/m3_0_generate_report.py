#!/usr/bin/env python3
"""Produce a reviewable M3.0 update-and-evaluation report from artifacts.

The script never runs a model or changes an artifact.  It consumes a completed
formal update report and a paired frozen-evaluation report produced by
``analyze_probe_ab.py``.  The resulting JSON and Markdown keep both favorable
and unfavorable outcomes, including infrastructure exclusions and termination
failure taxonomy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _outcome(delta: float, ci: list[float], pvalue: float) -> str:
    if ci[0] > 0 and pvalue < 0.05:
        return "updated policy has positive paired evidence on this frozen evaluation"
    if delta <= 0:
        return "no improvement is supported; preserve this as a negative or neutral result"
    return "point estimate is positive but evidence is inconclusive"


def build_report(
    update: dict[str, Any],
    paired: dict[str, Any],
    *,
    update_path: Path,
    paired_path: Path,
) -> dict[str, Any]:
    if not update.get("complete") or not update.get("passed"):
        raise ValueError("formal update report must have complete=true and passed=true")
    if not update.get("formal_update"):
        raise ValueError("report is not a formal update artifact")
    if paired.get("comparable_pairs", 0) <= 0:
        raise ValueError("paired evaluation contains no comparable pairs")

    delta = float(paired["paired_success_rate_delta_b_minus_a"])
    ci = [float(value) for value in paired["task_bootstrap_95ci_delta_b_minus_a"]]
    if len(ci) != 2:
        raise ValueError("paired evaluation must contain a two-sided bootstrap interval")
    pvalue = float(paired["exact_mcnemar_pvalue"])
    policy_a = paired["policy_a_metrics"]
    policy_b = paired["policy_b_metrics"]
    return {
        "schema_version": "m3_0_delivery_report_v1",
        "formal_update": {
            "report_path": str(update_path),
            "code_git_sha": update.get("code_git_sha"),
            "source_artifact": update.get("source_artifact"),
            "source_artifact_sha256": update.get("source_artifact_sha256"),
            "source_adapter_sha256": update.get("source_adapter_sha256"),
            "checkpoint_path": update.get("checkpoint_path"),
            "checkpoint_adapter_sha256": update.get("checkpoint_adapter_sha256"),
            "sampling_distribution": update.get("sampling_distribution"),
            "selected_group": update.get("selected_group"),
            "pre_update_replay_audit": update.get("pre_update_replay_audit"),
            "training_runtime": update.get("training_runtime"),
            "optimizer": update.get("optimizer"),
            "reload_forward": update.get("reload_forward"),
        },
        "paired_frozen_evaluation": {
            "report_path": str(paired_path),
            "experiment_identity": paired.get("experiment_identity"),
            "baseline_policy": paired.get("policy_a"),
            "updated_policy": paired.get("policy_b"),
            "total_pairs": paired.get("total_pairs"),
            "comparable_pairs": paired.get("comparable_pairs"),
            "baseline": policy_a,
            "updated": policy_b,
            "paired_table": paired.get("paired_table"),
            "success_delta_b_minus_a": delta,
            "task_bootstrap_95ci_delta_b_minus_a": ci,
            "exact_mcnemar_pvalue": pvalue,
            "termination_reason_baseline": paired.get("termination_reason_a"),
            "termination_reason_updated": paired.get("termination_reason_b"),
            "conclusion": _outcome(delta, ci, pvalue),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    update = report["formal_update"]
    evaluation = report["paired_frozen_evaluation"]
    group = update.get("selected_group") or {}
    optimizer = update.get("optimizer") or {}
    baseline = evaluation["baseline"]
    updated = evaluation["updated"]
    ci = evaluation["task_bootstrap_95ci_delta_b_minus_a"]
    return f"""# MiniWebWork-RL M3.0 Delivery Report

## Provenance

| Item | Value |
|---|---|
| Training code Git SHA | `{update.get('code_git_sha')}` |
| Source rollout artifact | `{update.get('source_artifact')}` |
| Source adapter SHA-256 | `{update.get('source_adapter_sha256')}` |
| Updated adapter SHA-256 | `{update.get('checkpoint_adapter_sha256')}` |
| Sampling distribution | `{update.get('sampling_distribution')}` |
| Strict group | `{group.get('task_id')}`; valid={group.get('valid_for_grpo_update')} |

## Training correctness

- Valid trajectories: {group.get('valid_trajectories')}/{group.get('total_trajectories')}; infrastructure errors: {group.get('infrastructure_errors')}.
- Raw/sampling max difference: `{group.get('max_raw_sampling_logprob_abs_diff')}` (tolerance `{group.get('strict_logprob_match_tolerance')}`).
- Pre-update old/current max difference: `{(update.get('pre_update_replay_audit') or {}).get('max_abs_difference')}`.
- LoRA tensors with non-zero gradient: `{optimizer.get('nonzero_gradient_parameter_tensors')}` / `{optimizer.get('trainable_parameter_tensors')}`; changed tensors: `{optimizer.get('changed_parameter_tensors')}`.
- Maximum adapter parameter delta: `{optimizer.get('max_parameter_abs_delta')}`.

## Frozen paired evaluation

| Metric | M2.2R baseline | Updated policy |
|---|---:|---:|
| Comparable success | {_percent(evaluation['baseline'].get('overall_success_rate'))} | {_percent(evaluation['updated'].get('overall_success_rate'))} |
| Infrastructure errors | {evaluation['baseline'].get('infrastructure_errors')} | {evaluation['updated'].get('infrastructure_errors')} |
| Feasible success | {_percent(evaluation['baseline'].get('feasible_success_rate'))} | {_percent(evaluation['updated'].get('feasible_success_rate'))} |
| False no-solution count | {evaluation['baseline'].get('false_no_solution_count')} | {evaluation['updated'].get('false_no_solution_count')} |

- Paired success delta (updated − baseline): {_percent(evaluation['success_delta_b_minus_a'])}.
- Task-bootstrap 95% CI: [{_percent(ci[0])}, {_percent(ci[1])}].
- Exact McNemar p-value: `{evaluation['exact_mcnemar_pvalue']}`.
- Conclusion: **{evaluation['conclusion']}**.

## Failure and infrastructure taxonomy

Baseline termination reasons: `{evaluation.get('termination_reason_baseline')}`

Updated-policy termination reasons: `{evaluation.get('termination_reason_updated')}`

Infrastructure-invalid trajectories are excluded from paired success denominators
and are reported above; they never enter training rewards or gradients.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-update", required=True, type=Path)
    parser.add_argument("--paired-evaluation", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()

    update_path = args.formal_update.expanduser().resolve()
    paired_path = args.paired_evaluation.expanduser().resolve()
    report = build_report(
        _read_json(update_path),
        _read_json(paired_path),
        update_path=update_path,
        paired_path=paired_path,
    )
    _write_json(args.output_json.expanduser().resolve(), report)
    output_markdown = args.output_markdown.expanduser().resolve()
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json.expanduser().resolve()),
        "output_markdown": str(output_markdown),
        "conclusion": report["paired_frozen_evaluation"]["conclusion"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
