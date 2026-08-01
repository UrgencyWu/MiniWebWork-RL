#!/usr/bin/env python3
"""Render the audited M4 v2 final report as an evidence-only Markdown summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ALGORITHMS = ("sft", "rsft", "rloo", "grpo", "gspo")
SEEDS = (20260801, 20260802, 20260803)
LENGTH_BINS = ("0-4", "5-9", "10-14", "15-20")


def _number(value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric value, got {value!r}")
    return float(value)


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _rate(value: Any) -> tuple[int, int, float | None]:
    if not isinstance(value, dict):
        raise ValueError("rate must be a mapping")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    rendered = value.get("value")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        raise ValueError("rate numerator and denominator must be integers")
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("rate numerator/denominator is invalid")
    if denominator == 0:
        if rendered is not None:
            raise ValueError("zero-denominator rate must have null value")
        return numerator, denominator, None
    if not isinstance(rendered, (int, float)) or abs(float(rendered) - numerator / denominator) > 1e-12:
        raise ValueError("rate value does not match its numerator and denominator")
    return numerator, denominator, float(rendered)


def _cost(summary: dict[str, Any], key: str) -> float:
    cost = summary.get("cost")
    if not isinstance(cost, dict):
        raise ValueError("missing cost mapping")
    return _number(cost.get(key))


def _require_summary(summary: Any, *, algorithm: str, seed: int) -> dict[str, Any]:
    if not isinstance(summary, dict) or summary.get("complete") is not True:
        raise ValueError(f"incomplete summary for {algorithm}/{seed}")
    required = (
        "primary_task_macro_success",
        "raw_attempt_success_rate",
        "primary_task_cluster_bootstrap_95ci",
        "failure_taxonomy",
        "cost",
        "trajectory_cost_task_macro",
        "trajectory_length_strata",
        "json_action_quality",
        "adapter_lineage",
        "frozen_task_roster_sha256",
    )
    if any(key not in summary for key in required):
        raise ValueError(f"missing v2 report field for {algorithm}/{seed}")
    if summary["adapter_lineage"].get("verified") is not True:
        raise ValueError(f"unverified adapter lineage for {algorithm}/{seed}")
    if not isinstance(summary["frozen_task_roster_sha256"], str):
        raise ValueError(f"missing frozen roster hash for {algorithm}/{seed}")
    trajectory = summary["trajectory_cost_task_macro"]
    if not isinstance(trajectory, dict) or set(trajectory) != {"environment_steps", "model_turns", "action_tokens"}:
        raise ValueError(f"invalid trajectory cost summary for {algorithm}/{seed}")
    for metric in trajectory.values():
        if not isinstance(metric, dict) or not isinstance(metric.get("task_cluster_bootstrap_95ci"), list):
            raise ValueError(f"trajectory metric lacks a task-level CI for {algorithm}/{seed}")
    strata = summary["trajectory_length_strata"]
    if not isinstance(strata, dict) or set(strata) != set(LENGTH_BINS):
        raise ValueError(f"trajectory bins are incomplete for {algorithm}/{seed}")
    if sum(int(strata[label].get("valid_rollouts", -1)) for label in LENGTH_BINS) != summary.get("valid_attempts"):
        raise ValueError(f"trajectory bin denominators do not cover valid rollouts for {algorithm}/{seed}")
    quality = summary["json_action_quality"]
    if not isinstance(quality, dict):
        raise ValueError(f"missing JSON action quality for {algorithm}/{seed}")
    for field in ("strict_json_failure_rate", "schema_invalid_rate", "fallback_recovered_rate", "environment_action_failure_rate"):
        _rate(quality.get(field))
    return summary


def _require_report(report: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    if report.get("schema_version") != "m4_final_report_v2" or report.get("complete") is not True:
        raise ValueError("input is not a complete M4 v2 final report")
    audit = report.get("audit")
    if not isinstance(audit, dict) or audit.get("adapter_lineage_verified") is not True or audit.get("frozen_record_roster_verified") is not True:
        raise ValueError("report-level adapter/roster audit is incomplete")
    matrix = report.get("matrix")
    if not isinstance(matrix, dict) or set(matrix) != set(ALGORITHMS):
        raise ValueError("report does not contain exactly the five required algorithms")
    for algorithm in ALGORITHMS:
        by_seed = matrix[algorithm]
        if not isinstance(by_seed, dict) or set(by_seed) != {str(seed) for seed in SEEDS}:
            raise ValueError(f"{algorithm} does not contain exactly the three study seeds")
        for seed in SEEDS:
            _require_summary(by_seed[str(seed)], algorithm=algorithm, seed=seed)
    return matrix


def _failure_counts(summary: dict[str, Any]) -> Counter[str]:
    taxonomy = summary.get("failure_taxonomy")
    primary = taxonomy.get("primary_failure_counts") if isinstance(taxonomy, dict) else None
    if not isinstance(primary, dict):
        raise ValueError("missing primary failure counts")
    return Counter({str(key): int(value) for key, value in primary.items()})


def _aggregate_rate(summaries: list[dict[str, Any]], field: str) -> tuple[int, int, float | None]:
    numerators = denominators = 0
    for summary in summaries:
        numerator, denominator, _ = _rate(summary["json_action_quality"][field])
        numerators += numerator
        denominators += denominator
    return numerators, denominators, numerators / denominators if denominators else None


def _render(report: dict[str, Any]) -> str:
    matrix = _require_report(report)
    audit = report["audit"]
    lines = [
        "# MiniWebWork-RL M4 冻结测试研究摘要",
        "",
        "本摘要仅陈述冻结测试的观测证据，不把相关性表述为算法因果结论。",
        "",
        "## 完整性与谱系门禁",
        "",
        "- 5 种算法 × 3 个随机种子均通过完整性门禁；不完整测试工件不会以零填充。",
        f"- 所有最终测试记录精确匹配同一冻结任务 roster：`{audit['frozen_task_roster_sha256']}`。",
        f"- 所有训练从同一初始 adapter 开始：`{audit['common_initial_adapter_sha256']}`。",
        "- 每个最终 adapter 都已与离线训练或两轮在线更新的 SHA-256 谱系闭合验证。",
        "",
        "## 三种子汇总",
        "",
        "| 算法 | 任务宏平均成功率 | 原始尝试成功率 | 种子标准差 | 总 action tokens | 总模型轮数 | 总环境步数 | 总墙钟秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        summaries = [matrix[algorithm][str(seed)] for seed in SEEDS]
        successes = [_number(summary["primary_task_macro_success"]) for summary in summaries]
        raw_successes = [_number(summary["raw_attempt_success_rate"]) for summary in summaries]
        tokens = sum(_cost(summary, "action_tokens") for summary in summaries)
        turns = sum(_cost(summary, "model_turns") for summary in summaries)
        steps = sum(_cost(summary, "environment_steps") for summary in summaries)
        wall = sum(_cost(summary, "reported_wall_seconds") for summary in summaries)
        std = _number(report["aggregates"][algorithm]["population_std_primary_task_macro_success"])
        lines.append(
            f"| {algorithm.upper()} | {_format_float(mean(successes))} | {_format_float(mean(raw_successes))} | {_format_float(std)} | "
            f"{int(tokens)} | {int(turns)} | {int(steps)} | {_format_float(wall, 1)} |"
        )

    lines.extend(["", "## 每种子的成功率与 95% CI", "", "| 算法 | 种子 | 成功率 | 任务聚类 95% CI |", "|---|---:|---:|---:|"])
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            summary = matrix[algorithm][str(seed)]
            ci = summary["primary_task_cluster_bootstrap_95ci"]
            if not isinstance(ci, list) or len(ci) != 2:
                raise ValueError(f"invalid success CI for {algorithm}/{seed}")
            lines.append(
                f"| {algorithm.upper()} | {seed} | {_format_float(_number(summary['primary_task_macro_success']))} | "
                f"[{_format_float(_number(ci[0]))}, {_format_float(_number(ci[1]))}] |"
            )

    lines.extend(["", "## 变长轨迹与任务级成本", "", "| 算法 | 种子 | 平均环境步数 95% CI | 平均 action tokens 95% CI |", "|---|---:|---:|---:|"])
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            metrics = matrix[algorithm][str(seed)]["trajectory_cost_task_macro"]
            environment = metrics["environment_steps"]
            action_tokens = metrics["action_tokens"]
            env_ci = environment["task_cluster_bootstrap_95ci"]
            token_ci = action_tokens["task_cluster_bootstrap_95ci"]
            lines.append(
                f"| {algorithm.upper()} | {seed} | {_format_float(_number(environment['mean']), 2)} "
                f"[{_format_float(_number(env_ci[0]), 2)}, {_format_float(_number(env_ci[1]), 2)}] | "
                f"{_format_float(_number(action_tokens['mean']), 1)} "
                f"[{_format_float(_number(token_ci[0]), 1)}, {_format_float(_number(token_ci[1]), 1)}] |"
            )

    lines.extend(["", "## 变长轨迹分层（跨三种子、描述性）", "", "| 算法 | 环境步数 bin | 有效 rollout | 成功率 |", "|---|---:|---:|---:|"])
    for algorithm in ALGORITHMS:
        summaries = [matrix[algorithm][str(seed)] for seed in SEEDS]
        for label in LENGTH_BINS:
            numerator = denominator = 0
            for summary in summaries:
                stratum = summary["trajectory_length_strata"][label]
                success_numerator, success_denominator, _ = _rate(stratum["success_rate"])
                if stratum.get("valid_rollouts") != success_denominator:
                    raise ValueError(f"trajectory success denominator drifted for {algorithm}/{label}")
                numerator += success_numerator
                denominator += success_denominator
            value = "n/a" if denominator == 0 else _format_float(numerator / denominator)
            lines.append(f"| {algorithm.upper()} | {label} | {denominator} | {numerator}/{denominator} ({value}) |")

    lines.extend(["", "## JSON 动作质量（跨三种子、有效 rollout）", "", "| 算法 | strict JSON 失败 | schema 无效 | fallback 恢复 | 已执行环境动作失败 |", "|---|---:|---:|---:|---:|"])
    for algorithm in ALGORITHMS:
        summaries = [matrix[algorithm][str(seed)] for seed in SEEDS]
        rendered = []
        for field in ("strict_json_failure_rate", "schema_invalid_rate", "fallback_recovered_rate", "environment_action_failure_rate"):
            numerator, denominator, value = _aggregate_rate(summaries, field)
            suffix = "n/a" if value is None else _format_float(value)
            rendered.append(f"{numerator}/{denominator} ({suffix})")
        lines.append(f"| {algorithm.upper()} | " + " | ".join(rendered) + " |")

    lines.extend(["", "## 失败分类（跨三种子总计）", ""])
    for algorithm in ALGORITHMS:
        counts: Counter[str] = Counter()
        for seed in SEEDS:
            counts.update(_failure_counts(matrix[algorithm][str(seed)]))
        rendered = ", ".join(f"{name}: {count}" for name, count in counts.most_common()) or "无失败"
        lines.append(f"- **{algorithm.upper()}**：{rendered}")

    lines.extend(["", "## 成对任务置换检验（每种子）", ""])
    comparisons = report.get("pairwise_task_clustered_comparisons")
    if not isinstance(comparisons, dict):
        raise ValueError("missing pairwise comparisons")
    for label, by_seed in sorted(comparisons.items()):
        if not isinstance(by_seed, dict) or set(by_seed) != {str(seed) for seed in SEEDS}:
            raise ValueError(f"invalid comparison matrix for {label}")
        values = []
        for seed in SEEDS:
            result = by_seed[str(seed)]
            ci = result.get("paired_task_cluster_bootstrap_95ci") if isinstance(result, dict) else None
            if not isinstance(ci, list) or len(ci) != 2:
                raise ValueError(f"invalid paired CI for {label}/{seed}")
            values.append(
                f"{seed}: Δ={_format_float(_number(result['paired_primary_delta_b_minus_a']))}, "
                f"95% CI=[{_format_float(_number(ci[0]))}, {_format_float(_number(ci[1]))}], "
                f"p={_format_float(_number(result['paired_permutation_pvalue']))}"
            )
        lines.append(f"- **{label}**：" + "； ".join(values))

    lines.extend(
        [
            "",
            "## 面向机制的证据解读",
            "",
            "- **稀疏终态奖励**：以任务宏平均成功率、任务聚类 CI 与置换检验共同判断；不以单次 raw reward 排名。",
            "- **变长轨迹**：环境步数和 action-token 的任务级 CI 与固定长度分层同时呈现。长度是策略输出，因此该分层仅作描述，不作条件化因果解释。",
            "- **JSON 动作失败**：strict JSON、schema、fallback 和环境动作错误都以 `分子/分母` 报告，并与主失败分类交叉解读，避免将格式失败误归因于任务推理能力。",
            "- **外推边界**：这是固定模型、任务环境、训练预算和随机种子下的经验比较，不构成通用算法排序。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    rendered = _render(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"report": str(args.output), "bytes": len(rendered.encode("utf-8"))}, ensure_ascii=False))
    return 0
