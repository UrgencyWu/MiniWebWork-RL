"""M2.2 Analysis: stability, failure transitions, overfitting audit, RL readiness."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def analyze_stability(output_dir: Path, base_metrics: dict = None) -> dict:
    """Analyze multi-seed training stability."""
    print("=== Stability Analysis ===")

    seeds_metrics = []
    for seed_dir in sorted(output_dir.glob("seed_*")):
        metrics_path = seed_dir / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            seeds_metrics.append(m)
            print(f"  Seed {m['seed']}: train_loss={m['train_loss']:.4f}, "
                  f"eval_loss={m['eval_loss']:.4f}, exact={m['exact_match']:.1%}")

    if not seeds_metrics:
        print("  No seed metrics found.")
        return {}

    # Stability stats
    eval_losses = [m["eval_loss"] for m in seeds_metrics]
    exact_matches = [m["exact_match"] for m in seeds_metrics]

    stability = {
        "num_seeds": len(seeds_metrics),
        "eval_loss": {
            "mean": sum(eval_losses) / len(eval_losses),
            "std": (sum((x - sum(eval_losses)/len(eval_losses))**2 for x in eval_losses) / len(eval_losses)) ** 0.5,
            "min": min(eval_losses),
            "max": max(eval_losses),
        },
        "exact_match": {
            "mean": sum(exact_matches) / len(exact_matches),
            "std": (sum((x - sum(exact_matches)/len(exact_matches))**2 for x in exact_matches) / len(exact_matches)) ** 0.5,
            "min": min(exact_matches),
            "max": max(exact_matches),
        },
        "per_seed": seeds_metrics,
    }

    print(f"\n  Eval loss: {stability['eval_loss']['mean']:.4f} +/- {stability['eval_loss']['std']:.4f}")
    print(f"  Exact match: {stability['exact_match']['mean']:.1%} +/- {stability['exact_match']['std']:.1%}")

    # Stability verdict
    cv = stability["eval_loss"]["std"] / max(abs(stability["eval_loss"]["mean"]), 1e-6)
    if cv < 0.05:
        verdict = "STABLE"
    elif cv < 0.15:
        verdict = "MODERATE_VARIANCE"
    else:
        verdict = "HIGH_VARIANCE"
    stability["verdict"] = verdict
    print(f"  Coefficient of variation: {cv:.3f} → {verdict}")

    return stability


def analyze_failure_transitions(base_eval_path: Path, sft_evals: list) -> dict:
    """Compare failure patterns between base and SFT models."""
    print("\n=== Failure Transition Analysis ===")

    base_data = load_json(base_eval_path)
    if not base_data:
        print("  No base evaluation data found.")
        return {}

    base_tasks = {}
    if "frozen_test" in base_data:
        for t in base_data["frozen_test"].get("per_task", []):
            base_tasks[t["task_id"]] = t

    # Load SFT per-task results
    sft_task_results = {}
    for eval_path in sft_evals:
        data = load_json(eval_path)
        if data and "frozen_test" in data:
            seed_name = eval_path.stem.replace("eval_", "")
            sft_task_results[seed_name] = {}
            for t in data["frozen_test"].get("per_task", []):
                sft_task_results[seed_name][t["task_id"]] = t

    if not sft_task_results:
        print("  No SFT evaluation data found.")
        return {}

    # Compute transitions (averaged across seeds)
    base_success = {tid: t.get("success", False) for tid, t in base_tasks.items()}

    transitions = {
        "base_success_sft_success": [],
        "base_success_sft_failure": [],
        "base_failure_sft_success": [],
        "base_failure_sft_failure": [],
    }

    for seed_name, seed_tasks in sft_task_results.items():
        for tid, base_result in base_tasks.items():
            sft_result = seed_tasks.get(tid, {})
            base_ok = base_result.get("success", False)
            sft_ok = sft_result.get("success", False)

            if base_ok and sft_ok:
                transitions["base_success_sft_success"].append(tid)
            elif base_ok and not sft_ok:
                transitions["base_success_sft_failure"].append(tid)
            elif not base_ok and sft_ok:
                transitions["base_failure_sft_success"].append(tid)
            else:
                transitions["base_failure_sft_failure"].append(tid)

    # Count
    n = len(base_tasks)
    print(f"  Tasks: {n}")
    print(f"  Base success → SFT success: {len(transitions['base_success_sft_success'])} "
          f"({len(transitions['base_success_sft_success'])/max(n,1):.0%})")
    print(f"  Base success → SFT failure: {len(transitions['base_success_sft_failure'])} "
          f"({len(transitions['base_success_sft_failure'])/max(n,1):.0%})")
    print(f"  Base failure → SFT success: {len(transitions['base_failure_sft_success'])} "
          f"({len(transitions['base_failure_sft_success'])/max(n,1):.0%})")
    print(f"  Base failure → SFT failure: {len(transitions['base_failure_sft_failure'])} "
          f"({len(transitions['base_failure_sft_failure'])/max(n,1):.0%})")

    return {
        "base_success_count": sum(1 for v in base_success.values() if v),
        "transitions": {k: len(v) for k, v in transitions.items()},
        "base_success_sft_success_tasks": transitions["base_success_sft_success"],
        "base_success_sft_failure_tasks": transitions["base_success_sft_failure"],
        "base_failure_sft_success_tasks": transitions["base_failure_sft_success"],
    }


def analyze_overfitting(output_dir: Path) -> dict:
    """Audit for overfitting, action collapse, and no_solution bias."""
    print("\n=== Overfitting / Collapse Audit ===")

    seeds_metrics = []
    for seed_dir in sorted(output_dir.glob("seed_*")):
        metrics_path = seed_dir / "metrics.json"
        if metrics_path.exists():
            seeds_metrics.append(json.loads(metrics_path.read_text()))

    if not seeds_metrics:
        return {}

    # Train-valid loss gap
    gaps = [m["train_loss"] - m["eval_loss"] for m in seeds_metrics]
    avg_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)

    print(f"  Train-Valid loss gap (avg): {avg_gap:.4f}")
    print(f"  Train-Valid loss gap (max): {max_gap:.4f}")

    # Check for collapse (train loss much lower than eval)
    collapse_threshold = 0.5
    collapse_detected = max_gap > collapse_threshold
    print(f"  Collapse threshold: {collapse_threshold}")
    print(f"  Collapse detected: {collapse_detected}")

    # Action distribution (from frozen test)
    action_dists = []
    for seed_dir in sorted(output_dir.glob("seed_*")):
        frozen_path = None
        for eval_file in output_dir.glob(f"eval_*_{seed_dir.name.replace('seed_', '')}.json"):
            data = json.loads(eval_file.read_text())
            if "frozen_test" in data:
                action_dists.append(data["frozen_test"].get("action_distribution", {}))
                break

    # Check if distribution is healthy (no single action > 70%)
    collapse_actions = []
    for dist in action_dists:
        total = sum(dist.values())
        if total > 0:
            for action, count in dist.items():
                if count / total > 0.8:
                    collapse_actions.append(action)

    print(f"  Action distribution collapse (>80% single action): {collapse_actions}")

    # No-solution bias
    no_solution_rates = []
    for seed_dir in sorted(output_dir.glob("seed_*")):
        for eval_file in output_dir.glob(f"eval_*_{seed_dir.name.replace('seed_', '')}.json"):
            data = json.loads(eval_file.read_text())
            if "frozen_test" in data:
                dist = data["frozen_test"].get("action_distribution", {})
                no_sol = dist.get("finish", 0)
                total = sum(dist.values())
                if total > 0:
                    no_solution_rates.append(no_sol / total)

    avg_no_solution = sum(no_solution_rates) / len(no_solution_rates) if no_solution_rates else 0
    print(f"  Avg 'finish' action rate: {avg_no_solution:.1%}")
    no_solution_bias = avg_no_solution > 0.5
    print(f"  No-solution bias detected: {no_solution_bias}")

    # Base capability regression (compare eval metrics)
    base_exact = 0.0  # Would need base eval data
    sft_exact = [m.get("exact_match", 0) for m in seeds_metrics]
    avg_sft_exact = sum(sft_exact) / len(sft_exact) if sft_exact else 0

    result = {
        "avg_train_valid_gap": avg_gap,
        "max_train_valid_gap": max_gap,
        "collapse_detected": collapse_detected,
        "action_collapse_actions": collapse_actions,
        "no_solution_bias": no_solution_bias,
        "avg_no_solution_rate": avg_no_solution,
        "base_capability_regression": avg_sft_exact < base_exact if base_exact > 0 else "unknown",
    }

    print(f"\n  Verdict: {'OVERFITTING RISK' if collapse_detected else 'NO OVERFITTING'}")
    print(f"  Verdict: {'ACTION COLLAPSE' if collapse_actions else 'NO COLLAPSE'}")
    print(f"  Verdict: {'NO_SOLUTION BIAS' if no_solution_bias else 'NO BIAS'}")

    return result


def rl_readiness(stability: dict, overfitting: dict, frozen_metrics: list) -> str:
    """Determine RL readiness based on all analysis."""
    print("\n=== RL Readiness Assessment ===")

    criteria = {
        "stable_training": stability.get("verdict") in ("STABLE", "MODERATE_VARIANCE"),
        "no_overfitting": not overfitting.get("collapse_detected", True),
        "no_action_collapse": len(overfitting.get("action_collapse_actions", [])) == 0,
        "no_no_solution_bias": not overfitting.get("no_solution_bias", True),
        "improves_over_base": False,
        "min_success_rate": False,
    }

    # Check if SFT improves over base
    if frozen_metrics:
        best_success = max(m.get("success_rate", 0) for m in frozen_metrics)
        base_success = 5 / 15  # 33.3%
        criteria["improves_over_base"] = best_success > base_success
        criteria["min_success_rate"] = best_success >= 0.3  # At least 30%

    passed = sum(1 for v in criteria.values() if v)
    total = len(criteria)

    print(f"  Criteria passed: {passed}/{total}")
    for k, v in criteria.items():
        status = "PASS" if v else "FAIL"
        print(f"    {k}: {status}")

    if passed == total:
        verdict = "READY_FOR_AGENTIC_RL"
    elif passed >= total - 1:
        verdict = "READY_FOR_DATA_EXPANSION"
    else:
        verdict = "M2_2_REQUIRES_REVISION"

    print(f"\n  RL Readiness: {verdict}")
    return verdict


def main():
    parser = argparse.ArgumentParser(description="M2.2 Analysis")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "m2_2"))
    parser.add_argument("--base-eval", default=str(PROJECT_ROOT / "artifacts" / "m2_0" / "m2_0_base_agent_metrics.json"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    base_eval_path = Path(args.base_eval)

    # Run all analyses
    stability = analyze_stability(output_dir)
    sft_eval_paths = list(output_dir.glob("eval_*.json"))
    base_data = load_json(base_eval_path)
    base_eval_path_for_transitions = output_dir / "base_eval.json"

    # If we have base eval data saved locally, use it
    if base_eval_path_for_transitions.exists():
        transitions = analyze_failure_transitions(base_eval_path_for_transitions, sft_eval_paths)
    else:
        # Create a synthetic base eval from the known metrics
        base_eval_data = {
            "frozen_test": {
                "per_task": [
                    {"task_id": f"TASK-{i:03d}", "success": i < 5}
                    for i in range(1, 16)
                ]
            }
        }
        transitions = analyze_failure_transitions(
            output_dir / "_base_synthetic.json", sft_eval_paths
        )

    overfitting = analyze_overfitting(output_dir)

    # Load frozen test results for RL readiness
    frozen_results = []
    for eval_path in sft_eval_paths:
        data = load_json(eval_path)
        if data and "frozen_test" in data:
            frozen_results.append(data["frozen_test"])

    verdict = rl_readiness(stability, overfitting, frozen_results)

    # Save full analysis
    analysis = {
        "stability": stability,
        "failure_transitions": transitions,
        "overfitting": overfitting,
        "rl_readiness": verdict,
    }
    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False, default=str))
    print(f"\nAnalysis saved: {analysis_path}")
    print(f"\nFinal verdict: {verdict}")

    return 0 if verdict == "READY_FOR_AGENTIC_RL" else 1


if __name__ == "__main__":
    sys.exit(main())
