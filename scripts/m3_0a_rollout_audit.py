#!/usr/bin/env python3
"""M3.0A: Rollout Readiness Audit for no_feasible_product tasks.

Phases:
  1. Expert replay TASK-012 — verify environment submission contract
  2. No-solution task audit — slice behavior across all seeds
  3. Random rollout probe — K-trajectory collection with stochastic sampling
  4. Report + route decision — A (GRPO), B (SFT patch), or C (fix env)

Usage:
    python scripts/m3_0a_rollout_audit.py --phase 1          # Expert replay only
    python scripts/m3_0a_rollout_audit.py --phase 2          # No-solution audit
    python scripts/m3_0a_rollout_audit.py --phase 3          # Rollout probe
    python scripts/m3_0a_rollout_audit.py --phase all         # Full audit (default)
    python scripts/m3_0a_rollout_audit.py --route-a-check     # Quick GRPO viability check
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Heavy imports (torch, peft, transformers, miniwebwork modules) are done
# INSIDE each phase function to avoid triggering the PyTorch AllocatorConfig
# crash (c10_cuda.so SIGABRT) in Slurm sbatch environments.

DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"

# ---------------------------------------------------------------------------
# Expert agent adapter — wraps OracleExpertProcurementAgent to implement the
# same interface as QwenBrowserAgent so we can reuse run_model_episode().
# ---------------------------------------------------------------------------


class ExpertEpisodeRunner:
    """Thin wrapper that makes OracleExpertProcurementAgent compatible with
    run_model_episode() (which expects agent.reset() + agent.act() -> ModelActionAttempt).
    """

    def __init__(self, oracle: dict, max_steps: int = 25):
        self._oracle = oracle
        self._max_steps = max_steps
        self._expert: Optional[OracleExpertProcurementAgent] = None
        self.model_turn = 0
        self._task_id = ""
        self._instruction = ""

    def reset(self, task_id: str, instruction: str):
        # Lazy import to avoid PyTorch AllocatorConfig crash at module load
        from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent
        self._task_id = task_id
        self._instruction = instruction
        self.model_turn = 0
        self._expert = OracleExpertProcurementAgent(self._oracle, self._max_steps)
        self._expert.reset()

    def act(self, observation):
        # Lazy import to avoid PyTorch AllocatorConfig crash at module load
        from miniwebwork.model_agent.qwen_agent import ModelActionAttempt
        assert self._expert is not None, "Call reset() before act()"
        self.model_turn += 1
        attempt = ModelActionAttempt(model_turn_index=self.model_turn)

        try:
            action = self._expert.act(observation)
            attempt.action = action
            attempt.schema_valid = True
            attempt.strict_json_success = True
            attempt.raw_output = action.to_dict()
        except Exception as e:
            attempt.errors.append(f"expert_error: {e}")

        return attempt

    def record_feedback(self, attempt, action_result, page_type: str):
        """Expert maintains its own state; no feedback needed."""
        pass


# ---------------------------------------------------------------------------
# Model / backend / agent setup
# ---------------------------------------------------------------------------


def _find_free_gpu() -> str:
    """Find a GPU with at least 20GB free memory."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 2:
                idx = int(parts[0].strip())
                free_mb = int(parts[1].strip())
                if free_mb >= 20000:  # At least 20GB free
                    return str(idx)
    except Exception:
        pass
    return "0"  # fallback


def load_policy(
    base_model_path: str,
    adapter_path: str,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
):
    """Load base model + LoRA adapter and create backend + agent."""
    # Lazy imports to avoid PyTorch AllocatorConfig crash at module load
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from miniwebwork.model_agent.model_backend import ModelConfig, QwenTransformersBackend
    from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
    from miniwebwork.model_agent.output_parser import parse
    import miniwebwork.model_agent.prompt_builder as pb

    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model ({base_model_path})...")
    device = _find_free_gpu()
    print(f"  Using GPU {device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    base_model.eval()

    if adapter_path and adapter_path != base_model_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model, adapter_path, torch_dtype=torch.bfloat16
        )
    else:
        model = base_model

    model.eval()

    config = ModelConfig(
        model_path=base_model_path,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        dtype="bfloat16",
        device="cuda:0",
        enable_thinking=False,
    )
    backend = QwenTransformersBackend(config)
    backend._model = model
    backend._tokenizer = tokenizer
    backend._loaded = True

    pb.HISTORY_WINDOW = 5

    agent = QwenBrowserAgent(backend, pb, parse)
    return backend, agent, tokenizer


def _get_adapter_path(seed: int, output_dir: Path) -> Optional[str]:
    """Get adapter path for a seed, or None if base model."""
    if seed == 0:
        return DEFAULT_BASE_MODEL
    path = output_dir / f"seed_{seed}" / "final_adapter"
    if path.exists():
        return str(path)
    return None


# ---------------------------------------------------------------------------
# Phase 1: Expert Replay TASK-012
# ---------------------------------------------------------------------------


def phase1_expert_replay(
    task_id: str = "TASK-012",
    output_dir: Path = None,
) -> dict:
    """Run OracleExpertProcurementAgent on TASK-012 and audit submission invariants."""
    print("\n" + "=" * 70)
    print(f"PHASE 1: Expert Replay — {task_id}")
    print("=" * 70)

    # Lazy imports to avoid PyTorch AllocatorConfig crash at module load
    from miniwebwork.agent_env.environment import ProcurementBrowserEnv
    from miniwebwork.model_agent.agent_loop import run_model_episode
    from miniwebwork.tasks import get_oracle

    oracle = get_oracle(task_id)
    if oracle is None:
        print(f"ERROR: Oracle not found for {task_id}")
        return {"error": f"Oracle not found for {task_id}"}

    print(f"Task type: {oracle.get('task_type', 'unknown')}")
    print(f"Expected decision: {oracle.get('expected_decision_type', 'unknown')}")
    print(f"Expected product: {oracle.get('expected_product_id', 'none')}")

    runner = ExpertEpisodeRunner(oracle, max_steps=25)
    run_id = f"expert_{task_id}_{uuid.uuid4().hex[:8]}"
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks")

    result = {
        "task_id": task_id,
        "oracle": {
            "task_type": oracle.get("task_type"),
            "expected_decision_type": oracle.get("expected_decision_type"),
            "expected_product_id": oracle.get("expected_product_id"),
            "constraints": oracle.get("constraints", {}),
        },
        "episode_run": None,
        "invariants": {},
        "route": None,
    }

    try:
        with ProcurementBrowserEnv(max_steps=15, run_id=run_id) as env:
            env.set_agent_name("oracle_expert")

            episode_result = run_model_episode(
                task_id, env, runner, max_model_turns=20, max_env_steps=15
            )

            result["episode_run"] = {
                "episode_id": episode_result.get("episode_id", ""),
                "success": episode_result.get("success", False),
                "reward": episode_result.get("reward", 0.0),
                "termination_reason": episode_result.get("termination_reason", ""),
                "failure_reasons": episode_result.get("failure_reasons", []),
                "model_turns": episode_result.get("model_turns", 0),
                "environment_steps": episode_result.get("environment_steps", 0),
                "elapsed_s": episode_result.get("elapsed_s", 0.0),
                "error": episode_result.get("error", ""),
            }

            # Audit submission invariants from trajectory
            if env.trajectory:
                v = env.trajectory.verification
                result["invariants"] = {
                    "submission_id": v.get("submission_id", ""),
                    "decision_type": v.get("decision_type", ""),
                    "verifier_success": v.get("success", False),
                    "failure_reasons": v.get("failure_reasons", []),
                    "error": v.get("details", {}).get("error", ""),
                }

            # Analyze page flow from trajectory steps
            page_flow = []
            for step in (env.trajectory.steps if env.trajectory else []):
                page_flow.append({
                    "step": step.get("step_index"),
                    "page_type": step.get("observation", {}).get("page_type", ""),
                    "action": step.get("action", {}).get("action", ""),
                    "action_target": step.get("action", {}).get("target", ""),
                    "action_success": (step.get("action_result") or {}).get(
                        "success", False
                    ),
                })
            result["page_flow"] = page_flow

    except Exception as e:
        result["episode_run"] = {"error": str(e)[:500]}
        import traceback
        traceback.print_exc()

    # Routing decision
    inv = result.get("invariants", {})
    episode = result.get("episode_run", {})
    term_reason = episode.get("termination_reason", "")

    if episode.get("success"):
        route = "A"
        reason = "Expert succeeded — environment contract OK"
    elif inv.get("failure_reasons") and "missing_submission" in inv.get(
        "failure_reasons", []
    ):
        route = "C"
        reason = (
            "Expert got missing_submission — environment/verifier contract bug"
        )
    elif term_reason == "model_output_failure_limit":
        route = "C"
        reason = "Expert hit parse failures — check environment compatibility"
    else:
        route = "C"
        reason = f"Unexpected failure: {term_reason}"

    result["route"] = route
    result["route_reason"] = reason

    print(f"\n  Termination reason : {term_reason}")
    print(f"  Verifier success   : {inv.get('verifier_success', False)}")
    print(f"  Failure reasons    : {inv.get('failure_reasons', [])}")
    print(f"  Submission ID      : {inv.get('submission_id', 'NONE')}")
    print(f"  Decision type      : {inv.get('decision_type', 'NONE')}")
    print(f"  Model turns        : {episode.get('model_turns', 0)}")
    print(f"  Env steps          : {episode.get('environment_steps', 0)}")
    print(f"\n  ROUTE: {route} — {reason}")

    # Save result
    if output_dir:
        out_path = output_dir / "phase1_expert_replay.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str)
        )
        print(f"  Saved: {out_path}")

    return result


# ---------------------------------------------------------------------------
# Phase 2: No-Solution Task Audit
# ---------------------------------------------------------------------------


def phase2_no_solution_audit(
    output_dir: Path,
    seeds: list[int] = None,
    base_model_path: str = DEFAULT_BASE_MODEL,
    max_new_tokens: int = 128,
    max_env_steps: int = 15,
) -> dict:
    """Audit all no_feasible_product tasks across all configured seeds."""
    # Lazy imports to avoid PyTorch AllocatorConfig crash at module load
    from miniwebwork.model_agent.rollout_collector import (
        RolloutCollector, RolloutGroup, compute_group_stats, compute_no_solution_metrics,
    )
    from miniwebwork.tasks import load_public_tasks

    print("\n" + "=" * 70)
    print("PHASE 2: No-Solution Task Audit")
    print("=" * 70)

    if seeds is None:
        seeds = [0, 42, 1234, 20260726]

    # Find all no_feasible_product tasks
    public_tasks = load_public_tasks()
    no_sol_tasks = [t for t in public_tasks if t.get("task_type") == "no_feasible_product"]

    if not no_sol_tasks:
        print("WARNING: No no_feasible_product tasks found!")
        return {"error": "No no_feasible_product tasks found"}

    print(f"\nNo-solution tasks ({len(no_sol_tasks)}):")
    for t in no_sol_tasks:
        print(f"  {t['task_id']}: {t.get('instruction', '')[:80]}")

    # ---- Evaluate each seed in a subprocess (fresh GPU per seed) ----
    all_results = {}
    adapter_dir = PROJECT_ROOT / "outputs" / "m2_2r"
    subprocess_outputs = output_dir / "phase2_subprocess"
    subprocess_outputs.mkdir(parents=True, exist_ok=True)

    eval_script = PROJECT_ROOT / "scripts" / "m3_0a_eval_single_seed.py"
    python_exe = sys.executable

    for seed in seeds:
        adapter_path = _get_adapter_path(seed, adapter_dir)
        label = f"seed_{seed}" if seed else "base_canonical_v2"
        print(f"\n--- Evaluating {label} ---")

        for task in no_sol_tasks:
            task_id = task["task_id"]
            out_file = subprocess_outputs / f"seed_{seed}_{task_id}.json"

            if out_file.exists():
                print(f"  [{task_id}] cached")
                continue

            cmd = [
                python_exe, str(eval_script),
                "--seed", str(seed),
                "--task-id", task_id,
                "--base-model", base_model_path,
                "--max-new-tokens", str(max_new_tokens),
                "--max-env-steps", str(max_env_steps),
                "--output", str(out_file),
            ]
            if adapter_path:
                cmd.extend(["--adapter-path", adapter_path])

            subprocess_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
            subprocess_env["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks")
            # Workaround for PyTorch 2.10.0 AllocatorConfig crash on sm_100 (Blackwell)
            subprocess_env.setdefault("TORCH_CUDA_ARCH_LIST", "80;86;89")

            print(f"  [{task_id}] spawning...", end=" ", flush=True)
            t0 = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=False,
                    timeout=300,
                    env=subprocess_env,
                )
                if proc.returncode != 0:
                    print(f"FAIL(exit={proc.returncode})")
                else:
                    print(f"done ({time.time()-t0:.0f}s)")
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT")
            except Exception as e:
                print(f"ERROR: {e}")

        # Aggregate results for this seed
        seed_results = []
        for task in no_sol_tasks:
            task_id = task["task_id"]
            out_file = subprocess_outputs / f"seed_{seed}_{task_id}.json"
            if out_file.exists():
                try:
                    data = json.loads(out_file.read_text())
                    seed_results.append(data)
                except Exception:
                    pass

        behavior_metrics = compute_no_solution_metrics(seed_results)
        all_results[label] = {
            "seed": seed,
            "adapter_path": adapter_path or base_model_path,
            "per_task": seed_results,
            "behavior_metrics": behavior_metrics,
            "success_count": sum(1 for r in seed_results if r.get("success")),
            "total_tasks": len(seed_results),
        }
        # Brief pause between tasks
        time.sleep(1)

    # Print summary for each seed
    for label, data in all_results.items():
        bm = data.get("behavior_metrics", {})
        print(f"\n  Summary for {label}:")
        print(f"    Success: {data['success_count']}/{data['total_tasks']}")
        print(f"    Empty result reached: {bm.get('empty_result_reached_rate', 0):.1%}")
        print(f"    No-solution emitted:  {bm.get('no_solution_selected_rate', 0):.1%}")
        print(f"    No-solution clicked:  {bm.get('no_solution_clicked_rate', 0):.1%}")
        print(f"    Form reached:         {bm.get('form_reached_rate', 0):.1%}")
        print(f"    Submission persisted: {bm.get('submission_persisted_rate', 0):.1%}")
        print(f"    Verifier success:     {bm.get('verifier_success_rate', 0):.1%}")
        print(f"    Missing submission:   {bm.get('missing_submission_count', 0)}")
        print(f"    False no-solution:    {bm.get('false_no_solution_count', 0)}")

    # Determine where each seed fails in the pipeline
    pipeline_analysis = {}
    for label, data in all_results.items():
        bm = data["behavior_metrics"]
        # Find the first stage with < 100% rate
        stages = [
            ("empty_result", bm.get("empty_result_reached_rate", 0)),
            ("no_sol_emitted", bm.get("no_solution_selected_rate", 0)),
            ("no_sol_clicked", bm.get("no_solution_clicked_rate", 0)),
            ("form_reached", bm.get("form_reached_rate", 0)),
            ("submitted", bm.get("submission_persisted_rate", 0)),
            ("verifier_ok", bm.get("verifier_success_rate", 0)),
        ]
        bottleneck = "unknown"
        for stage_name, rate in stages:
            if rate < 1.0:
                bottleneck = stage_name
                break
        pipeline_analysis[label] = {
            "bottleneck_stage": bottleneck,
            "stage_rates": {s[0]: s[1] for s in stages},
        }
        print(f"\n  {label} bottleneck: {bottleneck}")

    result = {
        "no_solution_tasks": [t["task_id"] for t in no_sol_tasks],
        "seeds_evaluated": list(all_results.keys()),
        "seed_results": all_results,
        "pipeline_analysis": pipeline_analysis,
    }

    # Save
    out_path = output_dir / "phase2_no_solution_audit.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Saved: {out_path}")

    return result


# ---------------------------------------------------------------------------
# Phase 3: Random Rollout Probe
# ---------------------------------------------------------------------------


def phase3_rollout_probe(
    task_ids: list[str],
    policy_seed: int,
    output_dir: Path,
    base_model_path: str = DEFAULT_BASE_MODEL,
    K: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    max_env_steps: int = 15,
) -> dict:
    """Run K-trajectory rollout probe on no-solution tasks.

    Uses seed_1234 (lowest valid loss) as the rollout policy.
    """
    # Lazy imports to avoid PyTorch AllocatorConfig crash at module load
    from miniwebwork.model_agent.rollout_collector import RolloutCollector, compute_no_solution_metrics

    print("\n" + "=" * 70)
    print(f"PHASE 3: Random Rollout Probe (seed_{policy_seed}, K={K})")
    print("=" * 70)
    print(f"  Tasks: {task_ids}")
    print(f"  Temperature: {temperature}, Top-p: {top_p}")

    adapter_path = _get_adapter_path(policy_seed, PROJECT_ROOT / "outputs" / "m2_2r")
    if not adapter_path:
        print(f"ERROR: Adapter not found for seed_{policy_seed}")
        return {"error": f"Adapter not found for seed_{policy_seed}"}

    backend, agent, tokenizer = load_policy(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    policy_version = f"seed_{policy_seed}"
    collector = RolloutCollector(
        backend=backend,
        agent=agent,
        output_dir=output_dir,
        prompt_contract_version="browser_agent_v2",
    )
    collector.backend._policy_version = policy_version

    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks")

    groups = []
    start_time = time.time()

    for i, task_id in enumerate(task_ids):
        print(f"\n--- Task {i+1}/{len(task_ids)}: {task_id} ---")
        group = collector.collect_group(
            task_id=task_id,
            K=K,
            temperature=temperature,
            top_p=top_p,
            policy_version=policy_version,
            verbose=True,
        )
        collector.save_group(group)
        groups.append(group)

    elapsed = time.time() - start_time

    # Compute aggregate statistics
    total_trajectories = sum(len(g.trajectories) for g in groups)
    total_successes = sum(g.success_count for g in groups)
    tasks_with_success = sum(1 for g in groups if g.success_count > 0)
    tasks_with_variance = sum(1 for g in groups if g.group_reward_std > 0)
    tasks_valid = sum(1 for g in groups if g.valid_for_update)

    # Per-task summary
    per_task_summary = []
    for g in groups:
        # Compute no-solution behavior for each task
        behavior = compute_no_solution_metrics(g.trajectories)
        per_task_summary.append({
            "task_id": g.task_id,
            "K": len(g.trajectories),
            "success_count": g.success_count,
            "group_reward_mean": g.group_reward_mean,
            "group_reward_std": g.group_reward_std,
            "valid_for_update": g.valid_for_update,
            "reward_sequence": g.reward_sequence,
            "submission_count": g.submission_count,
            "no_solution_selected_count": g.no_solution_selected_count,
            "behavior": behavior,
        })

    result = {
        "policy_seed": policy_seed,
        "adapter_path": adapter_path,
        "K": K,
        "temperature": temperature,
        "top_p": top_p,
        "num_tasks": len(task_ids),
        "total_trajectories": total_trajectories,
        "total_successes": total_successes,
        "tasks_with_success": tasks_with_success,
        "tasks_with_variance": tasks_with_variance,
        "tasks_valid_for_update": tasks_valid,
        "runtime_s": elapsed,
        "per_task": per_task_summary,
        "groups": [g.to_dict() for g in groups],
    }

    # Save summary
    summary_path = collector.save_groups_summary(groups)
    full_path = output_dir / "phase3_rollout_probe.json"
    full_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    print(f"\n{'='*60}")
    print(f"ROLLOUT PROBE SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks: {len(task_ids)}, Trajectories: {total_trajectories}")
    print(f"  Successes: {total_successes}/{total_trajectories}")
    print(f"  Tasks with success: {tasks_with_success}/{len(task_ids)}")
    print(f"  Tasks with variance: {tasks_with_variance}/{len(task_ids)}")
    print(f"  Tasks valid for update: {tasks_valid}/{len(task_ids)}")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"\n  Per-task detail:")
    for pts in per_task_summary:
        print(
            f"    {pts['task_id']}: "
            f"success={pts['success_count']}/{pts['K']}, "
            f"mean={pts['group_reward_mean']:.2f}, "
            f"std={pts['group_reward_std']:.3f}, "
            f"valid={pts['valid_for_update']}"
        )
    print(f"\n  Saved: {full_path}")
    print(f"  Saved: {summary_path}")

    return result


# ---------------------------------------------------------------------------
# Phase 4: Report and Route Decision
# ---------------------------------------------------------------------------


def make_route_decision(
    phase1: dict,
    phase2: dict,
    phase3: dict,
) -> dict:
    """Combine all phase results and make routing decision.

    Routes:
    - A: Environment contract OK, rollout shows reward variance → GRPO pilot
    - B: Environment OK, but zero rewards in all rollouts → SFT patch needed
    - C: Environment contract broken → fix environment first
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Route Decision")
    print("=" * 70)

    decision = {
        "route": "UNKNOWN",
        "reason": "",
        "blockers": [],
        "recommendations": [],
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
    }

    # Check Phase 1: Expert replay
    p1_route = phase1.get("route", "C")
    if p1_route == "C":
        decision["route"] = "C"
        decision["reason"] = phase1.get("route_reason", "Environment contract failure")
        decision["blockers"].append("Expert cannot complete no_solution task — environment/verifier bug")
        decision["recommendations"].append(
            "Fix submission contract: "
            "Environment → Web submission → DB persistence → Verifier lookup"
        )
        print(f"\n  ROUTE C: {decision['reason']}")
        print(f"  Blockers: {decision['blockers']}")
        return decision

    assert p1_route == "A", f"Unexpected Phase 1 route: {p1_route}"
    print(f"\n  Phase 1: PASS — expert successfully completed no_solution task")

    # Check Phase 3: Rollout probe
    if "error" in phase3:
        decision["route"] = "C"
        decision["reason"] = f"Rollout probe failed: {phase3['error']}"
        decision["blockers"].append("Rollout probe execution failure")
        print(f"\n  ROUTE C: {decision['reason']}")
        return decision

    tasks_valid = phase3.get("tasks_valid_for_update", 0)
    num_tasks = phase3.get("num_tasks", 0)
    tasks_with_success = phase3.get("tasks_with_success", 0)
    tasks_with_variance = phase3.get("tasks_with_variance", 0)

    print(f"\n  Phase 3 results:")
    print(f"    Tasks with success: {tasks_with_success}/{num_tasks}")
    print(f"    Tasks with variance: {tasks_with_variance}/{num_tasks}")
    print(f"    Tasks valid for GRPO: {tasks_valid}/{num_tasks}")

    if tasks_valid > 0:
        decision["route"] = "A"
        decision["reason"] = (
            f"Environment contract OK + "
            f"{tasks_valid}/{num_tasks} tasks show reward variance under stochastic sampling"
        )
        decision["recommendations"] = [
            "Proceed to M3.0B: Outcome-only GRPO pilot",
            "First batch: single task group with K=8 trajectories",
            "Reward: binary verifier success (1/0)",
            "No process rewards or KL penalty in first iteration",
        ]
        print(f"\n  ROUTE A: {decision['reason']}")
    elif tasks_with_success > 0:
        decision["route"] = "B"
        decision["reason"] = (
            "Expert succeeds but rollout produces zero reward variance "
            "(policy always fails despite expert demonstration)"
        )
        decision["blockers"].append("No reward variance — cannot compute group advantages")
        decision["recommendations"] = [
            "Conduct M2.3-mini: targeted SFT patch for no_solution tasks",
            "Add 20-40 expert trajectories covering no_solution pipeline stages",
            "Focus: empty-result recognition, no_solution form submission",
            "Re-run rollout probe after patch",
        ]
        print(f"\n  ROUTE B: {decision['reason']}")
    else:
        decision["route"] = "B"
        decision["reason"] = (
            "Environment contract OK but model never succeeds "
            "under stochastic sampling"
        )
        decision["blockers"].append("Zero successes across all rollout trajectories")
        decision["recommendations"] = [
            "Diagnose failure mode from rollout trajectories",
            "Check if model never reaches empty result or never clicks no_solution",
            "Consider M2.3-mini SFT patch or prompt engineering fix",
        ]
        print(f"\n  ROUTE B (zero success): {decision['reason']}")

    return decision


def write_final_report(
    decision: dict,
    output_dir: Path,
) -> Path:
    """Write the final M3.0A audit report."""
    report = {
        "m3_0a_status": "COMPLETE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "route": decision["route"],
        "route_reason": decision["reason"],
        "blockers": decision["blockers"],
        "recommendations": decision["recommendations"],
        "readiness": {
            "GENERAL_TASK_RL_READINESS": decision["route"] == "A",
            "NO_SOLUTION_CAPABILITY_GATE": decision["route"] in ("A", "B"),
            "READY_FOR_M3_0_ROLLOUT_PILOT": decision["route"] in ("A", "B"),
            "READY_FOR_GRPO_POLICY_UPDATE": decision["route"] == "A",
        },
        "phase_summaries": {
            "phase1_expert_replay": {
                "route": decision["phase1"].get("route"),
                "expert_success": decision["phase1"]
                .get("episode_run", {})
                .get("success", False),
                "verifier_success": decision["phase1"]
                .get("invariants", {})
                .get("verifier_success", False),
            },
            "phase2_no_solution_audit": {
                "num_tasks": len(decision["phase2"].get("no_solution_tasks", [])),
                "seeds_evaluated": len(decision["phase2"].get("seeds_evaluated", [])),
            },
            "phase3_rollout_probe": {
                "num_tasks": decision["phase3"].get("num_tasks", 0),
                "total_trajectories": decision["phase3"].get("total_trajectories", 0),
                "tasks_valid_for_update": decision["phase3"].get("tasks_valid_for_update", 0),
            },
        },
    }

    path = output_dir / "m3_0a_final_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Final report: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="M3.0A Rollout Readiness Audit"
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "m3_0a",
        help="Output directory",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=1234,
        help="Seed to use for rollout probe (default: 1234, lowest valid loss)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=8,
        help="Number of trajectories per task for rollout probe",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for rollout probe",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p for rollout probe",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens per generation",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=15,
        help="Max environment steps per episode",
    )
    parser.add_argument(
        "--route-a-check",
        action="store_true",
        help="Quick check: only run Phase 1 expert + K=2 rollout on TASK-012",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("M3.0A: Rollout Readiness Audit")
    print("=" * 70)
    print(f"  Base model  : {args.base_model}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Policy seed : {args.policy_seed}")
    print(f"  K           : {args.K}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Top-p       : {args.top_p}")

    phase1_result = {}
    phase2_result = {}
    phase3_result = {}

    # Phase 1: Expert Replay
    if args.phase in ("1", "all", "route-a-check"):
        phase1_result = phase1_expert_replay(
            task_id="TASK-012",
            output_dir=args.output_dir,
        )
        if args.route_a_check:
            # Quick check: only 2 trajectories on TASK-012
            print("\n--- Quick rollout probe (K=2, TASK-012 only) ---")
            adapter_path = _get_adapter_path(
                args.policy_seed, PROJECT_ROOT / "outputs" / "m2_2r"
            )
            if adapter_path:
                backend, agent, _ = load_policy(
                    base_model_path=args.base_model,
                    adapter_path=adapter_path,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                collector = RolloutCollector(
                    backend, agent, args.output_dir
                )
                group = collector.collect_group(
                    task_id="TASK-012",
                    K=2,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    policy_version=f"seed_{args.policy_seed}",
                    verbose=True,
                )
                collector.save_group(group)
                phase3_result = {
                    "num_tasks": 1,
                    "total_trajectories": 2,
                    "tasks_with_success": group.success_count,
                    "tasks_with_variance": 1 if group.group_reward_std > 0 else 0,
                    "tasks_valid_for_update": 1 if group.valid_for_update else 0,
                    "groups": [group.to_dict()],
                }

    # Phase 2: No-Solution Audit
    if args.phase in ("2", "all"):
        phase2_result = phase2_no_solution_audit(
            output_dir=args.output_dir,
            base_model_path=args.base_model,
            max_new_tokens=args.max_new_tokens,
            max_env_steps=args.max_env_steps,
        )

    # Phase 3: Rollout Probe
    if args.phase in ("3", "all"):
        if not phase2_result or "error" in phase2_result:
            # Need to run audit first to find no-solution tasks
            print("Running Phase 2 first to identify no-solution tasks...")
            phase2_result = phase2_no_solution_audit(
                output_dir=args.output_dir,
                base_model_path=args.base_model,
                max_new_tokens=args.max_new_tokens,
                max_env_steps=args.max_env_steps,
            )

        no_sol_tasks = phase2_result.get("no_solution_tasks", [])
        if not no_sol_tasks:
            # Fallback: known no-solution tasks
            no_sol_tasks = ["TASK-004", "TASK-012"]

        phase3_result = phase3_rollout_probe(
            task_ids=no_sol_tasks,
            policy_seed=args.policy_seed,
            output_dir=args.output_dir,
            base_model_path=args.base_model,
            K=args.K,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_env_steps=args.max_env_steps,
        )

    # Phase 4: Route Decision
    if args.phase == "all":
        decision = make_route_decision(phase1_result, phase2_result, phase3_result)
        write_final_report(decision, args.output_dir)

        print(f"\n{'='*70}")
        print(f"M3.0A AUDIT COMPLETE — ROUTE {decision['route']}")
        print(f"{'='*70}")
        print(f"  {decision['reason']}")
        if decision["blockers"]:
            print(f"  Blockers:")
            for b in decision["blockers"]:
                print(f"    - {b}")
        if decision["recommendations"]:
            print(f"  Recommendations:")
            for r in decision["recommendations"]:
                print(f"    - {r}")
    elif args.phase == "route-a-check":
        print("\nQuick check complete. Review Phase 1 and rollout results above.")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
