#!/usr/bin/env python3
"""M2.3-mini: Collect expert trajectories for rollout_dev_no_solution_v1 tasks.

Collects clean expert trajectories for all 38 no_solution tasks in the
rollout_dev_no_solution_v1 task set. Also extracts intermediate recovery
states (off-expert correction examples) from the expert's own rollouts.

Output:
    data/expert/m2_3_mini/
    ├── train_trajectories.json   # 28 expert trajectories
    ├── valid_trajectories.json   # 10 expert trajectories
    ├── recovery_states.json      # Intermediate obs+action pairs for recovery training
    └── manifest.json

Usage (local test):
    python scripts/m2_3_mini_collect_expert.py --split train --max-tasks 3

Usage (Slurm):
    sbatch scripts/slurm/m2_3_mini_collect_expert.sbatch
"""
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")
TASK_DIR = PROJECT_ROOT / "data" / "tasks" / "rollout_dev_no_solution_v1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "expert" / "m2_3_mini"


def load_task_set(split: str) -> list:
    """Load oracle tasks for a split."""
    public_path = TASK_DIR / f"{split}_public.jsonl"
    oracle_path = TASK_DIR / f"{split}_oracle.jsonl"

    if not public_path.exists() or not oracle_path.exists():
        raise FileNotFoundError(f"Task files not found in {TASK_DIR} for split={split}")

    public_lines = public_path.read_text().strip().split("\n")
    oracle_lines = oracle_path.read_text().strip().split("\n")

    tasks = []
    for pub, ora in zip(public_lines, oracle_lines):
        pub_data = json.loads(pub)
        ora_data = json.loads(ora)
        tasks.append({
            "task_id": pub_data["task_id"],
            "instruction": pub_data["instruction"],
            "start_path": pub_data.get("start_path", "/products"),
            "oracle": ora_data,
        })
    return tasks


def collect_expert_trajectory(task_id: str, oracle: dict, env: ProcurementBrowserEnv,
                               max_steps: int = 25) -> dict:
    """Run expert agent on a single task and record full trajectory."""
    obs = env.reset(task_id)
    agent = OracleExpertProcurementAgent(oracle, max_steps=max_steps)
    agent.reset()
    env.set_agent_name("oracle_expert_m2_3")

    turns = []
    recovery_states = []

    for step in range(max_steps):
        # Get expert action ONCE per step (agent.act mutates internal state).
        action = agent.act(obs)
        action_dict = action.to_dict()

        # Record recovery state BEFORE stepping — the observation a model
        # would see at this decision point, paired with the correct action.
        # Skip terminal finish actions (no learning value for recovery training).
        if action_dict.get("action") != "finish":
            recovery_states.append({
            "step": step,
            "observation_page": obs.page_type,
            "observation_url": obs.url,
            "observation_path": obs.path,
            "observation_title": obs.title,
            "visible_text_snippet": (obs.visible_text or "")[:500],
            "element_count": len(obs.elements or []),
            "element_ids": [e.element_id for e in (obs.elements or [])[:20]],
            "instruction": obs.instruction,
            "correct_action": action_dict,
        })

        try:
            result = env.step(action)
            terminated = result.terminated
            truncated = result.truncated
            reward = result.reward
            if result.observation:
                obs = result.observation
        except Exception as e:
            terminated = True
            truncated = False
            reward = 0.0
            turns.append({
                "step": step,
                "action": action_dict,
                "observation_page": "error",
                "reward": 0.0,
                "terminated": True,
                "error": str(e)[:200],
            })
            break

        turns.append({
            "step": step,
            "action": action_dict,
            "observation_page": obs.page_type,
            "reward": reward,
            "terminated": terminated,
        })

        if terminated or truncated:
            break

    # Get final result
    traj = env.trajectory
    success = reward > 0
    if traj and traj.verification:
        success = traj.verification.get("success", success)

    return {
        "task_id": task_id,
        "success": success,
        "steps": len(turns),
        "reward": reward,
        "verification": (traj.verification if traj else {}),
        "termination_reason": (traj.termination_reason if traj else ""),
        "turns": turns,
        "recovery_states": recovery_states,
    }


def main():
    parser = argparse.ArgumentParser(description="M2.3-mini expert trajectory collector")
    parser.add_argument("--split", choices=["train", "valid", "all"], default="all")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks (for testing)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--task-dir", type=Path, default=TASK_DIR)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MINIWEBWORK_TASK_DIR"] = str(args.task_dir)

    splits = ["train", "valid"] if args.split == "all" else [args.split]

    run_id = f"m2_3_mini_{uuid.uuid4().hex[:8]}"
    print(f"Run ID: {run_id}")
    print(f"Task dir: {args.task_dir}")
    print(f"Output dir: {args.output_dir}")

    all_metrics = {}
    all_trajectories = {}
    all_recovery = []

    with ProcurementBrowserEnv(max_steps=25, run_id=run_id, headless=args.headless) as env:
        env.set_agent_name("m2_3_mini_expert_collector")

        for split in splits:
            tasks = load_task_set(split)
            if args.max_tasks:
                tasks = tasks[:args.max_tasks]

            print(f"\n=== {split}: {len(tasks)} tasks ===")
            trajectories = []
            successes = 0

            for i, task in enumerate(tasks):
                tid = task["task_id"]
                oracle = task["oracle"]
                try:
                    traj = collect_expert_trajectory(tid, oracle, env)
                    trajectories.append(traj)
                    all_recovery.extend(traj.get("recovery_states", []))

                    status = "PASS" if traj["success"] else "FAIL"
                    if traj["success"]:
                        successes += 1
                    print(f"  [{i+1}/{len(tasks)}] {tid}: {status} "
                          f"({traj['steps']} steps) "
                          f"{traj.get('termination_reason', '')[:40]}")
                except Exception as e:
                    print(f"  [{i+1}/{len(tasks)}] {tid}: ERROR {str(e)[:100]}")
                    trajectories.append({
                        "task_id": tid,
                        "success": False,
                        "steps": 0,
                        "reward": 0.0,
                        "turns": [],
                        "recovery_states": [],
                        "error": str(e)[:200],
                    })

            out_path = args.output_dir / f"{split}_trajectories.json"
            out_path.write_text(json.dumps(trajectories, indent=2, ensure_ascii=False, default=str))
            print(f"  Saved {len(trajectories)} trajectories to {out_path}")

            all_trajectories[split] = trajectories
            all_metrics[split] = {
                "total": len(trajectories),
                "success": successes,
                "rate": successes / max(len(trajectories), 1),
            }
            print(f"  {split}: {successes}/{len(trajectories)} ({all_metrics[split]['rate']:.1%})")

    # Save recovery states
    recovery_path = args.output_dir / "recovery_states.json"
    recovery_path.write_text(json.dumps(all_recovery, indent=2, ensure_ascii=False))
    print(f"\nRecovery states: {len(all_recovery)} saved to {recovery_path}")

    # Manifest
    manifest = {
        "schema_version": "1.0",
        "phase": "m2_3_mini",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "task_dir": str(args.task_dir),
        "task_set_seed": 20260727,
        "train_task_count": len(all_trajectories.get("train", [])),
        "valid_task_count": len(all_trajectories.get("valid", [])),
        "train_success": all_metrics.get("train", {}).get("success", 0),
        "valid_success": all_metrics.get("valid", {}).get("success", 0),
        "recovery_states_count": len(all_recovery),
        "recovery_states_path": str(recovery_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")

    # Summary
    print("\n=== Summary ===")
    for split, m in all_metrics.items():
        print(f"  {split}: {m['success']}/{m['total']} ({m['rate']:.1%})")
    print(f"  Recovery states: {len(all_recovery)}")
    total_steps = sum(len(t.get("turns", [])) for t in all_trajectories.get("train", []))
    print(f"  Total step samples (train): {total_steps}")

    # Leakage check
    existing_ids = set()
    for legacy_dir in [PROJECT_ROOT / "data" / "tasks" / "m2_1"]:
        if legacy_dir.exists():
            for f in legacy_dir.glob("*_public.jsonl"):
                for line in f.read_text().strip().split("\n"):
                    if line.strip():
                        existing_ids.add(json.loads(line)["task_id"])

    new_ids = set()
    for traj in all_trajectories.get("train", []) + all_trajectories.get("valid", []):
        new_ids.add(traj["task_id"])

    leaked = new_ids & existing_ids
    if leaked:
        print(f"\nWARNING: {len(leaked)} task IDs overlap with existing set!")
        print(f"  Leaked IDs: {leaked}")
    else:
        print("\nNo task ID leakage detected.")

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
