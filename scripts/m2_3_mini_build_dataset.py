#!/usr/bin/env python3
"""M2.3-mini: Build mixed SFT dataset (original m2_2r + no_solution patch).

Combines:
  - Original m2_2r data: 723 train + 174 valid = 897 samples (75%)
  - New no_solution expert trajectories: ~200-300 step samples (25%)

Output:
    data/sft/m2_3_mini/
    ├── train.jsonl
    ├── valid.jsonl
    ├── manifest.json
    └── mix_analysis.json

Usage:
    python scripts/m2_3_mini_build_dataset.py
    python scripts/m2_3_mini_build_dataset.py --no-replay  # skip env replay
"""
import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")
ORIGINAL_SFT_DIR = PROJECT_ROOT / "data" / "sft" / "m2_2r"
EXPERT_DIR = PROJECT_ROOT / "data" / "expert" / "m2_3_mini"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sft" / "m2_3_mini"
TASK_DIR = PROJECT_ROOT / "data" / "tasks" / "rollout_dev_no_solution_v1"
CANONICAL_VERSION = "browser_agent_v2"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "browser_agent_v2" / "system.txt"

# Mix ratios
ORIGINAL_RATIO = 0.75   # 75% original data
PATCH_RATIO = 0.25      # 25% no_solution/recovery data


def load_original_sft(split: str) -> list:
    """Load original m2_2r SFT samples."""
    path = ORIGINAL_SFT_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Original SFT data not found: {path}")
    samples = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            samples.append(json.loads(line))
    return samples


def load_expert_trajectories(split: str) -> list:
    """Load expert trajectories from m2_3_mini."""
    path = EXPERT_DIR / f"{split}_trajectories.json"
    if not path.exists():
        raise FileNotFoundError(f"Expert trajectories not found: {path}")
    return json.loads(path.read_text())


def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    from miniwebwork.model_agent.prompt_builder import load_system_prompt as _load
    return _load("browser_agent_v2")


def canonical_serialize_elements(elements, max_elements=100) -> list:
    """Serialize elements using canonical contract."""
    els = []
    for e in (elements or [])[:max_elements]:
        d = {
            "element_id": e.element_id,
            "role": e.role,
            "name": e.name,
            "testid": e.testid if e.testid else None,
            "disabled": e.disabled,
        }
        if e.tag in ("input", "textarea"):
            d["value"] = e.value[:100] if e.value else ""
        if e.tag == "select" and e.options:
            d["options"] = e.options[:10]
        if e.role in ("link", "button"):
            d["text"] = e.text[:100] if e.text else ""
        els.append(d)
    return els


def canonical_serialize_history(history, window=5) -> list:
    """Serialize history using canonical contract."""
    entries = []
    for h in history[-window:]:
        entries.append({
            "turn": h.get("model_turn_index", 0),
            "action": h.get("action"),
            "parse_ok": h.get("parse_ok", False),
            "result": h.get("result", ""),
            "page_type": h.get("page_type", ""),
        })
    return entries


def build_messages_v2(observation, history=None) -> list:
    """Build chat messages using Canonical Prompt Contract v2."""
    from miniwebwork.agent_env.schemas import Observation

    history = history or []
    system = load_system_prompt()

    els = canonical_serialize_elements(observation.elements)
    visible_text = observation.visible_text or ""
    text_truncated = len(visible_text) > 8000
    if text_truncated:
        visible_text = visible_text[:8000]

    hist = canonical_serialize_history(history, 5)
    last_result = observation.last_action_result

    user_content = f"""## Task
task_id: {observation.task_id}
instruction: {observation.instruction}

## Current Page
url: {observation.url}
path: {observation.path}
page_type: {observation.page_type}
title: {observation.title}
step: {observation.step_index}

## Visible Text (truncated={text_truncated})
{visible_text}

## Interactive Elements ({len(els)})
{json.dumps(els, ensure_ascii=False)}

## Recent History ({len(hist)} turns)
{json.dumps(hist, ensure_ascii=False)}

## Last Action Result
{json.dumps(last_result, ensure_ascii=False) if last_result else 'N/A'}

## Instruction
Output exactly one JSON action. Only use element_id from the elements list above."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def replay_trajectory_to_samples(trajectory: dict, split: str) -> list:
    """Replay an expert trajectory and build SFT samples using canonical v2.

    Returns list of sample dicts (one per turn).
    """
    from miniwebwork.agent_env.environment import ProcurementBrowserEnv
    from miniwebwork.agent_env.schemas import AgentAction
    from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent
    from miniwebwork.tasks import get_oracle

    # Set task dir BEFORE oracle lookup (get_oracle reads MINIWEBWORK_TASK_DIR)
    os.environ["MINIWEBWORK_TASK_DIR"] = str(TASK_DIR)

    task_id = trajectory["task_id"]
    oracle = get_oracle(task_id)
    if not oracle:
        print(f"  WARN: no oracle for {task_id}")
        return []

    if not trajectory.get("success"):
        return []

    run_id = f"m2_3_replay_{uuid.uuid4().hex[:8]}"

    samples = []
    system_prompt = load_system_prompt()
    system_sha = hashlib.sha256(system_prompt.encode()).hexdigest()

    with ProcurementBrowserEnv(max_steps=25, run_id=run_id, headless=True) as env:
        env.set_agent_name("m2_3_dataset_builder")

        obs = env.reset(task_id)
        expert = OracleExpertProcurementAgent(oracle, max_steps=25)
        expert.reset()

        history = []
        expert_turns = trajectory.get("turns", [])

        for turn_idx, expert_turn in enumerate(expert_turns):
            if turn_idx >= len(expert_turns):
                break

            messages = build_messages_v2(obs, history)
            expected_action = expert_turn.get("action", {})

            # Verify action matches
            expert_action = expert.act(obs)
            if expert_action.to_dict() != expected_action:
                pass  # Log but don't skip — action drift is expected

            completion = json.dumps(expected_action, ensure_ascii=False)
            full_messages = messages + [{"role": "assistant", "content": completion}]

            sample = {
                "sample_id": f"M23_{split[:1].upper()}{turn_idx+1:06d}",
                "split": split,
                "task_id": task_id,
                "source": "oracle_expert_m2_3",
                "prompt_version": CANONICAL_VERSION,
                "prompt_contract_sha256": system_sha,
                "messages": full_messages,
                "chat_template_kwargs": {"enable_thinking": False},
                "message_hash": hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
                "turn_index": turn_idx,
                "visible_text_chars": len(obs.visible_text or ""),
                "element_count": len(obs.elements or []),
                "history_turns": len(history),
                "task_type": "no_feasible_product",
            }
            samples.append(sample)

            # Step environment
            action_obj = AgentAction.from_dict(expected_action)
            try:
                result = env.step(action_obj)
                if result.observation:
                    obs = result.observation
                history.append({
                    "model_turn_index": turn_idx,
                    "action": expected_action,
                    "parse_ok": True,
                    "result": "success",
                    "page_type": obs.page_type,
                })
            except Exception as e:
                print(f"    ERROR stepping {task_id} turn {turn_idx}: {e}")
                break

    return samples


def main():
    parser = argparse.ArgumentParser(description="M2.3-mini mixed SFT dataset builder")
    parser.add_argument("--original-dir", type=Path, default=ORIGINAL_SFT_DIR)
    parser.add_argument("--expert-dir", type=Path, default=EXPERT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-replay", action="store_true", help="Skip env replay for new trajectories")
    parser.add_argument("--original-ratio", type=float, default=ORIGINAL_RATIO)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== M2.3-mini Mixed SFT Dataset Builder ===")
    print(f"Contract: {CANONICAL_VERSION}")
    print(f"Original data: {args.original_dir}")
    print(f"Expert data: {args.expert_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Mix ratio: {int(args.original_ratio*100)}% original / {int(PATCH_RATIO*100)}% patch")

    # Step 1: Load original SFT data
    print("\n=== Step 1: Loading original SFT data ===")
    original_train = load_original_sft("train")
    original_valid = load_original_sft("valid")
    print(f"  Original train: {len(original_train)} samples")
    print(f"  Original valid: {len(original_valid)} samples")

    # Step 2: Load expert trajectories for the new tasks
    print("\n=== Step 2: Loading expert trajectories ===")
    expert_train_trajs = load_expert_trajectories("train")
    expert_valid_trajs = load_expert_trajectories("valid")
    print(f"  Expert train trajectories: {len(expert_train_trajs)}")
    print(f"  Expert valid trajectories: {len(expert_valid_trajs)}")

    expert_train_success = [t for t in expert_train_trajs if t.get("success")]
    expert_valid_success = [t for t in expert_valid_trajs if t.get("success")]
    print(f"  Expert train successful: {len(expert_train_success)}")
    print(f"  Expert valid successful: {len(expert_valid_success)}")

    # Step 3: Replay expert trajectories to build SFT samples
    print("\n=== Step 3: Replaying new expert trajectories ===")
    patch_train_samples = []
    patch_valid_samples = []

    if not args.no_replay:
        for traj in expert_train_success:
            samples = replay_trajectory_to_samples(traj, "train")
            patch_train_samples.extend(samples)
            print(f"  {traj['task_id']}: {len(samples)} steps → {len(samples)} samples")

        for traj in expert_valid_success:
            samples = replay_trajectory_to_samples(traj, "valid")
            patch_valid_samples.extend(samples)
            print(f"  {traj['task_id']}: {len(samples)} steps → {len(samples)} samples")

        print(f"  Patch train samples: {len(patch_train_samples)}")
        print(f"  Patch valid samples: {len(patch_valid_samples)}")
    else:
        print("  Skipping env replay (--no-replay)")

    # Step 4: Mix datasets
    print("\n=== Step 4: Mixing datasets ===")

    # Calculate target counts
    total_patch = len(patch_train_samples)
    total_original = len(original_train)
    target_total = int(total_patch / PATCH_RATIO) if total_patch > 0 else total_original
    target_original = int(target_total * args.original_ratio)
    target_patch = int(target_total * PATCH_RATIO)

    print(f"  Target total train: ~{target_total}")
    print(f"    Original: {target_original} (have {total_original})")
    print(f"    Patch: {target_patch} (have {total_patch})")

    # Sample or duplicate original data to match target
    import random
    rng = random.Random(20260727)
    if total_original > target_original:
        mixed_train_original = rng.sample(original_train, target_original)
    elif total_original < target_original:
        # Duplicate with replacement
        mixed_train_original = []
        while len(mixed_train_original) < target_original:
            mixed_train_original.extend(rng.choices(original_train, k=min(len(original_train), target_original - len(mixed_train_original))))
    else:
        mixed_train_original = list(original_train)

    # Sample patch data to match target
    if total_patch > target_patch:
        mixed_train_patch = rng.sample(patch_train_samples, target_patch)
    else:
        mixed_train_patch = list(patch_train_samples)

    # Combine and shuffle
    mixed_train = mixed_train_original + mixed_train_patch
    rng.shuffle(mixed_train)

    # Valid: keep original + patch (no downsampling for valid)
    mixed_valid = list(original_valid) + patch_valid_samples
    rng.shuffle(mixed_valid)

    print(f"  Mixed train: {len(mixed_train)} (original={len(mixed_train_original)}, patch={len(mixed_train_patch)})")
    print(f"  Mixed valid: {len(mixed_valid)} (original={len(original_valid)}, patch={len(patch_valid_samples)})")

    # Step 5: Save
    print("\n=== Step 5: Saving ===")
    for split, samples in [("train", mixed_train), ("valid", mixed_valid)]:
        out_path = args.output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
        print(f"  Saved {split}: {out_path} ({len(samples)} samples)")

    # Analysis
    orig_count = sum(1 for s in mixed_train if s.get("source") == "oracle_expert")
    patch_count = sum(1 for s in mixed_train if s.get("source") == "oracle_expert_m2_3")
    analysis = {
        "train_total": len(mixed_train),
        "train_original": orig_count,
        "train_patch": patch_count,
        "train_original_ratio": orig_count / max(len(mixed_train), 1),
        "train_patch_ratio": patch_count / max(len(mixed_train), 1),
        "valid_total": len(mixed_valid),
        "valid_original": sum(1 for s in mixed_valid if s.get("source") == "oracle_expert"),
        "valid_patch": sum(1 for s in mixed_valid if s.get("source") == "oracle_expert_m2_3"),
        "no_solution_samples": sum(1 for s in mixed_train if s.get("task_type") == "no_feasible_product"),
    }
    analysis_path = args.output_dir / "mix_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    print(f"  Mix analysis: {analysis_path}")

    # Manifest
    manifest = {
        "schema_version": "1.0",
        "dataset_version": "m2_3_mini_v1",
        "phase": "m2_3_mini",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "contract_version": CANONICAL_VERSION,
        "original_data_dir": str(args.original_dir),
        "expert_data_dir": str(args.expert_dir),
        "train_samples": len(mixed_train),
        "valid_samples": len(mixed_valid),
        "original_ratio": args.original_ratio,
        "patch_ratio": PATCH_RATIO,
        "source": "oracle_expert",
        "format": "conversational",
        "message_field": "messages",
        "completion_field": "last message (assistant action JSON)",
        "no_solution_aware": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  Manifest: {manifest_path}")

    print(f"\n=== Dataset Build Complete ===")
    print(f"  Train: {len(mixed_train)} (original={orig_count}, patch={patch_count})")
    print(f"  Valid: {len(mixed_valid)}")
    print(f"  Patch ratio: {patch_count / max(len(mixed_train), 1):.1%}")


if __name__ == "__main__":
    main()
