"""M2.2 Dataset Builder: replay expert trajectories to build conversational prompt-completion pairs.

Uses TRL's conversational format:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
   "chat_template_kwargs": {"enable_thinking": False}}
"""

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent
from miniwebwork.model_agent.prompt_builder import build_messages, load_system_prompt
from miniwebwork.tasks import get_public_task

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAIN_TRAJ = PROJECT_ROOT / "data" / "expert" / "m2_1" / "train_trajectories.json"
VALID_TRAJ = PROJECT_ROOT / "data" / "expert" / "m2_1" / "valid_trajectories.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sft" / "m2_2"
DATASET_VERSION = "m2_2_sft_v1"


def load_trajectories(path: str) -> list:
    data = json.loads(Path(path).read_text())
    return [t for t in data if t.get("success", False)]


def replay_and_build(trajectories: list, split: str, env: ProcurementBrowserEnv) -> tuple:
    """Replay expert trajectories and build conversational message pairs."""
    samples = []
    mismatches = 0
    total_turns = 0

    for traj in trajectories:
        task_id = traj["task_id"]
        from miniwebwork.tasks import get_oracle
        oracle = get_oracle(task_id)
        if not oracle:
            print(f"  WARN: no oracle for {task_id}")
            continue

        # Reset environment
        obs = env.reset(task_id)
        expert = OracleExpertProcurementAgent(oracle, max_steps=25)
        expert.reset()
        env.set_agent_name("oracle_expert")

        history = []
        turn_idx = 0
        expert_turns = traj.get("turns", [])

        for expert_turn in expert_turns:
            if turn_idx >= len(expert_turns):
                break

            # Build system + user messages
            messages = build_messages(obs, history)

            # Truncate user message for training efficiency
            user_msg = messages[1]
            user_content = user_msg["content"]

            # Parse sections by ## headers (line-based to avoid regex pitfalls)
            sections = {}
            current_header = None
            current_lines = []
            for line in user_content.split('\n'):
                if line.startswith('## '):
                    if current_header is not None:
                        sections[current_header] = '\n'.join(current_lines)
                    current_header = line.strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            if current_header is not None:
                sections[current_header] = '\n'.join(current_lines)

            # 1. Truncate visible text to 300 chars
            for h in sections:
                if h.startswith('## Visible Text'):
                    vt = sections[h]
                    if len(vt) > 300:
                        sections[h] = vt[:300] + "..."
                    break

            # 2. Truncate elements to 8, keep only essential fields
            for h in sections:
                if h.startswith('## Interactive Elements'):
                    elements_str = sections[h]
                    try:
                        elements_list = json.loads(elements_str)
                        if len(elements_list) > 8:
                            minimal = []
                            for e in elements_list[:8]:
                                minimal.append({
                                    "element_id": e.get("element_id", ""),
                                    "role": e.get("role", ""),
                                    "name": (e.get("name") or "")[:20],
                                    "disabled": e.get("disabled", False),
                                })
                            sections[h] = json.dumps(minimal, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass
                    break

            # 3. Remove history section entirely for training
            sections.pop('## Recent History', None)
            for h in list(sections.keys()):
                if h.startswith('## Recent History'):
                    del sections[h]

            # Rebuild user content with canonical section order
            parts = []
            header_order = [
                '## Task',
                '## Current Page',
                '## Visible Text',
                '## Interactive Elements',
                '## Last Action Result',
                '## Instruction',
            ]
            for expected in header_order:
                for h in sections:
                    if h == expected or h.startswith(expected):
                        parts.append(h)
                        parts.append(sections[h])
                        break
            user_content = '\n'.join(parts)
            messages[1] = {"role": "user", "content": user_content}

            # Get expected action
            expected_action = expert_turn.get("action", {})

            # Verify action matches
            expert_action = expert.act(obs)
            if expert_action.to_dict() != expected_action:
                mismatches += 1
                if mismatches <= 3:
                    print(f"    MISMATCH {task_id} turn {turn_idx}: "
                          f"expected={expected_action}, got={expert_action.to_dict()}")

            # Build conversational format: system + user + assistant
            completion = json.dumps(expected_action, ensure_ascii=False)
            full_messages = messages + [{"role": "assistant", "content": completion}]

            samples.append({
                "sample_id": expert_turn.get("sample_id", f"{split[:1].upper()}{turn_idx+1:06d}"),
                "split": split,
                "task_id": task_id,
                "source": "oracle_expert",
                "prompt_version": "browser_agent_v1",
                "messages": full_messages,   # system + user + assistant (TRL conversational)
                "chat_template_kwargs": {"enable_thinking": False},
                "turn_index": turn_idx,
            })
            total_turns += 1

            # Step environment
            from miniwebwork.agent_env.schemas import AgentAction
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

            turn_idx += 1

    return samples, mismatches, total_turns


def main():
    print("=== M2.2 SFT Dataset Builder ===")
    print(f"Version: {DATASET_VERSION}")
    start_time = time.time()

    print("\nLoading trajectories...")
    train_trajs = load_trajectories(str(TRAIN_TRAJ))
    valid_trajs = load_trajectories(str(VALID_TRAJ))
    print(f"  Train successful trajectories: {len(train_trajs)}")
    print(f"  Valid successful trajectories: {len(valid_trajs)}")

    run_id = f"m2_2_build_{uuid.uuid4().hex[:8]}"
    print(f"\nRun ID: {run_id}")
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks" / "m2_1")

    with ProcurementBrowserEnv(max_steps=25, run_id=run_id, headless=True) as env:
        env.set_agent_name("dataset_builder")

        print("\nBuilding train samples...")
        train_samples, train_mm, train_turns = replay_and_build(train_trajs, "train", env)
        print(f"  Train samples: {len(train_samples)}, mismatches: {train_mm}, turns: {train_turns}")

        print("\nBuilding valid samples...")
        valid_samples, valid_mm, valid_turns = replay_and_build(valid_trajs, "valid", env)
        print(f"  Valid samples: {len(valid_samples)}, mismatches: {valid_mm}, turns: {valid_turns}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, samples in [("train", train_samples), ("valid", valid_samples)]:
        out_path = OUTPUT_DIR / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
        print(f"  Saved {split}: {out_path} ({len(samples)} samples)")

    # Manifest
    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "train_trajectories": len(train_trajs),
        "valid_trajectories": len(valid_trajs),
        "train_mismatches": train_mm,
        "valid_mismatches": valid_mm,
        "prompt_version": "browser_agent_v1",
        "format": "conversational",
        "message_field": "messages (list of chat turns)",
        "completion_field": "last message is assistant action JSON",
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest: {manifest_path}")

    total_time = time.time() - start_time
    print(f"\n=== Dataset Builder Complete ===")
    print(f"  Total samples: {len(train_samples) + len(valid_samples)}")
    print(f"  Runtime: {total_time:.1f}s")
    print(f"  Mismatches: {train_mm + valid_mm} (should be 0)")


if __name__ == "__main__":
    main()
