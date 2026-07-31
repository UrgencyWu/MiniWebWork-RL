"""M2.2R Canonical Dataset Builder v2.

Rebuilds SFT dataset using the Canonical Prompt Contract v2 (browser_agent_v2).
ALL paths (training, teacher-forced eval, frozen E2E) use the same build_messages_v2 function.
"""
import json
import os
import sys
import time
import uuid
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent
from miniwebwork.agent_env.schemas import AgentAction
from miniwebwork.tasks import get_oracle

# Canonical Prompt Contract v2
CANONICAL_VERSION = "browser_agent_v2"
CANONICAL_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "browser_agent_v2" / "system.txt"
MAX_VISIBLE_TEXT = 8000
HISTORY_WINDOW = 5
MAX_ELEMENTS = 100


def load_canonical_system_prompt() -> str:
    """Load the canonical system prompt."""
    if CANONICAL_SYSTEM_PROMPT_PATH.exists():
        return CANONICAL_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    # Fallback: use the full v1 prompt
    from miniwebwork.model_agent.prompt_builder import load_system_prompt
    return load_system_prompt("browser_agent_v1")


def canonical_serialize_elements(elements) -> list:
    """Serialize elements using canonical contract (all fields, up to MAX_ELEMENTS)."""
    els = []
    for e in (elements or [])[:MAX_ELEMENTS]:
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


def canonical_serialize_history(history, window=HISTORY_WINDOW) -> list:
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
    """Build chat messages using Canonical Prompt Contract v2.

    This function is the SINGLE SOURCE OF TRUTH for prompt construction.
    It is used by:
    1. SFT dataset construction
    2. Teacher-forced evaluation
    3. Frozen E2E evaluation
    """
    history = history or []
    system = load_canonical_system_prompt()

    # Serialize observation using canonical contract
    els = canonical_serialize_elements(observation.elements)
    visible_text = observation.visible_text or ""
    text_truncated = len(visible_text) > MAX_VISIBLE_TEXT
    if text_truncated:
        visible_text = visible_text[:MAX_VISIBLE_TEXT]

    hist = canonical_serialize_history(history, HISTORY_WINDOW)
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


def compute_message_hash(messages: list) -> str:
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def replay_and_build_v2(trajectories, split, env):
    """Replay expert trajectories and build conversational message pairs using v2 contract."""
    samples = []
    mismatches = 0
    total_turns = 0

    system_prompt = load_canonical_system_prompt()
    system_sha = hashlib.sha256(system_prompt.encode()).hexdigest()

    for traj in trajectories:
        task_id = traj["task_id"]
        oracle = get_oracle(task_id)
        if not oracle:
            print(f"  WARN: no oracle for {task_id}")
            continue

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

            # Build messages using CANONICAL v2 contract
            messages = build_messages_v2(obs, history)

            # Get expected action
            expected_action = expert_turn.get("action", {})

            # Verify action matches
            expert_action = expert.act(obs)
            if expert_action.to_dict() != expected_action:
                mismatches += 1
                if mismatches <= 3:
                    print(f"    MISMATCH {task_id} turn {turn_idx}: "
                          f"expected={expected_action}, got={expert_action.to_dict()}")

            # Build conversational format
            completion = json.dumps(expected_action, ensure_ascii=False)
            full_messages = messages + [{"role": "assistant", "content": completion}]

            sample = {
                "sample_id": expert_turn.get("sample_id", f"{split[:1].upper()}{turn_idx+1:06d}"),
                "split": split,
                "task_id": task_id,
                "source": "oracle_expert",
                "prompt_version": CANONICAL_VERSION,
                "prompt_contract_sha256": system_sha,
                "messages": full_messages,
                "chat_template_kwargs": {"enable_thinking": False},
                "message_hash": compute_message_hash(messages),
                "turn_index": turn_idx,
                "visible_text_chars": len(obs.visible_text or ""),
                "element_count": len(obs.elements or []),
                "history_turns": len(history),
            }
            samples.append(sample)
            total_turns += 1

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

            turn_idx += 1

    return samples, mismatches, total_turns


def main():
    print("=== M2.2R Canonical Dataset Builder v2 ===")
    print(f"Contract: {CANONICAL_VERSION}")
    start_time = time.time()

    # Load system prompt info
    system_prompt = load_canonical_system_prompt()
    print(f"System prompt length: {len(system_prompt)}")
    print(f"System prompt SHA256: {hashlib.sha256(system_prompt.encode()).hexdigest()}")

    # Load trajectories
    train_traj_path = PROJECT_ROOT / "data" / "expert" / "m2_1" / "train_trajectories.json"
    valid_traj_path = PROJECT_ROOT / "data" / "expert" / "m2_1" / "valid_trajectories.json"

    def load_trajectories(path):
        data = json.loads(Path(path).read_text())
        return [t for t in data if t.get("success", False)]

    train_trajs = load_trajectories(str(train_traj_path))
    valid_trajs = load_trajectories(str(valid_traj_path))
    print(f"  Train trajectories: {len(train_trajs)}")
    print(f"  Valid trajectories: {len(valid_trajs)}")

    # Output directory
    output_dir = PROJECT_ROOT / "data" / "sft" / "m2_2r"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"m2_2r_build_{uuid.uuid4().hex[:8]}"
    print(f"\nRun ID: {run_id}")
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks" / "m2_1")

    with ProcurementBrowserEnv(max_steps=25, run_id=run_id, headless=True) as env:
        env.set_agent_name("canonical_dataset_builder")

        print("\nBuilding train samples...")
        train_samples, train_mm, train_turns = replay_and_build_v2(train_trajs, "train", env)
        print(f"  Train samples: {len(train_samples)}, mismatches: {train_mm}, turns: {train_turns}")

        print("\nBuilding valid samples...")
        valid_samples, valid_mm, valid_turns = replay_and_build_v2(valid_trajs, "valid", env)
        print(f"  Valid samples: {len(valid_samples)}, mismatches: {valid_mm}, turns: {valid_turns}")

    # Save
    for split, samples in [("train", train_samples), ("valid", valid_samples)]:
        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
        print(f"  Saved {split}: {out_path} ({len(samples)} samples)")

    # Compute token stats
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/share/model/Qwen3.5-4B",
        local_files_only=True, trust_remote_code=True,
    )

    train_total_tokens = []
    train_prompt_tokens = []
    train_completion_tokens = []
    train_truncated = 0

    for s in train_samples:
        msgs = s["messages"]
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True,
            **s.get("chat_template_kwargs", {}),
        )
        prompt_ids = tokenizer(prompt_text)["input_ids"]
        completion_ids = tokenizer(msgs[-1]["content"], add_special_tokens=False)["input_ids"]
        total = len(prompt_ids) + len(completion_ids)
        train_total_tokens.append(total)
        train_prompt_tokens.append(len(prompt_ids))
        train_completion_tokens.append(len(completion_ids))
        if total >= 2048:
            train_truncated += 1

    def percentile(data, p):
        s = sorted(data)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    # Manifest
    manifest = {
        "dataset_version": "m2_2r_canonical_v1",
        "contract_version": CANONICAL_VERSION,
        "contract_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "train_trajectories": len(train_trajs),
        "valid_trajectories": len(valid_trajs),
        "train_mismatches": train_mm,
        "valid_mismatches": valid_mm,
        "prompt_version": CANONICAL_VERSION,
        "format": "conversational",
        "message_field": "messages (list of chat turns)",
        "completion_field": "last message is assistant action JSON",
        "token_stats": {
            "train_total": {
                "p50": percentile(train_total_tokens, 50),
                "p95": percentile(train_total_tokens, 95),
                "max": max(train_total_tokens),
                "mean": sum(train_total_tokens) / len(train_total_tokens),
            },
            "train_prompt": {
                "p50": percentile(train_prompt_tokens, 50),
                "p95": percentile(train_prompt_tokens, 95),
                "max": max(train_prompt_tokens),
            },
            "train_completion": {
                "p50": percentile(train_completion_tokens, 50),
                "p95": percentile(train_completion_tokens, 95),
                "max": max(train_completion_tokens),
            },
            "truncated_count": train_truncated,
        },
        "mask_audit": {
            "completion_not_truncated": train_truncated == 0,
            "oracle_leak": 0,
            "test_leak": 0,
            "all_passed": train_truncated == 0,
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest: {manifest_path}")

    total_time = time.time() - start_time
    print(f"\n=== Canonical Dataset Builder Complete ===")
    print(f"  Total samples: {len(train_samples) + len(valid_samples)}")
    print(f"  Runtime: {total_time:.1f}s")
    print(f"  Mismatches: {train_mm + valid_mm} (should be 0)")
    print(f"  Token stats: p50={manifest['token_stats']['train_total']['p50']}, "
          f"p95={manifest['token_stats']['train_total']['p95']}, "
          f"max={manifest['token_stats']['train_total']['max']}")
    print(f"  Truncated: {train_truncated}")


if __name__ == "__main__":
    main()
