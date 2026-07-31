#!/usr/bin/env python3
"""Single-task evaluation (run as subprocess by phase2).

Usage:
    python scripts/m3_0a_eval_single_seed.py --seed 42 --task-id TASK-004 --output result.json
"""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.model_agent.agent_loop import run_model_episode
from miniwebwork.model_agent.model_backend import ModelConfig, QwenTransformersBackend
from miniwebwork.model_agent.output_parser import parse
from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
from miniwebwork.tasks import get_public_task

DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"


def _get_adapter_path(seed: int, output_dir: Path):
    if seed == 0:
        return DEFAULT_BASE_MODEL
    path = output_dir / f"seed_{seed}" / "final_adapter"
    if path.exists():
        return str(path)
    return ""


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
                if free_mb >= 20000:
                    return str(idx)
    except Exception:
        pass
    return "0"


def load_policy(base_model_path, adapter_path, max_new_tokens=128):
    device = _find_free_gpu()
    print(f"  Using GPU {device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    base_model.eval()

    if adapter_path and adapter_path != base_model_path:
        model = PeftModel.from_pretrained(
            base_model, adapter_path, torch_dtype=torch.bfloat16,
        )
    else:
        model = base_model
    model.eval()

    config = ModelConfig(
        model_path=base_model_path,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        dtype="bfloat16",
        device="cuda:0",
        enable_thinking=False,
    )
    backend = QwenTransformersBackend(config)
    backend._model = model
    backend._tokenizer = tokenizer
    backend._loaded = True

    import miniwebwork.model_agent.prompt_builder as pb
    pb.HISTORY_WINDOW = 5
    agent = QwenBrowserAgent(backend, pb, parse)
    return backend, agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-env-steps", type=int, default=15)
    args = parser.parse_args()

    adapter_dir = PROJECT_ROOT / "outputs" / "m2_2r"
    if not args.adapter_path:
        args.adapter_path = _get_adapter_path(args.seed, adapter_dir) or ""

    label = f"seed_{args.seed}" if args.seed else "base_canonical_v2"
    task = get_public_task(args.task_id)
    if task is None:
        result = {"task_id": args.task_id, "success": False,
                  "termination_reason": "unknown_task", "model_turns": 0,
                  "environment_steps": 0, "turns": []}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result))
        return

    print(f"Evaluating {label} on {args.task_id} (single-task subprocess)")

    backend, agent = load_policy(
        args.base_model, args.adapter_path, args.max_new_tokens,
    )

    run_id = f"no_sol_{args.seed}_{args.task_id}_{uuid.uuid4().hex[:8]}"
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks")

    try:
        with ProcurementBrowserEnv(max_steps=args.max_env_steps, run_id=run_id) as env:
            env.set_agent_name(f"no_sol_{label}")
            print(f"  Running episode...", end=" ", flush=True)
            result = run_model_episode(
                args.task_id, env, agent, max_model_turns=20, max_env_steps=args.max_env_steps
            )
            status = "PASS" if result.get("success") else "FAIL"
            reason = result.get("termination_reason", "?")[:30]
            turns = result.get("model_turns", 0)
            print(f"{status} turns={turns} reason={reason}")
    except Exception as e:
        print(f"  ENV ERROR: {e}")
        result = {
            "task_id": args.task_id, "success": False,
            "termination_reason": "model_error",
            "error": str(e)[:200],
            "model_turns": 0, "environment_steps": 0, "turns": [],
        }

    output = {
        "task_id": args.task_id,
        "seed": args.seed,
        "label": label,
        "success": result.get("success", False),
        "reward": result.get("reward", 0.0),
        "termination_reason": result.get("termination_reason", ""),
        "failure_reasons": result.get("failure_reasons", []),
        "model_turns": result.get("model_turns", 0),
        "environment_steps": result.get("environment_steps", 0),
        "verifier_success": result.get("verifier_success", False),
        "elapsed_s": result.get("elapsed_s", 0),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  Saved: {args.output}")

    # Cleanup
    del backend, agent
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
