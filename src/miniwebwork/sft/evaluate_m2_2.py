"""M2.2 Evaluation: teacher-forced valid set evaluation + frozen test end-to-end."""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.model_agent.prompt_builder import build_messages, load_system_prompt
from miniwebwork.model_agent.output_parser import parse
from miniwebwork.model_agent.agent_loop import run_model_episode
from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
from miniwebwork.model_agent.model_backend import QwenTransformersBackend, ModelConfig
from miniwebwork.agent_env.schemas import AgentAction
from miniwebwork.tasks import load_public_tasks

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"


def teacher_forced_eval(model, tokenizer, dataset, max_length: int, split: str = "valid") -> dict:
    """Evaluate model on conversational dataset using teacher-forced generation."""
    print(f"\n=== Teacher-forced evaluation on {split} set ===")
    print(f"  Samples: {len(dataset)}")

    exact = 0
    action_type = 0
    schema_valid = 0
    parse_fallback = 0
    total = 0
    errors = []

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            messages = example["messages"]
            expected_completion = messages[-1]["content"]
            chat_kwargs = example.get("chat_template_kwargs", {})

            try:
                expected_action = json.loads(expected_completion)
            except json.JSONDecodeError:
                errors.append(f"Sample {i}: invalid completion JSON")
                continue

            # Apply chat template to messages[:-1] (exclude assistant turn)
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs,
            )

            # Tokenize
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs.input_ids.to(model.device)
            input_len = input_ids.shape[1]

            # Expected token count
            expected_ids = tokenizer(expected_completion, add_special_tokens=False)
            max_new = max(len(expected_ids["input_ids"]) + 10, 32)

            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                new_ids = outputs[0][input_len:]
                predicted = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            except Exception as e:
                predicted = ""
                errors.append(f"Sample {i}: generation error: {e}")

            total += 1
            pred_parsed = parse(predicted)
            pred_action = pred_parsed.parsed_payload or {}

            exp_str = json.dumps(expected_action, sort_keys=True)
            pred_str = json.dumps(pred_action, sort_keys=True)

            if pred_str == exp_str:
                exact += 1
            if pred_action.get("action") == expected_action.get("action"):
                action_type += 1
            if pred_parsed.schema_valid:
                schema_valid += 1
            if pred_parsed.fallback_used:
                parse_fallback += 1

    model.train()

    result = {
        "split": split,
        "total": total,
        "exact_match": exact / max(total, 1),
        "action_type_match": action_type / max(total, 1),
        "schema_valid_rate": schema_valid / max(total, 1),
        "fallback_rate": parse_fallback / max(total, 1),
        "errors": errors[:10],
    }

    print(f"  Exact match: {result['exact_match']:.1%} ({exact}/{total})")
    print(f"  Action type match: {result['action_type_match']:.1%} ({action_type}/{total})")
    print(f"  Schema valid: {result['schema_valid_rate']:.1%} ({schema_valid}/{total})")
    print(f"  Fallback parse: {result['fallback_rate']:.1%} ({parse_fallback}/{total})")

    return result


def frozen_test_eval(
    model, tokenizer, base_model_path: str, adapter_path: str,
    max_new_tokens: int = 128, max_model_turns: int = 20, max_env_steps: int = 15,
) -> dict:
    """Run full end-to-end evaluation on frozen 15-task test set."""
    print("\n=== Frozen Test Evaluation ===")

    print(f"  Loading base model with adapter: {adapter_path}")
    config = ModelConfig(
        model_path=base_model_path,
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
        dtype="bfloat16",
        device="cuda:0",
    )
    backend = QwenTransformersBackend(config)

    # Load model with LoRA adapter (CPU first to avoid OOM)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    base_model = base_model.to("cuda:0")
    model_w = PeftModel.from_pretrained(base_model, adapter_path)
    model_w.eval()

    backend._model = model_w
    backend._loaded = True

    # Setup agent
    import miniwebwork.model_agent.prompt_builder as pb
    pb.HISTORY_WINDOW = 5

    agent = QwenBrowserAgent(backend, pb, parse)

    # Load frozen test tasks
    tasks = load_public_tasks()
    frozen_path = PROJECT_ROOT / "data" / "splits" / "m2_1" / "test_frozen.json"
    if frozen_path.exists():
        frozen_data = json.loads(frozen_path.read_text())
        frozen_ids = {t["task_id"] for t in frozen_data["test_frozen"]}
        frozen_tasks = [t for t in tasks if t["task_id"] in frozen_ids]
    else:
        frozen_tasks = tasks[:15]

    print(f"  Frozen test tasks: {len(frozen_tasks)}")

    # Setup environment
    run_id = f"m2_2_frozen_{uuid.uuid4().hex[:8]}"
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks" / "m2_1")

    results = []
    start_time = time.time()

    with ProcurementBrowserEnv(max_steps=max_env_steps, run_id=run_id) as env:
        env.set_agent_name("m2_2_sft")

        for i, task in enumerate(frozen_tasks):
            task_id = task["task_id"]
            print(f"  [{i+1}/{len(frozen_tasks)}] {task_id}...", end=" ", flush=True)

            try:
                result = run_model_episode(
                    task_id, env, agent, max_model_turns, max_env_steps
                )
            except Exception as e:
                result = {
                    "task_id": task_id, "success": False,
                    "termination_reason": "model_error",
                    "error": str(e)[:200],
                    "model_turns": 0, "environment_steps": 0, "turns": [],
                }

            status = "PASS" if result.get("success") else "FAIL"
            turns = result.get("model_turns", 0)
            reason = result.get("termination_reason", "?")[:30]
            print(f"{status} turns={turns} reason={reason}")

            results.append(result)

    elapsed = time.time() - start_time

    # Compute metrics
    successful = sum(1 for r in results if r.get("success"))
    total_tasks = len(results)
    avg_turns = sum(r.get("model_turns", 0) for r in results) / max(total_tasks, 1)

    all_turns = []
    for r in results:
        all_turns.extend(r.get("turns", []))

    total_actions = len(all_turns)
    strict_json = sum(1 for t in all_turns if t.get("strict_json_success"))
    schema_valid = sum(1 for t in all_turns if t.get("schema_valid"))
    env_success = sum(1 for t in all_turns if t.get("action_result") and t.get("action_result", {}).get("success"))
    invalid_actions = sum(1 for t in all_turns if not t.get("schema_valid"))

    action_dist = {}
    for t in all_turns:
        if t.get("action"):
            a = t["action"].get("action", "unknown")
            action_dist[a] = action_dist.get(a, 0) + 1

    metrics = {
        "adapter_path": adapter_path,
        "total_tasks": total_tasks,
        "successful_tasks": successful,
        "success_rate": successful / max(total_tasks, 1),
        "avg_model_turns": avg_turns,
        "total_generations": total_actions,
        "strict_json_rate": strict_json / max(total_actions, 1),
        "schema_valid_rate": schema_valid / max(total_actions, 1),
        "env_action_success_rate": env_success / max(total_actions, 1),
        "invalid_actions": invalid_actions,
        "action_distribution": action_dist,
        "runtime_s": elapsed,
        "per_task": [
            {
                "task_id": r["task_id"],
                "success": r.get("success", False),
                "model_turns": r.get("model_turns", 0),
                "env_steps": r.get("environment_steps", 0),
                "termination_reason": r.get("termination_reason", ""),
                "failure_reasons": r.get("failure_reasons", []),
            }
            for r in results
        ],
    }

    print(f"\n  Success: {successful}/{total_tasks} ({metrics['success_rate']:.1%})")
    print(f"  Avg turns: {avg_turns:.1f}")
    print(f"  Strict JSON: {metrics['strict_json_rate']:.1%}")
    print(f"  Schema valid: {metrics['schema_valid_rate']:.1%}")
    print(f"  Env action success: {metrics['env_action_success_rate']:.1%}")
    print(f"  Invalid actions: {invalid_actions}")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"  Action dist: {json.dumps(action_dist)}")

    return metrics


def evaluate_base(base_model_path: str, max_new_tokens: int = 128) -> dict:
    """Evaluate base model on frozen test."""
    print("\n=== Base Model Evaluation ===")
    return frozen_test_eval(
        model=None, tokenizer=None,
        base_model_path=base_model_path,
        adapter_path=base_model_path,
        max_new_tokens=max_new_tokens,
    )


def main():
    parser = argparse.ArgumentParser(description="M2.2 Evaluation")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "sft" / "m2_2"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "m2_2"))
    parser.add_argument("--mode", choices=["teacher_forced", "frozen_test", "both"], default="both")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-turns", type=int, default=20)
    parser.add_argument("--max-env-steps", type=int, default=15)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading datasets...")
    data_dir = Path(args.data_dir)
    valid_ds = None
    if (data_dir / "valid.jsonl").exists():
        valid_ds = load_dataset("json", data_files={"valid": str(data_dir / "valid.jsonl")}, split="valid")

    results = {}

    if args.mode in ("teacher_forced", "both") and valid_ds:
        print("Loading model with LoRA adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
        )
        base_model = base_model.to("cuda:0")
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model.eval()

        tf_metrics = teacher_forced_eval(model, tokenizer, valid_ds, max_length=526)
        results["teacher_forced"] = tf_metrics

        del model
        del base_model
        torch.cuda.empty_cache()

    if args.mode in ("frozen_test", "both"):
        frozen_metrics = frozen_test_eval(
            model=None, tokenizer=tokenizer,
            base_model_path=args.base_model,
            adapter_path=str(adapter_dir),
            max_new_tokens=args.max_new_tokens,
            max_model_turns=args.max_model_turns,
            max_env_steps=args.max_env_steps,
        )
        results["frozen_test"] = frozen_metrics

    # Save
    seed = adapter_dir.name if "seed_" in adapter_dir.name else "unknown"
    result_path = output_dir / f"eval_{seed}.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults saved: {result_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"
