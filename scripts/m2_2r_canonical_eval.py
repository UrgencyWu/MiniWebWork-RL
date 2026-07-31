"""M2.2R Canonical Evaluation: Base + SFT with Canonical Prompt Contract v2.

Evaluates:
- Base model with v2 prompt
- SFT Seed 42 with v2 prompt
- SFT Seed 1234 with v2 prompt
- SFT Seed 20260726 with v2 prompt

Both teacher-forced (on valid set) and frozen E2E (15 tasks).
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.model_agent.prompt_builder import build_messages, load_system_prompt, compute_message_hash, MAX_VISIBLE_TEXT, HISTORY_WINDOW, MAX_ELEMENTS
from miniwebwork.model_agent.output_parser import parse
from miniwebwork.model_agent.agent_loop import run_model_episode
from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
from miniwebwork.model_agent.model_backend import QwenTransformersBackend, ModelConfig
from miniwebwork.agent_env.schemas import AgentAction
from miniwebwork.tasks import load_public_tasks

DEFAULT_BASE_MODEL = "/data/share/model/Qwen3.5-4B"


def teacher_forced_eval_canonical(model, tokenizer, dataset, max_length=8192, split="valid"):
    """Teacher-forced eval using canonical prompt contract."""
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

            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs,
            )

            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs.input_ids.to(model.device)
            input_len = input_ids.shape[1]

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
            try:
                pred_parsed = parse(predicted)
            except Exception as e:
                errors.append(f"Sample {i}: parse error: {e}")
                continue

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
    if errors:
        print(f"  Errors ({len(errors)}): {errors[:5]}")

    return result


def frozen_test_eval_canonical(model, tokenizer, base_model_path, adapter_path,
                                max_new_tokens=128, max_model_turns=20, max_env_steps=15):
    """Frozen E2E eval using canonical prompt contract."""
    print(f"  Loading base model with adapter: {adapter_path}")

    config = ModelConfig(
        model_path=base_model_path,
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
        dtype="bfloat16",
        device="cuda:0",
    )
    backend = QwenTransformersBackend(config)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    base_model = base_model.to("cuda:0")

    # Only wrap with PeftModel if adapter_path differs from base model path
    if adapter_path != base_model_path:
        model_w = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model_w = base_model
    model_w.eval()

    backend._model = model_w
    backend._tokenizer = tokenizer
    backend._loaded = True

    import miniwebwork.model_agent.prompt_builder as pb
    pb.HISTORY_WINDOW = 5

    agent = QwenBrowserAgent(backend, pb, parse)

    tasks = load_public_tasks()
    frozen_path = PROJECT_ROOT / "data" / "splits" / "m2_1" / "test_frozen.json"
    if frozen_path.exists():
        frozen_data = json.loads(frozen_path.read_text())
        frozen_ids = {t["task_id"] for t in frozen_data["test_frozen"]}
        frozen_tasks = [t for t in tasks if t["task_id"] in frozen_ids]
    else:
        frozen_tasks = tasks[:15]

    print(f"  Frozen test tasks: {len(frozen_tasks)}")

    run_id = f"m2_2r_frozen_{uuid.uuid4().hex[:8]}"
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks" / "m2_1")

    results = []
    start_time = time.time()

    with ProcurementBrowserEnv(max_steps=max_env_steps, run_id=run_id) as env:
        env.set_agent_name("m2_2r_sft")

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

    # Compute metrics — handle None action_result safely
    successful = sum(1 for r in results if r.get("success"))
    total_tasks = len(results)
    avg_turns = sum(r.get("model_turns", 0) for r in results) / max(total_tasks, 1)

    all_turns = []
    for r in results:
        all_turns.extend(r.get("turns", []))

    total_actions = len(all_turns)
    strict_json = sum(1 for t in all_turns if t.get("strict_json_success"))
    schema_valid = sum(1 for t in all_turns if t.get("schema_valid"))
    env_success = sum(1 for t in all_turns
                      if t.get("action_result") and t.get("action_result", {}).get("success"))
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


def evaluate_model(name, base_model_path, adapter_path, data_dir, output_dir, tokenizer, max_length=8192):
    """Evaluate a single model (teacher-forced + frozen E2E)."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name} (max_length={max_length})")
    print(f"{'='*60}")

    results = {}

    # Load model with adapter (once — reused for both eval phases)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    base_model = base_model.to("cuda:0")

    if adapter_path == base_model_path:
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Teacher-forced eval
    valid_path = data_dir / "valid.jsonl"
    if valid_path.exists():
        print("\n--- Teacher-forced eval ---")
        valid_ds = load_dataset("json", data_files={"valid": str(valid_path)}, split="valid")
        tf_metrics = teacher_forced_eval_canonical(model, tokenizer, valid_ds, max_length=max_length)
        results["teacher_forced"] = tf_metrics

    # Frozen E2E eval (reuse same model — no reload needed)
    print("\n--- Frozen E2E eval ---")
    frozen_metrics = frozen_test_eval_canonical(
        model=model,
        tokenizer=tokenizer,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )
    results["frozen_test"] = frozen_metrics

    # Cleanup
    del model
    if adapter_path != base_model_path:
        del base_model
    torch.cuda.empty_cache()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=Path("data/sft/m2_2r"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m2_2r"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 20260726])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-turns", type=int, default=20)
    parser.add_argument("--max-env-steps", type=int, default=15)
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Max input token length for teacher-forced eval")
    args = parser.parse_args()

    print("=== M2.2R Canonical Evaluation ===")
    print(f"Base model: {args.base_model}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (shared)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # Evaluate base model
    print("\n" + "="*60)
    print("BASE MODEL EVALUATION")
    print("="*60)
    base_results = evaluate_model(
        name="base_canonical_v2",
        base_model_path=args.base_model,
        adapter_path=args.base_model,
        data_dir=args.data_dir,
        output_dir=output_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    all_results["base_canonical_v2"] = base_results

    # Evaluate each SFT seed
    for seed in args.seeds:
        adapter_path = args.output_dir / f"seed_{seed}" / "final_adapter"
        if not adapter_path.exists():
            print(f"SKIP: adapter not found at {adapter_path}")
            continue

        seed_results = evaluate_model(
            name=f"sft_seed_{seed}",
            base_model_path=args.base_model,
            adapter_path=str(adapter_path),
            data_dir=args.data_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        all_results[f"sft_seed_{seed}"] = seed_results

    # Save results
    results_path = output_dir / "canonical_eval_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    print(f"\n\nResults saved: {results_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"CANONICAL EVAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Success':>8} {'Schema%':>8} {'EnvAct%':>8} {'OutFail':>8}")
    print(f"{'-'*60}")

    for name, r in all_results.items():
        frozen = r.get("frozen_test", {})
        success = frozen.get("success_rate", 0)
        schema = frozen.get("schema_valid_rate", 0)
        env_act = frozen.get("env_action_success_rate", 0)
        out_fail = frozen.get("total_tasks", 0) - frozen.get("successful_tasks", 0)
        print(f"{name:<25} {success:>7.1%} {schema:>7.1%} {env_act:>7.1%} {out_fail:>7d}")


if __name__ == "__main__":
    main()
