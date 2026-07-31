"""M2.2R: 2x2 Prompt Matrix - diagnose training-inference contract drift.

Matrix:
                Full Prompt    Compact Prompt
  Base Model       A               B
  SFT Seed 42      C               D

Pilot on 3 tasks: exact_product, cheapest_feasible, no_feasible_product
"""
import json
import sys
import os
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
PROJECT_ROOT = Path("/home/wushaohua/data/MiniWebWork-RL")

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.model_agent.prompt_builder import (
    build_messages, load_system_prompt, compute_message_hash,
    MAX_VISIBLE_TEXT, HISTORY_WINDOW, MAX_ELEMENTS,
)
from miniwebwork.model_agent.output_parser import parse
from miniwebwork.model_agent.model_backend import QwenTransformersBackend, ModelConfig
from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
from miniwebwork.agent_env.schemas import AgentAction
from miniwebwork.tasks import load_public_tasks, get_oracle


# Compact prompt (same as browser_agent_v1_compact.txt)
COMPACT_SYSTEM_PROMPT = """You are a web browser agent. Output exactly one JSON action per turn.

## ACTIONS
click: {"action":"click","target":"<element_id>"}
fill: {"action":"fill","target":"<element_id>","value":"<text>"}
select: {"action":"select","target":"<element_id>","value":"<option_value>"}
check: {"action":"check","target":"<element_id>","checked":true}
back: {"action":"back"}
submit: {"action":"submit","target":"<element_id>"}
finish: {"action":"finish"}

## RULES
- Output ONLY a single JSON object. No markdown, no explanation.
- Only use element_id from the elements list.
- Do NOT invent element IDs. If no suitable element, use finish."""

COMPACT_SHA256 = hashlib.sha256(COMPACT_SYSTEM_PROMPT.encode()).hexdigest()

FULL_SYSTEM_PROMPT = load_system_prompt("browser_agent_v1")
FULL_sha256 = hashlib.sha256(FULL_SYSTEM_PROMPT.encode()).hexdigest()


def build_compact_messages(observation, history=None):
    """Build messages with compact prompt (matching SFT training data format)."""
    history = history or []
    els = []
    for e in (observation.elements or [])[:MAX_ELEMENTS]:
        d = {"element_id": e.element_id, "role": e.role, "name": e.name,
             "testid": e.testid if e.testid else None, "disabled": e.disabled}
        if e.tag in ("input", "textarea"):
            d["value"] = e.value[:100] if e.value else ""
        if e.tag == "select" and e.options:
            d["options"] = e.options[:10]
        if e.role in ("link", "button"):
            d["text"] = e.text[:100] if e.text else ""
        els.append(d)

    visible_text = observation.visible_text or ""
    text_truncated = len(visible_text) > MAX_VISIBLE_TEXT
    if text_truncated:
        visible_text = visible_text[:MAX_VISIBLE_TEXT]

    hist = []
    for h in history[-HISTORY_WINDOW:]:
        hist.append({
            "turn": h.get("model_turn_index", 0),
            "action": h.get("action"),
            "parse_ok": h.get("parse_ok", False),
            "result": h.get("result", ""),
            "page_type": h.get("page_type", ""),
        })

    last_result = observation.last_action_result

    # Compact format: NO task/page sections, same element fields as SFT data
    user_content = f"""## Visible Text (truncated={text_truncated})
{visible_text}

## Interactive Elements ({len(els)})
{json.dumps(els, ensure_ascii=False)}

## Last Action Result
{json.dumps(last_result, ensure_ascii=False) if last_result else 'N/A'}

## Instruction
Output exactly one JSON action. Only use element_id from the elements list above."""

    return [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class CustomPromptAgent:
    """Agent that uses a specific prompt builder (compact or full)."""
    def __init__(self, backend, prompt_builder_mode="full"):
        self._backend = backend
        self._mode = prompt_builder_mode
        self._history = []
        self._model_turn = 0

    def reset(self, task_id, instruction):
        self._history = []
        self._model_turn = 0

    def act(self, observation):
        self._model_turn += 1
        attempt = {
            "model_turn_index": self._model_turn,
            "raw_output": "",
            "strict_json_success": False,
            "fallback_used": False,
            "parsed_payload": None,
            "schema_valid": False,
            "action": None,
            "errors": [],
            "prompt_hash": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
        }

        if self._mode == "compact":
            messages = build_compact_messages(observation, self._history)
        else:
            messages = build_messages(observation, self._history)

        attempt["prompt_hash"] = compute_message_hash(messages)

        gen = self._backend.generate(messages)
        attempt["raw_output"] = gen.raw_text
        attempt["input_tokens"] = gen.input_tokens
        attempt["output_tokens"] = gen.new_tokens
        attempt["latency_ms"] = gen.latency_ms

        if gen.error:
            attempt["errors"].append(f"generation_error: {gen.error}")
            return attempt

        parsed = parse(gen.raw_text)
        attempt["strict_json_success"] = parsed.strict_json_success
        attempt["fallback_used"] = parsed.fallback_used
        attempt["parsed_payload"] = parsed.parsed_payload
        attempt["schema_valid"] = parsed.schema_valid
        attempt["errors"].extend(parsed.errors)

        if parsed.schema_valid and parsed.parsed_payload:
            try:
                attempt["action"] = AgentAction.from_dict(parsed.parsed_payload)
            except Exception as e:
                attempt["errors"].append(f"action_construction_error: {e}")

        return attempt

    def record_feedback(self, attempt, action_result, page_type):
        self._history.append({
            "model_turn_index": attempt["model_turn_index"],
            "action": attempt["action"].to_dict() if attempt["action"] else None,
            "parse_ok": attempt["schema_valid"],
            "result": action_result.to_dict() if action_result else {"success": False},
            "page_type": page_type,
        })

    @property
    def model_turn(self):
        return self._model_turn


def run_single_turn(task_id, backend, prompt_mode, adapter_path=None):
    """Run a single turn for a task with a specific prompt mode and adapter."""
    os.environ["MINIWEBWORK_TASK_DIR"] = str(PROJECT_ROOT / "data" / "tasks" / "m2_1")

    agent = CustomPromptAgent(backend, prompt_mode)

    with ProcurementBrowserEnv(max_steps=15, run_id=f"matrix_{prompt_mode}_{adapter_path.name if adapter_path else 'base'}", headless=True) as env:
        obs = env.reset(task_id)
        agent.reset(task_id, obs.instruction)

        attempt = agent.act(obs)
        parsed = parse(attempt["raw_output"])

        return {
            "task_id": task_id,
            "prompt_mode": prompt_mode,
            "adapter_path": str(adapter_path) if adapter_path else "base",
            "adapter_sha256": str(adapter_path.parent.name) if adapter_path else "N/A",
            "raw_output": attempt["raw_output"],
            "raw_output_length": len(attempt["raw_output"]),
            "output_empty": len(attempt["raw_output"]) == 0,
            "input_tokens": attempt["input_tokens"],
            "output_tokens": attempt["output_tokens"],
            "strict_json_success": parsed.strict_json_success,
            "fallback_used": parsed.fallback_used,
            "schema_valid": parsed.schema_valid,
            "parsed_payload": parsed.parsed_payload,
            "errors": parsed.errors,
            "prompt_hash": attempt["prompt_hash"],
            "latency_ms": attempt["latency_ms"],
        }


def get_pilot_tasks():
    """Get 3 representative tasks for pilot testing."""
    tasks = load_public_tasks()  # returns a list of task dicts
    # Find one of each type
    exact_product = None
    cheapest_feasible = None
    no_feasible = None

    for t in tasks:
        oracle = get_oracle(t["task_id"])
        if oracle:
            tt = oracle.get("task_type", "")
            if tt == "exact_product" and not exact_product:
                exact_product = t["task_id"]
            elif tt == "cheapest_feasible" and not cheapest_feasible:
                cheapest_feasible = t["task_id"]
            elif tt == "no_feasible_product" and not no_feasible:
                no_feasible = t["task_id"]

    result = [t for t in [exact_product, cheapest_feasible, no_feasible] if t]
    # Fallback to first 3 if not found
    if len(result) < 3:
        all_ids = [t["task_id"] for t in tasks[:5]]
        for tid in all_ids:
            if tid not in result:
                result.append(tid)
            if len(result) >= 3:
                break

    return result[:3]


def main():
    base_model = "/data/share/model/Qwen3.5-4B"
    adapters = {
        "base": None,
        "seed_42": PROJECT_ROOT / "outputs" / "m2_2" / "seed_42" / "final_adapter",
        "seed_1234": PROJECT_ROOT / "outputs" / "m2_2" / "seed_1234" / "final_adapter",
        "seed_20260726": PROJECT_ROOT / "outputs" / "m2_2" / "seed_20260726" / "final_adapter",
    }
    prompt_modes = ["full", "compact"]

    pilot_tasks = get_pilot_tasks()
    print(f"Pilot tasks: {pilot_tasks}")
    print(f"Prompt modes: {prompt_modes}")
    print(f"Models: {list(adapters.keys())}")

    # Run matrix
    results = []

    for model_name, adapter_path in adapters.items():
        if adapter_path and not adapter_path.exists():
            print(f"  SKIP {model_name}: adapter not found")
            continue

        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"  Adapter: {adapter_path}")

        # Load model
        config = ModelConfig(
            model_path=base_model,
            max_new_tokens=128,
            enable_thinking=False,
            dtype="bfloat16",
            device="cuda:0",
        )
        backend = QwenTransformersBackend(config)

        if adapter_path:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch

            tokenizer = AutoTokenizer.from_pretrained(
                base_model, local_files_only=True, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model_loaded = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16,
                local_files_only=True, trust_remote_code=True,
            )
            base_model_loaded = base_model_loaded.to("cuda:0")
            model_w = PeftModel.from_pretrained(base_model_loaded, str(adapter_path))
            model_w.eval()

            backend._model = model_w
            backend._loaded = True
            backend._tokenizer = tokenizer
        else:
            backend.load()

        model_info = backend.get_model_info()
        print(f"  Model loaded: {model_info['model_type']}")
        print(f"  Chat template SHA256: {model_info['chat_template_sha256']}")

        for prompt_mode in prompt_modes:
            print(f"\n  --- Prompt mode: {prompt_mode} ---")
            for task_id in pilot_tasks:
                print(f"    {task_id}...", end=" ", flush=True)
                try:
                    result = run_single_turn(task_id, backend, prompt_mode, adapter_path)
                    results.append(result)
                    status = "OK" if result["schema_valid"] else "FAIL"
                    print(f"{status} raw_len={result['raw_output_length']} "
                          f"json={result['strict_json_success']} "
                          f"schema={result['schema_valid']}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    results.append({
                        "task_id": task_id, "prompt_mode": prompt_mode,
                        "adapter_path": str(adapter_path) if adapter_path else "base",
                        "error": str(e)[:200],
                        "schema_valid": False,
                    })

        # Cleanup
        del backend._model
        import torch
        torch.cuda.empty_cache()

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": base_model,
        "compact_prompt_sha256": COMPACT_SHA256,
        "full_prompt_sha256": FULL_sha256,
        "pilot_tasks": pilot_tasks,
        "matrix": results,
    }

    output_path = PROJECT_ROOT / "artifacts" / "m2_2r" / "prompt_matrix_pilot.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n\nResults saved: {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"2x2 PROMPT MATRIX PILOT RESULTS")
    print(f"{'='*70}")

    for model_name in ["base", "seed_42", "seed_1234", "seed_20260726"]:
        model_results = [r for r in results if r.get("adapter_path", "").endswith(model_name) or (model_name == "base" and r.get("adapter_path") == "base")]
        if not model_results:
            continue
        print(f"\n{model_name}:")
        for pm in ["full", "compact"]:
            pm_results = [r for r in model_results if r.get("prompt_mode") == pm]
            if not pm_results:
                continue
            valid = sum(1 for r in pm_results if r.get("schema_valid"))
            nonempty = sum(1 for r in pm_results if not r.get("output_empty"))
            print(f"  {pm:8s}: {valid}/{len(pm_results)} schema_valid, "
                  f"{nonempty}/{len(pm_results)} nonempty")
            for r in pm_results:
                if not r.get("schema_valid"):
                    print(f"    FAIL {r['task_id']}: {r.get('errors', ['unknown'])[0]}")
                    print(f"    Raw output ({r.get('raw_output_length', 0)} chars): {r.get('raw_output', '')[:200]}")


if __name__ == "__main__":
    main()
