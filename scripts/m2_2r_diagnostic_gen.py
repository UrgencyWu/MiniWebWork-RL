"""M2.2R Diagnostic: capture raw model output for first task."""
import sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.model_agent.prompt_builder import build_messages, load_system_prompt, HISTORY_WINDOW
from miniwebwork.model_agent.output_parser import parse
from miniwebwork.model_agent.agent_loop import run_model_episode
from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent
from miniwebwork.model_agent.model_backend import QwenTransformersBackend, ModelConfig
from miniwebwork.tasks import load_public_tasks

BASE_MODEL = "/data/share/model/Qwen3.5-4B"
ADAPTER = Path("outputs/m2_2r/seed_42/final_adapter")

# Load model
print("Loading model...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)
base = base.to("cuda:0")
model = PeftModel.from_pretrained(base, str(ADAPTER))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check chat template
ct = tokenizer.chat_template or ""
print(f"Chat template has <think>: {'<think>' in ct}")
print(f"Chat template has thinking tag: {'thinking' in ct.lower()}")
print()

# Setup backend
config = ModelConfig(model_path=BASE_MODEL, max_new_tokens=512, enable_thinking=False, dtype="bfloat16", device="cuda:0")
backend = QwenTransformersBackend(config)
backend._model = model
backend._tokenizer = tokenizer
backend._loaded = True

import miniwebwork.model_agent.prompt_builder as pb
pb.HISTORY_WINDOW = 5
agent = QwenBrowserAgent(backend, pb, parse)

# Run ONE task and capture raw output
tasks = load_public_tasks()
os.environ["MINIWEBWORK_TASK_DIR"] = str(Path(".").resolve() / "data" / "tasks" / "m2_1")

print("Running TASK-001 with SFT seed 42...\n")
with ProcurementBrowserEnv(max_steps=5, run_id="diag") as env:
    env.set_agent_name("diag")
    result = run_model_episode("TASK-001", env, agent, max_model_turns=5, max_env_steps=15)

print(f"Success: {result['success']}")
print(f"Termination: {result['termination_reason']}")
print(f"Model turns: {result['model_turns']}")
print()

for i, turn in enumerate(result.get("turns", [])):
    print(f"--- Turn {i} ---")
    print(f"  Raw output ({len(turn.get('raw_output', ''))} chars):")
    raw = turn.get("raw_output", "")
    print(f"  {repr(raw[:500])}")
    print(f"  Strict JSON: {turn.get('strict_json_success')}")
    print(f"  Schema valid: {turn.get('schema_valid')}")
    print(f"  Errors: {turn.get('errors', [])}")
    print()
