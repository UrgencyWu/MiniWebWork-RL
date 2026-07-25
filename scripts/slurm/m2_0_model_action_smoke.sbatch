#!/bin/bash
#SBATCH --job-name=m2_0_smoke
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m2_0_smoke_%j.out
#SBATCH --error=logs/m2_0_smoke_%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork
export TOKENIZERS_PARALLELISM=false

mkdir -p artifacts/m2_0 logs

echo "=== M2.0 Model Action Smoke ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
import json, time, sys, os
from pathlib import Path
sys.path.insert(0, "src")

from miniwebwork.model_agent.model_backend import QwenTransformersBackend, ModelConfig
from miniwebwork.model_agent.prompt_builder import build_messages, load_system_prompt, prompt_sha256
from miniwebwork.model_agent.output_parser import parse
from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent

# Create a stub observation for TASK-001
from miniwebwork.agent_env.schemas import Observation, ElementDescriptor

obs = Observation(
    task_id="TASK-001", episode_id="EP-SMOKE",
    instruction="查找型号为 CC-A100X-80G 的 GPU 加速卡",
    step_index=0, url="http://127.0.0.1:18080/tasks/TASK-001",
    path="/tasks/TASK-001", page_type="task", title="Task TASK-001",
    elements=[
        ElementDescriptor("e1", "button", "button", "开始任务", "开始任务", "", "", "start-task-button", [], False),
    ],
    visible_text="MiniWebWork-RL Procurement\nTASK-001\n查找型号为 CC-A100X-80G 的 GPU 加速卡\n开始任务",
)

config = ModelConfig(model_path="/data/share/model/Qwen3.5-4B", max_new_tokens=128, enable_thinking=False)
backend = QwenTransformersBackend(config)
backend.load()

import miniwebwork.model_agent.prompt_builder as pb
agent = QwenBrowserAgent(backend, pb, parse)
agent.reset("TASK-001", obs.instruction)

results = []
for turn in range(3):
    print(f"\n--- Turn {turn+1} ---")
    attempt = agent.act(obs)
    d = {
        "turn": turn+1,
        "raw_output": attempt.raw_output,
        "strict_json_success": attempt.strict_json_success,
        "fallback_used": attempt.fallback_used,
        "schema_valid": attempt.schema_valid,
        "action": attempt.action.to_dict() if attempt.action else None,
        "errors": attempt.errors,
        "input_tokens": attempt.input_tokens,
        "output_tokens": attempt.output_tokens,
        "latency_ms": attempt.latency_ms,
    }
    results.append(d)
    print(json.dumps({k: v for k, v in d.items() if k != "raw_output"}, indent=2, ensure_ascii=False))
    print(f"raw_output: {attempt.raw_output[:200]}")

# Save
output = {
    "smoke_test": "M2.0 model action smoke",
    "model_info": backend.get_model_info(),
    "prompt_sha256": prompt_sha256(),
    "turns": results,
    "all_nonempty": all(r["raw_output"].strip() for r in results),
}
Path("artifacts/m2_0/m2_0_model_action_smoke.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
print("\nSmoke test complete")
PY

echo "=== Smoke done ==="
