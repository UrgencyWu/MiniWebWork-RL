#!/usr/bin/env bash
#SBATCH --job-name=m2_0_eval
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m2_0_eval_%j.out
#SBATCH --error=logs/m2_0_eval_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OUTPUT_DIR="$PROJECT_ROOT/artifacts/m2_0_canonical_v2"
mkdir -p "$OUTPUT_DIR" logs

# Environment children are closed by exact handles. Broad process-name cleanup
# is forbidden on a shared Slurm node.
trap 'echo "[signal] Base evaluation interrupted at $(date -Is)" >&2; exit 143' SIGTERM SIGINT

 echo "=== Canonical Base Agent Evaluation ==="
echo "Job: ${SLURM_JOB_ID:-none}"
echo "Git SHA: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

python -m pytest -q \
    tests/test_model_backend_logprobs.py \
    tests/test_qwen_agent_feedback_contract.py \
    tests/test_environment_lifecycle_contract.py

srun --ntasks=1 python -m miniwebwork.model_agent_runner \
    --model-path /data/share/model/Qwen3.5-4B \
    --tasks all \
    --prompt-version browser_agent_v2 \
    --max-model-turns 20 \
    --max-env-steps 15 \
    --history-window 5 \
    --output-dir "$OUTPUT_DIR"

echo "=== Complete at $(date -Is) ==="
