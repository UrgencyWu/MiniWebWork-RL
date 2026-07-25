#!/bin/bash
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p artifacts/m2_0 logs

cleanup() {
    echo "=== Cleanup ==="
    pkill -f "uvicorn miniwebwork" 2>/dev/null || true
    pkill -f "chromium" 2>/dev/null || true
    echo "Cleanup done at $(date)"
}
trap cleanup EXIT INT TERM

echo "=== M2.0 Base Agent Evaluation ==="
echo "Job: ${SLURM_JOB_ID:-none}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CUDA: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date)"

echo "=== Running tests ==="
python -m pytest -q 2>&1 | tail -5

echo "=== Running full evaluation ==="
python -m miniwebwork.model_agent_runner \
    --model-path /data/share/model/Qwen3.5-4B \
    --tasks all \
    --max-model-turns 20 \
    --max-env-steps 15 \
    --history-window 5 \
    --output-dir artifacts/m2_0

RC=$?
echo "=== Eval exit: $RC ==="

echo "=== Artifacts ==="
ls -la artifacts/m2_0/ 2>/dev/null
echo "Per-task: $(ls artifacts/m2_0/per_task/ 2>/dev/null | wc -l)"

echo "=== Complete at $(date) ==="
exit $RC
