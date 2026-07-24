#!/bin/bash
#SBATCH --job-name=m1_model_smoke
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m1_model_smoke_%j.out
#SBATCH --error=logs/m1_model_smoke_%j.err

# MiniWebWork-RL M1.0: Model Smoke Test
# Load Qwen3.5-4B and perform a minimal text generation.

set -euo pipefail

# Determine project root (use SLURM_SUBMIT_DIR since $0 is a spool path)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"

echo "=== M1.0 Model Smoke Test ==="
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "Job started at: $(date)"

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

echo "Python: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# Print GPU info
nvidia-smi

# Model path
export MODEL_PATH="${MODEL_PATH:-/data/share/model/Qwen3.5-4B}"

export MINIWEBWORK_ARTIFACTS="$PROJECT_ROOT/artifacts"
mkdir -p "$PROJECT_ROOT/artifacts"
mkdir -p "$PROJECT_ROOT/logs"

echo "Model path: $MODEL_PATH"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please download Qwen3.5-4B first using ModelScope SDK."
    exit 1
fi

echo "=== Running model smoke test ==="
python -m miniwebwork.model_smoke
MODEL_EXIT=$?

echo "=== Model test exit code: $MODEL_EXIT ==="

# Verify artifacts
if [ -f "$PROJECT_ROOT/artifacts/model_smoke_result.json" ]; then
    echo "model_smoke_result.json:"
    python -m json.tool "$PROJECT_ROOT/artifacts/model_smoke_result.json"
else
    echo "ERROR: model_smoke_result.json not found"
fi

echo "=== Model smoke test completed at $(date) ==="
exit $MODEL_EXIT
