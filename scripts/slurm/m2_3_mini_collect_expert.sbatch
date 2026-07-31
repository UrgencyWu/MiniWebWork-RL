#!/bin/bash
#SBATCH --job-name=m2_3_mini_expert
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/m2_3_mini_expert_%j.out
#SBATCH --error=logs/m2_3_mini_expert_%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

# Pre-load PyTorch CUDA context to avoid AllocatorConfig crash
python -c "import torch; torch.cuda.is_available()"

mkdir -p logs data/expert/m2_3_mini data/tasks/rollout_dev_no_solution_v1

cleanup() {
    echo "=== Cleanup ==="
    pkill -f "uvicorn miniwebwork" 2>/dev/null || true
    pkill -f "chromium" 2>/dev/null || true
    echo "Cleanup done at $(date)"
}
trap cleanup EXIT INT TERM

echo "=== M2.3-mini Expert Trajectory Collection ==="
echo "Job: ${SLURM_JOB_ID:-none}, Started: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"

export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/rollout_dev_no_solution_v1"

python scripts/m2_3_mini_collect_expert.py --split all

echo "=== Expert Collection Complete: $(date) ==="
