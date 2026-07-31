#!/bin/bash
#SBATCH --job-name=m2_3_mini_train
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/m2_3_mini_train_%j.out
#SBATCH --error=logs/m2_3_mini_train_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

# Pre-load PyTorch CUDA context
python -c "import torch; torch.cuda.is_available()"

mkdir -p logs outputs/m2_3_mini

echo "=== M2.3-mini Continued SFT Training ==="
echo "Job: ${SLURM_JOB_ID:-none}, Started: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"

export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/rollout_dev_no_solution_v1"

python scripts/m2_3_mini_continue_train.py \
    --lr 5e-5 \
    --epochs 2 \
    --batch-size 1 \
    --grad-accum 16 \
    --max-length 8192

echo "=== Training Complete: $(date) ==="
