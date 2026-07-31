#!/bin/bash
#SBATCH --job-name=m2_2r_canonical_train
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m2_2r_canonical_train_%j.out
#SBATCH --error=logs/m2_2r_canonical_train_%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/m2_1"

mkdir -p logs outputs/m2_2r artifacts/m2_2r

echo "=== M2.2R Canonical SFT Training ==="
echo "Job: ${SLURM_JOB_ID:-none}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"

python -u scripts/m2_2r_train_canonical.py --max-length 8192 2>&1 | tee logs/m2_2r_canonical_train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== Training complete at $(date) ==="
