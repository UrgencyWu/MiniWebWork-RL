#!/bin/bash
#SBATCH --job-name=m2_2r_diagnostic
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m2_2r_diagnostic_%j.out
#SBATCH --error=logs/m2_2r_diagnostic_%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/m2_1"

echo "=== M2.2R Diagnostic: Raw Output Capture ==="
echo "Job: ${SLURM_JOB_ID:-none}"
echo "Started: $(date)"

python -u scripts/m2_2r_diagnostic_gen.py 2>&1 | tee logs/m2_2r_diagnostic_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== Diagnostic complete at $(date) ==="
