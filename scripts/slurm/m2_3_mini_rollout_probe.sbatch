#!/bin/bash
#SBATCH --job-name=m2_3_mini_probe
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/m2_3_mini_probe_%j.out
#SBATCH --error=logs/m2_3_mini_probe_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

# Verify dependencies
python -c "import torch, transformers, peft, playwright; print(f'OK: torch={torch.__version__}, transformers={transformers.__version__}, peft={peft.__version__}')"

# Slurm assigns one GPU; export it explicitly so child processes see it.
# DO NOT override CUDA_VISIBLE_DEVICES in the Python probe script —
# doing so after CUDA context initialization causes IndexKernel crashes.
echo "Slurm GPU assignment: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Slurm GPU list: ${SLURM_JOB_GPUS:-unset}"

mkdir -p logs outputs/m2_3_mini

echo "=== M2.3-mini Rollout Probe ==="
echo "Job: ${SLURM_JOB_ID:-none}, Started: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"

export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/rollout_dev_no_solution_v1"

# Find adapter path
ADAPTER="$PROJECT_ROOT/outputs/m2_3_mini/seed_1234/final_adapter"
if [ ! -d "$ADAPTER" ]; then
    ADAPTER="$PROJECT_ROOT/outputs/m2_2r/seed_1234/final_adapter"
fi
echo "Adapter: $ADAPTER"

python scripts/m2_3_mini_rollout_probe.py \
    --adapter "$ADAPTER" \
    --temperatures 0.2 0.4 0.7 \
    --K 8 \
    --split valid

echo "=== Rollout Probe Complete: $(date) ==="
