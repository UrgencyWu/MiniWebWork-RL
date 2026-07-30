#!/bin/bash
#SBATCH --job-name=m2_3_single_probe
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/m2_3_single_probe_%j.out
#SBATCH --error=logs/m2_3_single_probe_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

echo "=== Single-Temperature Probe ==="
echo "Job: ${SLURM_JOB_ID:-none}, Started: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"

export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/rollout_dev_no_solution_v1"

POLICY="${1:-A}"
TEMP="${2:-0.2}"

if [ "$POLICY" = "A" ]; then
    ADAPTER="$PROJECT_ROOT/outputs/m2_2r/seed_1234/final_adapter"
else
    ADAPTER="$PROJECT_ROOT/outputs/m2_3_mini/seed_1234/final_adapter"
fi

echo "Policy: $POLICY, Temperature: $TEMP, Adapter: $ADAPTER"

python scripts/m2_3_mini_single_probe.py \
    --policy "$POLICY" \
    --adapter "$ADAPTER" \
    --temperature "$TEMP" \
    --K 8 \
    --split valid

echo "=== Complete: $(date) ==="
