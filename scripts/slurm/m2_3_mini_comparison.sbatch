#!/bin/bash
#SBATCH --job-name=m2_3_comparison
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/m2_3_comparison_%j.out
#SBATCH --error=logs/m2_3_comparison_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

echo "=== M2.3-mini Controlled Comparison Probe ==="
echo "Job: ${SLURM_JOB_ID:-none}, Started: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"

export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/rollout_dev_no_solution_v1"

POLICY_A="$PROJECT_ROOT/outputs/m2_2r/seed_1234/final_adapter"
POLICY_B="$PROJECT_ROOT/outputs/m2_3_mini/seed_1234/final_adapter"

echo "Policy A (baseline): $POLICY_A"
echo "Policy B (patched):  $POLICY_B"

python scripts/m2_3_mini_comparison_probe.py \
    --policy-a "$POLICY_A" \
    --policy-b "$POLICY_B" \
    --temperatures 0.2 0.4 0.7 \
    --K 8 \
    --split valid

echo "=== Comparison Probe Complete: $(date) ==="
