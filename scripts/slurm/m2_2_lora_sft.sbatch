#!/bin/bash
#SBATCH --job-name=m2_2_sft
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/m2_2_sft_%j.out
#SBATCH --error=logs/m2_2_sft_%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/m2_1"

mkdir -p logs outputs/m2_2

echo "=== M2.2 LoRA SFT Pilot ==="
echo "Job: ${SLURM_JOB_ID:-none}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -2)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date)"

# Step 1: Teacher-forced evaluation on valid set (174 samples per seed)
echo ""
echo "=== Step 1: Teacher-Forced Evaluation ==="
for seed_dir in outputs/m2_2/seed_*; do
    [ -d "$seed_dir/final_adapter" ] || continue
    seed_name=$(basename "$seed_dir")
    echo "Evaluating $seed_name (teacher-forced)..."
    python -u -m miniwebwork.sft.evaluate_m2_2 \
        --base-model /data/share/model/Qwen3.5-4B \
        --adapter-dir "$seed_dir/final_adapter" \
        --mode teacher_forced \
        2>&1 | tee "logs/m2_2_eval_tf_${seed_name}.log"
done

# Step 2: Frozen 15-task end-to-end evaluation
echo ""
echo "=== Step 2: Frozen 15-Task E2E Evaluation ==="
for seed_dir in outputs/m2_2/seed_*; do
    [ -d "$seed_dir/final_adapter" ] || continue
    seed_name=$(basename "$seed_dir")
    echo "Evaluating $seed_name (frozen test)..."
    python -u -m miniwebwork.sft.evaluate_m2_2 \
        --base-model /data/share/model/Qwen3.5-4B \
        --adapter-dir "$seed_dir/final_adapter" \
        --mode frozen_test \
        2>&1 | tee "logs/m2_2_eval_frozen_${seed_name}.log"
done

# Step 3: Analysis
echo ""
echo "=== Step 3: Analysis ==="
python -u -m miniwebwork.sft.analyze_m2_2 2>&1 | tee logs/m2_2_analysis.log

echo ""
echo "=== Artifacts ==="
find outputs/m2_2 -type f | sort
find artifacts/m2_2 -type f | sort 2>/dev/null || true

echo ""
echo "=== M2.2 Complete at $(date) ==="
