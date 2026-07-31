#!/usr/bin/env bash
#SBATCH --job-name=m3_0_update
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/m3_0_update_%j.out
#SBATCH --error=logs/m3_0_update_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/m3_0_updates
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

ARTIFACT="${1:?Usage: sbatch m3_0_single_batch_update.sbatch STRICT_ARTIFACT POLICY [TASK_ID] [LEARNING_RATE]}"
POLICY="${2:-A}"
TASK_ID="${3:-}"
LEARNING_RATE="${4:-1e-6}"

case "$POLICY" in
  A) ADAPTER="$PROJECT_ROOT/outputs/m2_2r/seed_1234/final_adapter" ;;
  B) ADAPTER="$PROJECT_ROOT/outputs/m2_3_mini/seed_1234/final_adapter" ;;
  *) echo "ERROR: policy must be A or B, got '$POLICY'" >&2; exit 2 ;;
esac
[[ "$ARTIFACT" = /* ]] || ARTIFACT="$PROJECT_ROOT/$ARTIFACT"
ARTIFACT="$(readlink -f "$ARTIFACT")"
[[ -f "$ARTIFACT" ]] || { echo "ERROR: artifact not found: $ARTIFACT" >&2; exit 1; }
[[ -d "$ADAPTER" ]] || { echo "ERROR: adapter not found: $ADAPTER" >&2; exit 1; }

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

echo "=== M3.0 formal single-batch GRPO update ==="
echo "Job: ${SLURM_JOB_ID:-none}"
echo "Git SHA: $(git rev-parse HEAD)"
echo "Artifact: $ARTIFACT"
echo "Policy: $POLICY"
echo "Adapter: $ADAPTER"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

ARGS=(
  --artifact "$ARTIFACT"
  --adapter "$ADAPTER"
  --output-dir "$PROJECT_ROOT/outputs/m3_0_updates/${POLICY}_${SLURM_JOB_ID:-manual}"
  --learning-rate "$LEARNING_RATE"
  --clip-epsilon 0.2
  --gradient-clip 1.0
  --old-current-tolerance 0.05
  --formal
)
if [[ -n "$TASK_ID" ]]; then
  ARGS+=(--task-id "$TASK_ID")
fi

srun --ntasks=1 python scripts/m3_0_single_batch_smoke.py "${ARGS[@]}"
echo "=== Complete: $(date -Is) ==="
