#!/usr/bin/env bash
#SBATCH --job-name=m3_0b1_smoke
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/m3_0b1_smoke_%j.out
#SBATCH --error=logs/m3_0b1_smoke_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/m3_0b1_smoke

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

# Arguments: STRICT_ARTIFACT POLICY [TASK_ID] [LEARNING_RATE]
ARTIFACT="${1:?Usage: sbatch m3_0_single_batch_smoke.sbatch STRICT_ARTIFACT POLICY [TASK_ID] [LEARNING_RATE]}"
POLICY="${2:-B}"
TASK_ID="${3:-}"
LEARNING_RATE="${4:-1e-6}"

case "$POLICY" in
    A) ADAPTER="$PROJECT_ROOT/outputs/m2_2r/seed_1234/final_adapter" ;;
    B) ADAPTER="$PROJECT_ROOT/outputs/m2_3_mini/seed_1234/final_adapter" ;;
    *) echo "ERROR: policy must be A or B, got '$POLICY'" >&2; exit 2 ;;
esac

if [[ "$ARTIFACT" != /* ]]; then
    ARTIFACT="$PROJECT_ROOT/$ARTIFACT"
fi
ARTIFACT="$(readlink -f "$ARTIFACT")"
OUTPUT_DIR="$PROJECT_ROOT/outputs/m3_0b1_smoke/${POLICY}_${SLURM_JOB_ID:-manual}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

[[ -f "$ARTIFACT" ]] || { echo "ERROR: strict artifact not found: $ARTIFACT" >&2; exit 1; }
[[ -d "$ADAPTER" ]] || { echo "ERROR: adapter not found: $ADAPTER" >&2; exit 1; }

python - <<'PY'
import importlib
for module in ("torch", "transformers", "peft", "safetensors"):
    importlib.import_module(module)
print("Dependency preflight: PASS")
PY

echo "=== M3.0B-1 Single-batch LoRA Smoke ==="
echo "Job: ${SLURM_JOB_ID:-none}"
echo "Git SHA: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Started: $(date -Is)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Policy: $POLICY"
echo "Artifact: $ARTIFACT"
echo "Adapter: $ADAPTER"
echo "Task ID: ${TASK_ID:-auto-first-valid-group}"
echo "Learning rate: $LEARNING_RATE"
echo "Output: $OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

ARGS=(
    --artifact "$ARTIFACT"
    --adapter "$ADAPTER"
    --output-dir "$OUTPUT_DIR"
    --learning-rate "$LEARNING_RATE"
    --clip-epsilon 0.2
    --gradient-clip 1.0
    --old-current-tolerance 0.05
)
if [[ -n "$TASK_ID" ]]; then
    ARGS+=(--task-id "$TASK_ID")
fi

srun --ntasks=1 python scripts/m3_0_single_batch_smoke.py "${ARGS[@]}"

echo "=== Complete: $(date -Is) ==="
