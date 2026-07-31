#!/usr/bin/env bash
#SBATCH --job-name=m3_0b_logprob_audit
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/m3_0b_logprob_audit_%j.out
#SBATCH --error=logs/m3_0b_logprob_audit_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

ARTIFACT="${1:?Usage: sbatch m3_0b_logprob_audit.sbatch STRICT_ARTIFACT [POLICY] [MAX_NEW_TOKENS]}"
POLICY="${2:-A}"
MAX_NEW_TOKENS="${3:-16}"

case "$POLICY" in
  A) ADAPTER="$PROJECT_ROOT/outputs/m2_2r/seed_1234/final_adapter" ;;
  B) ADAPTER="$PROJECT_ROOT/outputs/m2_3_mini/seed_1234/final_adapter" ;;
  *) echo "ERROR: policy must be A or B" >&2; exit 2 ;;
esac
[[ "$ARTIFACT" = /* ]] || ARTIFACT="$PROJECT_ROOT/$ARTIFACT"
ARTIFACT="$(readlink -f "$ARTIFACT")"
[[ -f "$ARTIFACT" ]] || { echo "ERROR: artifact not found: $ARTIFACT" >&2; exit 1; }
[[ -d "$ADAPTER" ]] || { echo "ERROR: adapter not found: $ADAPTER" >&2; exit 1; }

mkdir -p logs outputs/m3_0b_audit
OUTPUT="$PROJECT_ROOT/outputs/m3_0b_audit/logprob_audit_${POLICY}_${SLURM_JOB_ID:-manual}.json"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== M3.0B strict logprob audit ==="
echo "Job: ${SLURM_JOB_ID:-none}"
echo "Artifact: $ARTIFACT"
echo "Policy: $POLICY"
echo "Output: $OUTPUT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

srun --ntasks=1 python scripts/m3_0b_logprob_audit.py \
  --artifact "$ARTIFACT" \
  --adapter "$ADAPTER" \
  --output "$OUTPUT" \
  --max-new-tokens "$MAX_NEW_TOKENS"
