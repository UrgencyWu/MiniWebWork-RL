#!/usr/bin/env bash
#SBATCH --job-name=m3_0_policy_eval
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/m3_0_policy_eval_%j.out
#SBATCH --error=logs/m3_0_policy_eval_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

# LABEL ADAPTER TASK_SOURCE TEMPERATURE TOP_P TOP_K SEED K [MAX_TASKS]
LABEL="${1:?Usage: sbatch m3_0_policy_eval.sbatch LABEL ADAPTER TASK_SOURCE TEMPERATURE TOP_P TOP_K SEED K [MAX_TASKS]}"
ADAPTER="${2:?missing adapter}"
TASK_SOURCE="${3:?missing task source}"
TEMPERATURE="${4:?missing temperature}"
TOP_P="${5:?missing top_p}"
TOP_K="${6:?missing top_k}"
SEED="${7:?missing seed}"
K="${8:?missing K}"
MAX_TASKS="${9:-}"

[[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: label must contain only letters, digits, dot, underscore or hyphen" >&2
  exit 2
}

[[ "$ADAPTER" = /* ]] || ADAPTER="$PROJECT_ROOT/$ADAPTER"
ADAPTER="$(readlink -f "$ADAPTER")"
[[ -d "$ADAPTER" ]] || { echo "ERROR: adapter not found: $ADAPTER" >&2; exit 1; }
case "$TASK_SOURCE" in
  no_solution) TASK_DIR="$PROJECT_ROOT/data/tasks/rollout_dev_no_solution_v1" ;;
  feasible) TASK_DIR="$PROJECT_ROOT/data/tasks/rollout_dev_feasible_v2" ;;
  /*) TASK_DIR="$TASK_SOURCE" ;;
  *) TASK_DIR="$PROJECT_ROOT/$TASK_SOURCE" ;;
esac
TASK_DIR="$(cd "$TASK_DIR" && pwd)"
[[ -f "$TASK_DIR/valid_public.jsonl" ]] || { echo "ERROR: invalid task dir: $TASK_DIR" >&2; exit 1; }

float_tag() { awk -v value="$1" 'BEGIN {printf "%.8g", value}' | tr '.-' 'pm'; }
TAG="${LABEL}_$(basename "$TASK_DIR")_t$(float_tag "$TEMPERATURE")_p$(float_tag "$TOP_P")_k${TOP_K}_s${SEED}_k${K}_j${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/m3_0_evaluations/$TAG"
mkdir -p "$OUTPUT_DIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== M3.0 frozen policy evaluation ==="
echo "Git SHA: $(git rev-parse HEAD)"
echo "Label: $LABEL"
echo "Adapter: $ADAPTER"
echo "Task source: $TASK_DIR"
echo "Sampling: T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K K=$K seed=$SEED"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

ARGS=(
  --policy custom
  --policy-label "$LABEL"
  --adapter "$ADAPTER"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --seed "$SEED"
  --K "$K"
  --split valid
  --task-dir "$TASK_DIR"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$MAX_TASKS" ]]; then
  ARGS+=(--max-tasks "$MAX_TASKS")
fi
srun --ntasks=1 python scripts/m2_3_mini_single_probe.py "${ARGS[@]}"
