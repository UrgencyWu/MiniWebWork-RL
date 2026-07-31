#!/usr/bin/env bash
#SBATCH --job-name=m2_3_single_probe
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/m2_3_single_probe_%j.out
#SBATCH --error=logs/m2_3_single_probe_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/m2_3_mini

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

# Arguments: POLICY TEMPERATURE SEED K [MAX_TASKS] [TOP_P] [TOP_K]
POLICY="${1:-A}"
TEMP="${2:-0.2}"
SEED="${3:-20260731}"
K="${4:-8}"
MAX_TASKS="${5:-}"
TOP_P="${6:-0.9}"
TOP_K="${7:-0}"

case "$POLICY" in
    A) ADAPTER="$PROJECT_ROOT/outputs/m2_2r/seed_1234/final_adapter" ;;
    B) ADAPTER="$PROJECT_ROOT/outputs/m2_3_mini/seed_1234/final_adapter" ;;
    *) echo "ERROR: policy must be A or B, got '$POLICY'" >&2; exit 2 ;;
esac

TASK_DIR="$PROJECT_ROOT/data/tasks/rollout_dev_no_solution_v1"
export MINIWEBWORK_TASK_DIR="$TASK_DIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Set CUDA_LAUNCH_BLOCKING=1 only for a dedicated diagnostic run.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

float_tag() {
    awk -v value="$1" 'BEGIN {printf "%.8g", value}' | tr '.-' 'pm'
}
DIST_TAG="t$(float_tag "$TEMP")_p$(float_tag "$TOP_P")_k${TOP_K}"
HEARTBEAT_FILE="$PROJECT_ROOT/outputs/m2_3_mini/heartbeat_${POLICY}_${DIST_TAG}.json"
GPU_MEM_LOG="$PROJECT_ROOT/logs/gpu_mem_${SLURM_JOB_ID:-$$}.log"

cleanup() {
    if [[ -n "${MONITOR_PID:-}" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
}

on_signal() {
    echo "[signal] received at $(date -Is)" >&2
    echo "[signal] last heartbeat:" >&2
    cat "$HEARTBEAT_FILE" 2>/dev/null || echo "[signal] no heartbeat file" >&2
    echo "[signal] GPU processes:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
    exit 143
}

trap cleanup EXIT
trap on_signal SIGTERM SIGINT

(
    while true; do
        {
            date -Is
            nvidia-smi \
                --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader 2>/dev/null || true
        } >> "$GPU_MEM_LOG"
        sleep 60
    done
) &
MONITOR_PID=$!

[[ -d "$ADAPTER" ]] || { echo "ERROR: missing adapter: $ADAPTER" >&2; exit 1; }
[[ -f "$TASK_DIR/valid_public.jsonl" ]] || { echo "ERROR: missing valid public tasks" >&2; exit 1; }
[[ -f "$TASK_DIR/valid_oracle.jsonl" ]] || { echo "ERROR: missing valid oracle tasks" >&2; exit 1; }

python - <<'PY'
import importlib
for module in ("torch", "transformers", "peft", "playwright", "numpy"):
    importlib.import_module(module)
print("Dependency preflight: PASS")
PY

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "=== M2.3-mini Canonical Rollout Collection ==="
echo "Job: ${SLURM_JOB_ID:-none}"
echo "Git SHA: $GIT_SHA"
echo "Started: $(date -Is)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Policy: $POLICY"
echo "Temperature: $TEMP"
echo "Top-p: $TOP_P"
echo "Top-k: $TOP_K"
echo "Seed: $SEED"
echo "K: $K"
echo "Max tasks: ${MAX_TASKS:-all}"
echo "Adapter: $ADAPTER"
echo "Task dir: $TASK_DIR"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

ARGS=(
    --policy "$POLICY"
    --adapter "$ADAPTER"
    --temperature "$TEMP"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --seed "$SEED"
    --K "$K"
    --split valid
    --task-dir "$TASK_DIR"
)
if [[ -n "$MAX_TASKS" ]]; then
    ARGS+=(--max-tasks "$MAX_TASKS")
fi

srun --ntasks=1 python scripts/m2_3_mini_single_probe.py "${ARGS[@]}"

echo "=== Complete: $(date -Is) ==="
