#!/bin/bash
#SBATCH --job-name=m2_3_single_probe
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/m2_3_single_probe_%j.out
#SBATCH --error=logs/m2_3_single_probe_%j.err
#SBATCH --gres=gpu:1

set -o pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

echo "=== Single-Temperature Probe ==="
echo "Job: ${SLURM_JOB_ID:-none}, Started: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Host: $(hostname)"

# --- Diagnostics ---
export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/rollout_dev_no_solution_v1"

# Signal trap: write diagnostic info on cancellation/timeout
_JOB_ID="${SLURM_JOB_ID:-$$}"
trap '{
    echo "[TRAP] Signal received at $(date)"
    echo "[TRAP] GPU memory:"
    python3 -c "import torch; print(f\"  allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB\")" 2>/dev/null || true
}' SIGTERM SIGINT

# Monitor GPU memory in background
_MEM_LOG="$PROJECT_ROOT/logs/gpu_mem_${_JOB_ID}.log"
(
    while true; do
        python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.synchronize()
    print(f\"\$(date +%s) alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB\", flush=True)
" 2>/dev/null
        sleep 60
    done
) &
_MEM_PID=$!
echo "GPU memory monitor PID: $_MEM_PID"

cleanup() {
    echo "[cleanup] Stopping GPU memory monitor..."
    kill $_MEM_PID 2>/dev/null || true
    wait $_MEM_PID 2>/dev/null || true
}
trap cleanup EXIT

POLICY="${1:-A}"
TEMP="${2:-0.2}"

if [ "$POLICY" = "A" ]; then
    ADAPTER="$PROJECT_ROOT/outputs/m2_2r/seed_1234/final_adapter"
else
    ADAPTER="$PROJECT_ROOT/outputs/m2_3_mini/seed_1234/final_adapter"
fi

echo "Policy: $POLICY, Temperature: $TEMP, Adapter: $ADAPTER"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"

python scripts/m2_3_mini_single_probe.py \
    --policy "$POLICY" \
    --adapter "$ADAPTER" \
    --temperature "$TEMP" \
    --seed 20260731 \
    --K 8 \
    --split valid

_EXIT_CODE=$?
echo "=== Exit code: $_EXIT_CODE | Complete: $(date) ==="
exit $_EXIT_CODE
