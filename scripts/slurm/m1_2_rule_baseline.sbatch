#!/bin/bash
#SBATCH --job-name=m1_2_baseline
#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/m1_2_rule_baseline_%j.out
#SBATCH --error=logs/m1_2_rule_baseline_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"

RUN_ID="m1_2_slurm_${SLURM_JOB_ID:-local}"
echo "=== M1.2 Rule Baseline ==="
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Run ID: $RUN_ID"
echo "Started: $(date)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

export MINIWEBWORK_ARTIFACTS="$PROJECT_ROOT/artifacts"

mkdir -p "$PROJECT_ROOT/artifacts/m1_2/per_task"
mkdir -p "$PROJECT_ROOT/logs"

cleanup() {
    echo "=== Cleanup ==="
    pkill -f "uvicorn miniwebwork" 2>/dev/null || true
    pkill -f "chromium" 2>/dev/null || true
    echo "Cleanup done at $(date)"
}
trap cleanup EXIT INT TERM

echo "=== Running tests ==="
python -m pytest -q 2>&1 | tail -10

echo "=== Running baseline ==="
python -m miniwebwork.baseline_runner \
    --agent rule \
    --tasks all \
    --max-steps 15 \
    --run-id "$RUN_ID" \
    --output-dir "$PROJECT_ROOT/artifacts/m1_2"

RC=$?

echo "=== Baseline exit: $RC ==="
echo "=== Artifacts ==="
ls -la "$PROJECT_ROOT/artifacts/m1_2/" 2>/dev/null || echo "No artifacts"
ls "$PROJECT_ROOT/artifacts/m1_2/per_task/" 2>/dev/null | head -20 || echo "No per-task files"

echo "=== Completed at $(date) ==="
exit $RC
