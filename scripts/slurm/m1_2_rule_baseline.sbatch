#!/usr/bin/env bash
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
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

RUN_ID="m1_2_slurm_${SLURM_JOB_ID:-local}"
OUTPUT_DIR="$PROJECT_ROOT/artifacts/m1_2"
mkdir -p "$OUTPUT_DIR/per_task" "$PROJECT_ROOT/logs"

# The Environment owns its Uvicorn and Chromium children and closes them by
# exact process/thread handles. Never use broad pkill patterns on a shared node.
trap 'echo "[signal] baseline interrupted at $(date -Is)" >&2; exit 143' SIGTERM SIGINT

 echo "=== M1.2 Rule Baseline (historical continuity) ==="
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Run ID: $RUN_ID"
echo "Git SHA: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Started: $(date -Is)"

python -m pytest -q tests/test_agent_env.py tests/test_action_submit_contract.py

srun --ntasks=1 python -m miniwebwork.baseline_runner \
    --agent rule \
    --tasks all \
    --max-steps 15 \
    --run-id "$RUN_ID" \
    --output-dir "$OUTPUT_DIR"

echo "=== Completed at $(date -Is) ==="
