#!/bin/bash
#SBATCH --job-name=m1_1_e2e
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/m1_1_e2e_%j.out
#SBATCH --error=logs/m1_1_e2e_%j.err

# MiniWebWork-RL M1.1: Procurement E2E Test
# Start webapp, run Playwright procurement E2E, verify results.

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"

echo "=== M1.1 Procurement E2E Test ==="
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "Started: $(date)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

# Use job-specific DB
export MINIWEBWORK_DB_PATH="$PROJECT_ROOT/data/runtime/m1_1_slurm_${SLURM_JOB_ID:-local}.db"
export MINIWEBWORK_ARTIFACTS="$PROJECT_ROOT/artifacts"
export MINIWEBWORK_URL="http://127.0.0.1:18080"
export MINIWEBWORK_PORT="18080"

mkdir -p "$PROJECT_ROOT/artifacts/m1_1_screenshots"
mkdir -p "$PROJECT_ROOT/logs"

echo "DB: $MINIWEBWORK_DB_PATH"
echo "Artifacts: $MINIWEBWORK_ARTIFACTS"

# Init fresh DB
python -m miniwebwork.cli init-db
python -m miniwebwork.cli validate-tasks

# Cleanup trap
cleanup() {
    echo "=== Cleanup ==="
    if [ -n "${WEB_PID:-}" ]; then
        kill "$WEB_PID" 2>/dev/null || true
        wait "$WEB_PID" 2>/dev/null || true
    fi
    echo "Cleanup done at $(date)"
}
trap cleanup EXIT INT TERM

# Start webapp
echo "Starting webapp..."
python -m uvicorn miniwebwork.webapp:app \
    --host 127.0.0.1 --port "$MINIWEBWORK_PORT" \
    --log-level warning &
WEB_PID=$!
echo "Web PID: $WEB_PID"

# Wait for ready
echo "Waiting for webapp..."
for i in $(seq 1 30); do
    if curl -sS --max-time 2 "http://127.0.0.1:${MINIWEBWORK_PORT}/health" > /dev/null 2>&1; then
        echo "Webapp ready (attempt $i)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Webapp did not start"
        exit 1
    fi
    sleep 1
done

# Run E2E
echo "=== Running procurement E2E ==="
python -m miniwebwork.procurement_e2e
E2E_RC=$?

echo "E2E exit: $E2E_RC"

# Show results
if [ -f "$PROJECT_ROOT/artifacts/m1_1_e2e_result.json" ]; then
    echo "=== E2E Results ==="
    python -c "
import json
with open('$PROJECT_ROOT/artifacts/m1_1_e2e_result.json') as f:
    data = json.load(f)
for c in data['cases']:
    ok = 'PASS' if c['success'] else 'FAIL'
    print(f\"  {c['case']}: {ok} (ep={c.get('episode_id','')}, {c.get('elapsed_s','')}s)\")
    if not c['success'] and c.get('error'):
        print(f\"    error: {c['error'][:100]}\")
print(f\"Overall: {'PASS' if data['overall_success'] else 'FAIL'}\")
"

    # Show screenshot status
    echo "=== Screenshots ==="
    ls -la "$PROJECT_ROOT/artifacts/m1_1_screenshots/" 2>/dev/null || echo "no screenshots"
fi

echo "=== Completed at $(date) ==="
exit $E2E_RC
