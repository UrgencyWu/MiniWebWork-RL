#!/bin/bash
#SBATCH --job-name=m1_browser_smoke
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/m1_browser_smoke_%j.out
#SBATCH --error=logs/m1_browser_smoke_%j.err

# MiniWebWork-RL M1.0: Browser Smoke Test
# Start local web app, run Playwright/Chromium smoke test, collect artifacts.

set -euo pipefail

# Determine project root (use SLURM_SUBMIT_DIR since $0 is a spool path)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"

echo "=== M1.0 Browser Smoke Test ==="
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "Job started at: $(date)"

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork

echo "Python: $(python --version)"
echo "Playwright: $(python -m playwright --version)"

# Setup cleanup trap
cleanup() {
    echo "=== Cleanup ==="
    if [ -n "${WEB_PID:-}" ]; then
        kill "$WEB_PID" 2>/dev/null || true
        wait "$WEB_PID" 2>/dev/null || true
        echo "Web server stopped"
    fi
    echo "Cleanup done at $(date)"
}
trap cleanup EXIT INT TERM

# Import environment variables
export MINIWEBWORK_PORT="${MINIWEBWORK_PORT:-18080}"
export MINIWEBWORK_URL="http://127.0.0.1:${MINIWEBWORK_PORT}"
export MINIWEBWORK_ARTIFACTS="$PROJECT_ROOT/artifacts"

mkdir -p "$PROJECT_ROOT/artifacts"
mkdir -p "$PROJECT_ROOT/logs"

# Start local web app
echo "Starting web app on ${MINIWEBWORK_URL}..."
python -m uvicorn miniwebwork.webapp:app \
    --host 127.0.0.1 \
    --port "${MINIWEBWORK_PORT}" \
    --log-level info &
WEB_PID=$!
echo "Web server PID: $WEB_PID"

# Wait for /health to respond
echo "Waiting for web app to be ready..."
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if curl -sS --max-time 2 "http://127.0.0.1:${MINIWEBWORK_PORT}/health" > /dev/null 2>&1; then
        echo "Web app is ready (attempt $i)"
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "ERROR: Web app did not become ready within ${MAX_WAIT}s"
        exit 1
    fi
    sleep 1
done

# Health check
echo "=== Health check ==="
curl -sS "http://127.0.0.1:${MINIWEBWORK_PORT}/health" | python -m json.tool

# Run browser smoke test
echo "=== Running browser smoke test ==="
python -m miniwebwork.browser_smoke
SMOKE_EXIT=$?

echo "=== Smoke test exit code: $SMOKE_EXIT ==="

# Verify artifacts
echo "=== Artifacts ==="
if [ -f "$PROJECT_ROOT/artifacts/browser_smoke_result.json" ]; then
    echo "browser_smoke_result.json:"
    python -m json.tool "$PROJECT_ROOT/artifacts/browser_smoke_result.json"
else
    echo "ERROR: browser_smoke_result.json not found"
fi

if [ -f "$PROJECT_ROOT/artifacts/browser_smoke.png" ]; then
    echo "browser_smoke.png: exists ($(stat --format=%s "$PROJECT_ROOT/artifacts/browser_smoke.png") bytes)"
else
    echo "ERROR: browser_smoke.png not found"
fi

echo "=== Browser smoke test completed at $(date) ==="
exit $SMOKE_EXIT
