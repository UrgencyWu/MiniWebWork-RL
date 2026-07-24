#!/bin/bash
# Start MiniWebWork-RL procurement web application
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

HOST="${MINIWEBWORK_HOST:-127.0.0.1}"
PORT="${MINIWEBWORK_PORT:-18080}"

echo "Starting MiniWebWork-RL Procurement Site on http://${HOST}:${PORT}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork
exec python -m uvicorn miniwebwork.webapp:app --host "$HOST" --port "$PORT" --log-level info
