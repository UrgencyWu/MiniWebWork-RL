#!/bin/bash
# Start MiniWebWork-RL local web app for smoke testing.
# Usage: bash scripts/run_local_site.sh

set -euo pipefail

HOST="${MINIWEBWORK_HOST:-127.0.0.1}"
PORT="${MINIWEBWORK_PORT:-18080}"

echo "Starting MiniWebWork-RL web app on http://${HOST}:${PORT}"

cd "$(dirname "$0")/.."

exec python -m uvicorn miniwebwork.webapp:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info
