#!/bin/bash
# Initialize MiniWebWork-RL procurement database
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork
python -m miniwebwork.cli init-db
echo "Database initialized at: ${MINIWEBWORK_DB_PATH:-$PROJECT_ROOT/data/runtime/miniwebwork.db}"
