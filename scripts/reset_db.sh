#!/bin/bash
# Reset MiniWebWork-RL procurement database
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork
python -m miniwebwork.cli reset-db
echo "Database reset complete."
