#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m compileall -q src scripts tests
python -m miniwebwork.cli init-db
python -m miniwebwork.cli validate-seed
python -m miniwebwork.cli validate-tasks
python -m pytest -q "$@"
