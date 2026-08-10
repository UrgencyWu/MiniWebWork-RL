#!/usr/bin/env bash
# Audit an already-running shared service with one CPU; no model work occurs.
#SBATCH --job-name=m5-webshop-health
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/m5_webshop_health_%j.out
#SBATCH --error=logs/m5_webshop_health_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 preflight commit SHA}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
output="$repo_root/outputs/m5_webshop_credit_assignment_v1/preflight/server/health_audit.json"
workers="${M5_WEBSHOP_WORKERS:-4}"
export CUDA_VISIBLE_DEVICES=""
/home/wushaohua/miniconda3/envs/miniwebwork/bin/python scripts/m5_webshop_server_preflight.py health \
  --base-url http://127.0.0.1:44151 \
  --expected-workers "$workers" \
  --output "$output"
