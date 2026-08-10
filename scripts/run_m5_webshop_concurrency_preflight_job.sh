#!/usr/bin/env bash
# Exercise real reset/search traffic against one already-healthy shared service.
# No GPU or model generation is used.
#SBATCH --job-name=m5-webshop-load
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/m5_webshop_load_%j.out
#SBATCH --error=logs/m5_webshop_load_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 preflight commit SHA}"
: "${M5_WEBSHOP_WORKERS:?set the running service worker count}"
: "${M5_STRESS_LANES:?set the frozen stress lane count}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

case "$M5_WEBSHOP_WORKERS" in 8|16) ;; *) exit 2 ;; esac
case "$M5_STRESS_LANES" in 32|64) ;; *) exit 2 ;; esac

study_root="$repo_root/outputs/m5_webshop_credit_assignment_v1"
python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
health_audit="$study_root/preflight/server/health_audit.json"
output="$study_root/preflight/service_stress/workers_${M5_WEBSHOP_WORKERS}_lanes_${M5_STRESS_LANES}.json"
export CUDA_VISIBLE_DEVICES=""

"$python_bin" scripts/m5_webshop_concurrency_preflight.py \
  --workers "$M5_WEBSHOP_WORKERS" \
  --lanes "$M5_STRESS_LANES" \
  --episodes-per-lane "${M5_STRESS_EPISODES_PER_LANE:-4}" \
  --health-audit "$health_audit" \
  --output "$output"
