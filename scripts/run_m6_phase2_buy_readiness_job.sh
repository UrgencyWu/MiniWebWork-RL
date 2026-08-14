#!/usr/bin/env bash
# P0b: CPU-only calibration of public item/option buy-readiness.
#SBATCH --job-name=m6-p2-readiness
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/m6_phase2_readiness_%j.out
#SBATCH --error=logs/m6_phase2_readiness_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase2_root="${M6_PHASE2_ROOT:-$study_root/phase2_causal_validation_v1}"
eval_root="${M6_PHASE2_EVAL_ROOT:-$study_root/medium/formal_dev_eval_v1}"
output="$phase2_root/p0b_buy_readiness/report.json"
test ! -e "$output"
mkdir -p "$(dirname "$output")"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase2_buy_readiness.py \
  --eval-root "$eval_root" \
  --output "$output"
