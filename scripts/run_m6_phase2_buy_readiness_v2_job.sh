#!/usr/bin/env bash
# P0b-r2: CPU-only calibration of the equal-block public buy-readiness score.
#SBATCH --job-name=m6-p2-readiness-v2
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/m6_phase2_readiness_v2_%j.out
#SBATCH --error=logs/m6_phase2_readiness_v2_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase2_root="${M6_PHASE2_ROOT:-$study_root/phase2_causal_validation_v1}"
horizon_root="$phase2_root/p1_horizon"
output="$phase2_root/p0b_buy_readiness_v2/report.json"
test ! -e "$output"
test -f "$horizon_root/analysis_report_r2.json"
grep -q '"decision":"USE_FULL_HORIZON"' "$horizon_root/analysis_report_r2.json"
mkdir -p "$(dirname "$output")"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase2_buy_readiness.py \
  --eval-root "$horizon_root" \
  --identity full_18_15_prefix_replay_r2 \
  --formula-version equal_item_option_blocks_v2 \
  --validation-eval-root "$study_root/medium/formal_dev_eval_v1" \
  --validation-identity sft \
  --output "$output"

