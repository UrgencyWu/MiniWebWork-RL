#!/usr/bin/env bash
# CPU-only paired analysis after all eight formal-dev identities complete.
#SBATCH --job-name=m6-medium-eval-analysis
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/m6_medium_eval_analysis_%j.out
#SBATCH --error=logs/m6_medium_eval_analysis_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_EVAL_ROOT:?set M6_EVAL_ROOT}"
: "${M6_EVAL_PLAN:?set M6_EVAL_PLAN}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
/home/wushaohua/miniconda3/envs/miniwebwork/bin/python scripts/m6_medium_eval_analysis.py \
  --eval-root "$M6_EVAL_ROOT" \
  --online-root "$repo_root/outputs/m6_monotonic_posttraining_v1/medium/online_v1" \
  --plan "$M6_EVAL_PLAN" \
  --output "$M6_EVAL_ROOT/analysis_report.json"
