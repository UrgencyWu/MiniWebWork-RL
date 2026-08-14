#!/usr/bin/env bash
# CPU audit after both paired horizon arms complete.
#SBATCH --job-name=m6-p2-horizon-audit
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m6_phase2_horizon_audit_%j.out
#SBATCH --error=logs/m6_phase2_horizon_audit_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
root="$repo_root/outputs/m6_monotonic_posttraining_v1/phase2_causal_validation_v1/p1_horizon"
test ! -e "$root/analysis_report.json"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase2_horizon_analysis.py \
  --short-root "$root/short_6_6" \
  --full-root "$root/full_18_15" \
  --output "$root/analysis_report.json"
