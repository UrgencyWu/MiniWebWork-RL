#!/usr/bin/env bash
#SBATCH --job-name=m6-p1-final
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m6_phase1_finalize_%j.out
#SBATCH --error=logs/m6_phase1_finalize_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE1_ROOT:?missing M6_PHASE1_ROOT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase1_diagnostics.py \
  --online-root "$study_root/medium/online_v1" \
  --eval-root "$study_root/medium/formal_dev_eval_v1" \
  --seen-eval-root "$M6_PHASE1_ROOT/seen_task_eval" \
  --gpu-probe "$M6_PHASE1_ROOT/gpu_probe/report.json" \
  --output "$M6_PHASE1_ROOT/final_report.json"
