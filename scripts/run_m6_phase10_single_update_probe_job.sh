#!/usr/bin/env bash
# Phase10: matched one-update teacher/rehearsal corrective-SFT readiness probe.
#SBATCH --job-name=m6-p10-one-update
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase10_single_update_%j.out
#SBATCH --error=logs/m6_phase10_single_update_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE10_PROBE_INPUTS:?missing M6_PHASE10_PROBE_INPUTS}"
: "${M6_PHASE10_PROBE_OUTPUT:?missing M6_PHASE10_PROBE_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$M6_PHASE10_PROBE_INPUTS"
test ! -e "$M6_PHASE10_PROBE_OUTPUT/single_update_probe_report.json"

python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase10_run_single_update_probe.py \
  --inputs "$M6_PHASE10_PROBE_INPUTS" \
  --base-model /data/share/model/Qwen3.5-4B \
  --student-tokenizer /data/share/model/Qwen3.5-4B \
  --pi0-adapter "$study_root/mini/pilot_sft/final_adapter" \
  --output-root "$M6_PHASE10_PROBE_OUTPUT"
