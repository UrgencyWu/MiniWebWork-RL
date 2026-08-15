#!/usr/bin/env bash
# Phase10-B: frozen pi_0 behavior collection for the eight-task zero-update OPD smoke.
#SBATCH --job-name=m6-p10b-opd-behavior
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase10b_opd_behavior_%j.out
#SBATCH --error=logs/m6_phase10b_opd_behavior_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE10B_OPD_SMOKE_ROSTER:?missing M6_PHASE10B_OPD_SMOKE_ROSTER}"
: "${M6_PHASE10B_ROSTER_PRODUCER_GIT_SHA:?missing M6_PHASE10B_ROSTER_PRODUCER_GIT_SHA}"
: "${M6_PHASE10B_OPD_BEHAVIOR_OUTPUT:?missing M6_PHASE10B_OPD_BEHAVIOR_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$M6_PHASE10B_OPD_SMOKE_ROSTER"
test ! -e "$M6_PHASE10B_OPD_BEHAVIOR_OUTPUT/collection_report.json"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$M6_PHASE10B_OPD_BEHAVIOR_OUTPUT"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase10b_opd_smoke_behavior --role train --k 4 --seed 20260852 \
  --split-lock "$study_root/locks/m6_webshop_split_v1.json" \
  --goals "$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json" \
  --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE10B_OPD_SMOKE_ROSTER" \
  --task-roster-producer-git-sha "$M6_PHASE10B_ROSTER_PRODUCER_GIT_SHA" \
  --base-model /data/share/model/Qwen3.5-4B \
  --base-model-manifest "$repo_root/data/m4_long_horizon_base_model_manifest_v1.json" \
  --adapter "$study_root/mini/pilot_sft/final_adapter" \
  --tensor-parallel-size 1 --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$M6_PHASE10B_OPD_BEHAVIOR_OUTPUT"
