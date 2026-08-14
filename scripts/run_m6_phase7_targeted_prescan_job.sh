#!/usr/bin/env bash
# Phase7: one targeted SFT-policy K4 full-horizon prescan; inference only.
#SBATCH --job-name=m6-p7-targeted-prescan
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase7_targeted_prescan_%j.out
#SBATCH --error=logs/m6_phase7_targeted_prescan_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE7_ROSTER:?missing M6_PHASE7_ROSTER}"
: "${M6_PHASE7_ROSTER_PRODUCER_GIT_SHA:?missing M6_PHASE7_ROSTER_PRODUCER_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
sft_adapter="$study_root/mini/pilot_sft/final_adapter"
output="${M6_PHASE7_OUTPUT:-$study_root/phase7_targeted_prescan_v1/collection}"
test -f "$M6_PHASE7_ROSTER"
test -d "$sft_adapter"
test ! -e "$output/collection_report.json"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
mkdir -p "$output"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase4_data_synthesis --role train --k 4 --seed 20260832 \
  --split-lock "$split_lock" --goals "$goals" --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE7_ROSTER" \
  --task-roster-producer-git-sha "$M6_PHASE7_ROSTER_PRODUCER_GIT_SHA" \
  --adapter "$sft_adapter" \
  --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$output"
