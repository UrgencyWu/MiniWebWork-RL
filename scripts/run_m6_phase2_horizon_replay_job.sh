#!/usr/bin/env bash
# P1 r2: replay the audited short prefix and generate only the long continuation.
#SBATCH --job-name=m6-p2-horizon-replay
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase2_horizon_replay_%j.out
#SBATCH --error=logs/m6_phase2_horizon_replay_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE2_ROSTER:?missing M6_PHASE2_ROSTER}"
: "${M6_ROSTER_PRODUCER_GIT_SHA:?missing M6_ROSTER_PRODUCER_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase2_root="${M6_PHASE2_ROOT:-$study_root/phase2_causal_validation_v1}"
root="$phase2_root/p1_horizon"
source_root="$root/short_6_6"
output="$root/full_18_15_prefix_replay_r2"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
sft_adapter="$study_root/mini/pilot_sft/final_adapter"
test -f "$source_root/collection_report.json"
test ! -e "$output/collection_report.json"
mkdir -p "$output"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase2_horizon_evaluation --role train --k 4 --seed 20260821 \
  --split-lock "$split_lock" --goals "$goals" --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE2_ROSTER" \
  --task-roster-producer-git-sha "$M6_ROSTER_PRODUCER_GIT_SHA" \
  --replay-prefix-root "$source_root" \
  --adapter "$sft_adapter" \
  --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$output"
