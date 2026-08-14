#!/usr/bin/env bash
# P1: paired SFT rollout with only the model/environment horizon changed.
#SBATCH --job-name=m6-p2-horizon
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --output=logs/m6_phase2_horizon_%A_%a.out
#SBATCH --error=logs/m6_phase2_horizon_%A_%a.err

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
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
sft_adapter="$study_root/mini/pilot_sft/final_adapter"
case "${SLURM_ARRAY_TASK_ID:?missing array id}" in
  0) label=short_6_6; model_turns=6; environment_steps=6 ;;
  1) label=full_18_15; model_turns=18; environment_steps=15 ;;
esac
output="$phase2_root/p1_horizon/$label"
test ! -e "$output/collection_report.json"
mkdir -p "$output"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase2_horizon_evaluation --role train --k 4 --seed 20260821 \
  --split-lock "$split_lock" --goals "$goals" --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE2_ROSTER" \
  --task-roster-producer-git-sha "$M6_ROSTER_PRODUCER_GIT_SHA" \
  --adapter "$sft_adapter" \
  --max-model-turns "$model_turns" --max-environment-steps "$environment_steps" \
  --concurrent-groups 4 --output-dir "$output"
