#!/usr/bin/env bash
# P2: same-batch binary versus strict-dominant reward gradient counterfactual.
#SBATCH --job-name=m6-p2-reward-probe
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase2_reward_probe_%j.out
#SBATCH --error=logs/m6_phase2_reward_probe_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase2_root="${M6_PHASE2_ROOT:-$study_root/phase2_causal_validation_v1}"
groups_dir="$phase2_root/p1_horizon/full_18_15_prefix_replay_r2/groups"
readiness_report="$phase2_root/p0b_buy_readiness_v2/report.json"
sft_adapter="$study_root/mini/pilot_sft/final_adapter"
output="$phase2_root/p2_reward_counterfactual/report.json"
test ! -e "$output"
test -f "$readiness_report"
grep -q '"process_reward_calibration_passed":true' "$readiness_report"
mkdir -p "$(dirname "$output")"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase2_reward_probe.py \
  --groups-dir "$groups_dir" \
  --readiness-report "$readiness_report" \
  --sft-adapter "$sft_adapter" \
  --output "$output"

