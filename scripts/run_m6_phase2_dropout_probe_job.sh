#!/usr/bin/env bash
# P0a: compare training-forward gradient noise with RL dropout on versus off.
#SBATCH --job-name=m6-p2-dropout
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase2_dropout_%j.out
#SBATCH --error=logs/m6_phase2_dropout_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase2_root="${M6_PHASE2_ROOT:-$study_root/phase2_causal_validation_v1}"
group="${M6_PHASE2_GROUP:-$study_root/medium/online_v1/seed_20260812/multi_turn_grpo/iteration_0/collection/groups/g0000.json}"
sft_adapter="${M6_SFT_ADAPTER:-$study_root/mini/pilot_sft/final_adapter}"
output="${M6_PHASE2_DROPOUT_OUTPUT:-$phase2_root/p0a_dropout/report.json}"
test ! -e "$output"
mkdir -p "$(dirname "$output")"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase2_dropout_probe.py \
  --group "$group" \
  --sft-adapter "$sft_adapter" \
  --output "$output"
