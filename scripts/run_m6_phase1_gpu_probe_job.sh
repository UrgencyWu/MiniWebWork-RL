#!/usr/bin/env bash
# One frozen K8 group: parameter-gradient counterfactual plus three LR single steps.
#SBATCH --job-name=m6-p1-probe
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase1_probe_%j.out
#SBATCH --error=logs/m6_phase1_probe_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE1_ROOT:?missing M6_PHASE1_ROOT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
group="$study_root/medium/online_v1/seed_20260812/multi_turn_grpo/iteration_0/collection/groups/g0000.json"
sft_adapter="$study_root/mini/pilot_sft/final_adapter"
mkdir -p "$M6_PHASE1_ROOT/gpu_probe"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase1_gpu_probe.py --group "$group" \
  --sft-adapter "$sft_adapter" --learning-rates 0.000001 0.000003 0.00001 \
  --output "$M6_PHASE1_ROOT/gpu_probe/report.json"
