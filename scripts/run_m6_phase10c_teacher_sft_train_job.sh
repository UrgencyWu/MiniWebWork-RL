#!/usr/bin/env bash
# Phase10-C: two-update full-corpus capability-weighted Raw35 LoRA training.
#SBATCH --job-name=m6-p10c-q35-sft
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/m6_phase10c_q35_sft_%j.out
#SBATCH --error=logs/m6_phase10c_q35_sft_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE10C_WEIGHTED_CORPUS:?missing M6_PHASE10C_WEIGHTED_CORPUS}"
: "${M6_PHASE10C_SFT_OUTPUT:?missing M6_PHASE10C_SFT_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$M6_PHASE10C_WEIGHTED_CORPUS/manifest.json"
test ! -e "$M6_PHASE10C_SFT_OUTPUT/training_report.json"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$python_bin" scripts/m6_phase10c_train_weighted_teacher_lora.py \
  --mode train \
  --corpus-dir "$M6_PHASE10C_WEIGHTED_CORPUS" \
  --base-model /data/share/model/Qwen3.5-35B-A3B \
  --output-dir "$M6_PHASE10C_SFT_OUTPUT"
