#!/usr/bin/env bash
# Phase10-D: Qwen3.5-35B-A3B on the exact D4 supervision corpus.
#SBATCH --job-name=m6-p10d-s35-d4
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/m6_phase10d_s35_d4_%j.out
#SBATCH --error=logs/m6_phase10d_s35_d4_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE10D_D4_CORPUS:?missing M6_PHASE10D_D4_CORPUS}"
: "${M6_PHASE10D_OUTPUT:?missing M6_PHASE10D_OUTPUT}"
mode="${M6_PHASE10D_MODE:-probe}"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
resume_args=()
case "$mode" in
  probe|train) ;;
  extend)
    resume_adapter="${M6_PHASE10D_RESUME_ADAPTER:-$study_root/phase10d_same_corpus_scale_v1/s35_d4/formal_v1/final_adapter}"
    test -d "$resume_adapter"
    resume_args=(--resume-adapter "$resume_adapter")
    ;;
  *) echo "invalid mode: $mode" >&2; exit 2 ;;
esac
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$M6_PHASE10D_D4_CORPUS/train.jsonl"
test -f "$M6_PHASE10D_D4_CORPUS/dev.jsonl"
test ! -e "$M6_PHASE10D_OUTPUT/training_report.json"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$python_bin" scripts/m6_phase10d_train_s35_d4.py \
  --mode "$mode" \
  --corpus-dir "$M6_PHASE10D_D4_CORPUS" \
  --base-model /data/share/model/Qwen3.5-35B-A3B \
  --output-dir "$M6_PHASE10D_OUTPUT" \
  "${resume_args[@]}"
