#!/usr/bin/env bash
# Phase10-D: Qwen3.5-4B on the frozen D35 corpus (2x2 completion cell).
#SBATCH --job-name=m6-p10d-s4-d35
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase10d_s4_d35_%j.out
#SBATCH --error=logs/m6_phase10d_s4_d35_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE10D_S4D35_OUTPUT:?missing M6_PHASE10D_S4D35_OUTPUT}"
mode="${M6_PHASE10D_S4D35_MODE:-probe}"
case "$mode" in probe|train) ;; *) echo "invalid mode: $mode" >&2; exit 2 ;; esac
corpus_dir="${M6_PHASE10D_S4D35_CORPUS:-$repo_root/outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v1/replay_weighted_corpus_v1}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$corpus_dir/train.jsonl"
test -f "$corpus_dir/dev.jsonl"
test -f "$corpus_dir/manifest.json"
test ! -e "$M6_PHASE10D_S4D35_OUTPUT/training_report.json"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$python_bin" scripts/m6_phase10d_train_s4_d35.py \
  --mode "$mode" \
  --corpus-dir "$corpus_dir" \
  --base-model /data/share/model/Qwen3.5-4B \
  --output-dir "$M6_PHASE10D_S4D35_OUTPUT"
