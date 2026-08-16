#!/usr/bin/env bash
#SBATCH --job-name=m6-p10c-merge-sft35
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase10c_merge_sft35_%j.out
#SBATCH --error=logs/m6_phase10c_merge_sft35_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE10C_MERGED_MODEL:?missing M6_PHASE10C_MERGED_MODEL}"
: "${M6_PHASE10C_MERGED_MANIFEST:?missing M6_PHASE10C_MERGED_MANIFEST}"
: "${M6_PHASE10C_MERGE_REPORT:?missing M6_PHASE10C_MERGE_REPORT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase_root="$study_root/phase10c_qwen35_sft_specialist_opd_v1"
adapter="${M6_PHASE10C_MERGE_ADAPTER:-$phase_root/teacher_self_sft_v1/weighted_lora_v1/final_adapter}"
expected_r="${M6_PHASE10C_MERGE_EXPECTED_R:-8}"
expected_alpha="${M6_PHASE10C_MERGE_EXPECTED_ALPHA:-16}"
expected_targets="${M6_PHASE10C_MERGE_EXPECTED_TARGETS:-190}"
test -d "$adapter"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase10c_merge_teacher_lora.py \
  --base-model /data/share/model/Qwen3.5-35B-A3B \
  --base-model-manifest "$study_root/phase10b_multi_specialist_opd_v1/base_model_manifests/S_match.json" \
  --adapter "$adapter" \
  --output "$M6_PHASE10C_MERGED_MODEL" \
  --manifest "$M6_PHASE10C_MERGED_MANIFEST" \
  --report "$M6_PHASE10C_MERGE_REPORT" \
  --expected-lora-r "$expected_r" \
  --expected-lora-alpha "$expected_alpha" \
  --expected-target-count "$expected_targets" \
  --expected-git-sha "$M6_EXPECTED_GIT_SHA"
