#!/usr/bin/env bash
# Phase10-C scale: inference-only Raw Qwen3.5-35B exploration, 128 tasks x K8.
#SBATCH --job-name=m6-p10c-q35-scale
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/m6_phase10c_teacher_exploration_scale_%j.out
#SBATCH --error=logs/m6_phase10c_teacher_exploration_scale_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE10C_SCALE_ROSTER:?missing M6_PHASE10C_SCALE_ROSTER}"
: "${M6_PHASE10C_SCALE_ROSTER_PRODUCER_GIT_SHA:?missing M6_PHASE10C_SCALE_ROSTER_PRODUCER_GIT_SHA}"
: "${M6_PHASE10C_SCALE_OUTPUT:?missing M6_PHASE10C_SCALE_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$M6_PHASE10C_SCALE_ROSTER"
test ! -e "$M6_PHASE10C_SCALE_OUTPUT/collection_report.json"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
mkdir -p "$M6_PHASE10C_SCALE_OUTPUT"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase10c_teacher_exploration_scale --role train --k 8 --seed 20260863 \
  --split-lock "$study_root/locks/m6_webshop_split_v1.json" \
  --goals "$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json" \
  --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE10C_SCALE_ROSTER" \
  --task-roster-producer-git-sha "$M6_PHASE10C_SCALE_ROSTER_PRODUCER_GIT_SHA" \
  --base-model /data/share/model/Qwen3.5-35B-A3B \
  --base-model-manifest "$study_root/phase10b_multi_specialist_opd_v1/base_model_manifests/S_match.json" \
  --tensor-parallel-size 2 \
  --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$M6_PHASE10C_SCALE_OUTPUT"
