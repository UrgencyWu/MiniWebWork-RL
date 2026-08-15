#!/usr/bin/env bash
# Phase10-C: query-only Qwen3.5-35B public-data feasibility smoke.
#SBATCH --job-name=m6-p10c-q35-query
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/m6_phase10c_qwen35_query_%j.out
#SBATCH --error=logs/m6_phase10c_qwen35_query_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE10C_OUTPUT:?missing M6_PHASE10C_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$M6_PHASE10C_OUTPUT"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
phase_root="$repo_root/outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase10c_qwen35_query_smoke.py \
  --goals "$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json" \
  --phase10c-split "$phase_root/split_lock.json" \
  --base-model /data/share/model/Qwen3.5-35B-A3B \
  --base-model-manifest "$repo_root/outputs/m6_monotonic_posttraining_v1/phase10b_multi_specialist_opd_v1/base_model_manifests/S_match.json" \
  --base-url "$M6_SERVICE_BASE_URL" \
  --producer-git-sha "$M6_EXPECTED_GIT_SHA" \
  --output-dir "$M6_PHASE10C_OUTPUT"
