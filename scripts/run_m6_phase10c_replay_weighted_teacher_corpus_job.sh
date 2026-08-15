#!/usr/bin/env bash
# Phase10-C: CPU-only fresh replay and capability-weighted Raw35 SFT corpus.
#SBATCH --job-name=m6-p10c-q35-replay
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/m6_phase10c_q35_replay_%j.out
#SBATCH --error=logs/m6_phase10c_q35_replay_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE10C_REPLAY_OUTPUT:?missing M6_PHASE10C_REPLAY_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$M6_PHASE10C_REPLAY_OUTPUT/replay_report.json"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase10c_replay_weighted_teacher_corpus.py \
  --collection-root "$study_root/phase10c_qwen35_sft_specialist_opd_v1/teacher_self_exploration/raw35_collection_v1" \
  --collection-root "$study_root/phase10c_qwen35_sft_specialist_opd_v1/teacher_self_exploration_scale/raw35_collection_k8_v1" \
  --phase10c-split "$study_root/phase10c_qwen35_sft_specialist_opd_v1/split_lock.json" \
  --goals "$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json" \
  --base-model /data/share/model/Qwen3.5-35B-A3B \
  --base-url "$M6_SERVICE_BASE_URL" --workers 8 \
  --output-dir "$M6_PHASE10C_REPLAY_OUTPUT"
