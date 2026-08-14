#!/usr/bin/env bash
# Phase4: one bounded larger-model teacher K4 probe on student all-failure tasks.
#SBATCH --job-name=m6-p4-teacher-probe
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase4_teacher_probe_%j.out
#SBATCH --error=logs/m6_phase4_teacher_probe_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
teacher_root="${M6_TEACHER_ROOT:-$study_root/phase4_rl_data_v1/teacher_probe_qwen35_9b_v1}"
teacher_model="${M6_TEACHER_MODEL:-/data/share/model/Qwen3.5-9B}"
output="$teacher_root/collection"
test -f "$teacher_root/prep/teacher_roster.json"
test -f "$teacher_root/prep/teacher_base_model_manifest.json"
test ! -e "$output/collection_report.json"
mkdir -p "$output"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase4_teacher_probe --role train --k 4 --seed 20260826 \
  --split-lock "$study_root/locks/m6_webshop_split_v1.json" \
  --goals "$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json" \
  --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$teacher_root/prep/teacher_roster.json" \
  --task-roster-producer-git-sha "$M6_EXPECTED_GIT_SHA" \
  --base-model "$teacher_model" \
  --base-model-manifest "$teacher_root/prep/teacher_base_model_manifest.json" \
  --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$output"
