#!/usr/bin/env bash
# Phase10: matched student/teacher K2 suffix smoke from eight frozen historical states.
#SBATCH --job-name=m6-p10-state-smoke
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --output=logs/m6_phase10_state_suffix_smoke_%A_%a.out
#SBATCH --error=logs/m6_phase10_state_suffix_smoke_%A_%a.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE10_ROSTER:?missing M6_PHASE10_ROSTER}"
: "${M6_PHASE10_MANIFEST:?missing M6_PHASE10_MANIFEST}"
: "${M6_PHASE10_ROSTER_PRODUCER_GIT_SHA:?missing M6_PHASE10_ROSTER_PRODUCER_GIT_SHA}"
: "${M6_PHASE10_TEACHER_MANIFEST:?missing M6_PHASE10_TEACHER_MANIFEST}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase10_root="$study_root/phase10_student_state_teacher_correction_v1"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
student_model=/data/share/model/Qwen3.5-4B
student_manifest="$repo_root/data/m4_long_horizon_base_model_manifest_v1.json"
student_adapter="$study_root/mini/pilot_sft/final_adapter"
teacher_model=/data/share/model/Qwen3.6-35B-A3B-FP8

case "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}" in
  0)
    identity=student
    base_model="$student_model"
    base_model_manifest="$student_manifest"
    output="${M6_PHASE10_SMOKE_ROOT:-$phase10_root/smoke}/student_collection"
    adapter_args=(--adapter "$student_adapter")
    ;;
  1)
    identity=teacher
    base_model="$teacher_model"
    base_model_manifest="$M6_PHASE10_TEACHER_MANIFEST"
    output="${M6_PHASE10_SMOKE_ROOT:-$phase10_root/smoke}/teacher_collection"
    adapter_args=()
    ;;
  *)
    echo "invalid Phase10 smoke array index" >&2
    exit 2
    ;;
esac

test -f "$M6_PHASE10_ROSTER"
test -f "$M6_PHASE10_MANIFEST"
test -f "$base_model_manifest"
test -d "$base_model"
if [[ "$identity" == student ]]; then
  test -d "$student_adapter"
fi
test ! -e "$output/collection_report.json"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
mkdir -p "$output"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase10_state_suffix_smoke --phase10-smoke-identity "$identity" \
  --role train --k 2 --seed 20260841 \
  --split-lock "$split_lock" --goals "$goals" --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE10_ROSTER" \
  --task-roster-producer-git-sha "$M6_PHASE10_ROSTER_PRODUCER_GIT_SHA" \
  --state-correction-manifest "$M6_PHASE10_MANIFEST" \
  --base-model "$base_model" --base-model-manifest "$base_model_manifest" \
  "${adapter_args[@]}" \
  --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$output"
