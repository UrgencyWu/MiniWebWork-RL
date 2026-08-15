#!/usr/bin/env bash
# Phase10-B: one paired 24-task full-horizon Student or Specialist qualification arm.
#SBATCH --job-name=m6-p10b-qual
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase10b_qualification_%j.out
#SBATCH --error=logs/m6_phase10b_qualification_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE10B_QUALIFICATION_IDENTITY:?missing M6_PHASE10B_QUALIFICATION_IDENTITY}"
: "${M6_PHASE10B_COMPARISON_SPECIALIST:?missing M6_PHASE10B_COMPARISON_SPECIALIST}"
: "${M6_PHASE10B_QUALIFICATION_ROSTER:?missing M6_PHASE10B_QUALIFICATION_ROSTER}"
: "${M6_PHASE10B_ROSTER_PRODUCER_GIT_SHA:?missing M6_PHASE10B_ROSTER_PRODUCER_GIT_SHA}"
: "${M6_PHASE10B_QUALIFICATION_OUTPUT:?missing M6_PHASE10B_QUALIFICATION_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase_root="$study_root/phase10b_multi_specialist_opd_v1"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
student_adapter="$study_root/mini/pilot_sft/final_adapter"
identity="$M6_PHASE10B_QUALIFICATION_IDENTITY"
specialist="$M6_PHASE10B_COMPARISON_SPECIALIST"
adapter_args=()
tp=1
case "$identity:$specialist" in
  student:S_nav|student:S_match|student:S_finish)
    base_model=/data/share/model/Qwen3.5-4B
    base_manifest="$repo_root/data/m4_long_horizon_base_model_manifest_v1.json"
    adapter_args=(--adapter "$student_adapter")
    ;;
  S_nav:S_nav)
    base_model=/data/share/model/Qwen3.5-9B
    base_manifest="$phase_root/base_model_manifests/S_nav.json"
    ;;
  S_match:S_match)
    base_model=/data/share/model/Qwen3.5-35B-A3B
    base_manifest="$phase_root/base_model_manifests/S_match.json"
    tp=2
    ;;
  S_finish:S_finish)
    base_model=/data/share/model/Qwen3.6-35B-A3B-FP8
    base_manifest="$phase_root/base_model_manifests/S_finish.json"
    ;;
  *)
    echo "invalid Phase10-B qualification pairing: $identity:$specialist" >&2
    exit 2
    ;;
esac
test -f "$M6_PHASE10B_QUALIFICATION_ROSTER"
test -f "$base_manifest"
test -d "$base_model"
test ! -e "$M6_PHASE10B_QUALIFICATION_OUTPUT/collection_report.json"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
mkdir -p "$M6_PHASE10B_QUALIFICATION_OUTPUT"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase10b_specialist_qualification --role train --k 4 --seed 20260851 \
  --split-lock "$split_lock" --goals "$goals" --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE10B_QUALIFICATION_ROSTER" \
  --task-roster-producer-git-sha "$M6_PHASE10B_ROSTER_PRODUCER_GIT_SHA" \
  --phase10b-qualification-identity "$identity" \
  --base-model "$base_model" --base-model-manifest "$base_manifest" \
  "${adapter_args[@]}" \
  --tensor-parallel-size "$tp" \
  --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$M6_PHASE10B_QUALIFICATION_OUTPUT"
