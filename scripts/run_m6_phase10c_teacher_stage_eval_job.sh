#!/usr/bin/env bash
# Phase10-C: paired Raw35/SFT35/SFT4 full-environment evaluation arm.
#SBATCH --job-name=m6-p10c-stage-eval
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/m6_phase10c_stage_eval_%j.out
#SBATCH --error=logs/m6_phase10c_stage_eval_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE10C_EVAL_IDENTITY:?missing M6_PHASE10C_EVAL_IDENTITY}"
: "${M6_PHASE10C_EVAL_STAGE:?missing M6_PHASE10C_EVAL_STAGE}"
: "${M6_PHASE10C_EVAL_ROSTER:?missing M6_PHASE10C_EVAL_ROSTER}"
: "${M6_PHASE10C_ROSTER_PRODUCER_GIT_SHA:?missing M6_PHASE10C_ROSTER_PRODUCER_GIT_SHA}"
: "${M6_PHASE10C_EVAL_OUTPUT:?missing M6_PHASE10C_EVAL_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$M6_PHASE10C_EVAL_ROSTER"
test ! -e "$M6_PHASE10C_EVAL_OUTPUT/collection_report.json"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null

python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase_root="$study_root/phase10c_qwen35_sft_specialist_opd_v1"
adapter_args=()
case "$M6_PHASE10C_EVAL_IDENTITY" in
  raw35)
    base_model=/data/share/model/Qwen3.5-35B-A3B
    base_manifest="$study_root/phase10b_multi_specialist_opd_v1/base_model_manifests/S_match.json"
    tp=2
    ;;
  sft35)
    base_model=/data/share/model/Qwen3.5-35B-A3B
    base_manifest="$study_root/phase10b_multi_specialist_opd_v1/base_model_manifests/S_match.json"
    adapter_args=(--adapter "$phase_root/teacher_self_sft_v1/weighted_lora_v1/final_adapter")
    # vLLM 0.17 cannot activate packed Qwen3.5 LoRA slices under TP=2.
    # The 35B-A3B BF16 checkpoint fits one 102GB device at the frozen 0.8
    # engine memory fraction; sampling semantics stay identical to Raw35.
    tp=1
    ;;
  sft4)
    base_model=/data/share/model/Qwen3.5-4B
    base_manifest="$repo_root/data/m4_long_horizon_base_model_manifest_v1.json"
    adapter_args=(--adapter "$study_root/mini/pilot_sft/final_adapter")
    tp=1
    ;;
  *)
    echo "invalid Phase10-C evaluation identity: $M6_PHASE10C_EVAL_IDENTITY" >&2
    exit 2
    ;;
esac
case "$M6_PHASE10C_EVAL_STAGE" in
  dev) seed=20260864 ;;
  qualification) seed=20260865 ;;
  *) echo "invalid Phase10-C evaluation stage: $M6_PHASE10C_EVAL_STAGE" >&2; exit 2 ;;
esac
test -f "$base_manifest"
test -d "$base_model"
mkdir -p "$M6_PHASE10C_EVAL_OUTPUT"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase10c_teacher_stage_evaluation --role train --k 4 --seed "$seed" \
  --split-lock "$study_root/locks/m6_webshop_split_v1.json" \
  --goals "$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json" \
  --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE10C_EVAL_ROSTER" \
  --task-roster-producer-git-sha "$M6_PHASE10C_ROSTER_PRODUCER_GIT_SHA" \
  --phase10c-evaluation-identity "$M6_PHASE10C_EVAL_IDENTITY" \
  --base-model "$base_model" --base-model-manifest "$base_manifest" \
  "${adapter_args[@]}" --tensor-parallel-size "$tp" \
  --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$M6_PHASE10C_EVAL_OUTPUT"
