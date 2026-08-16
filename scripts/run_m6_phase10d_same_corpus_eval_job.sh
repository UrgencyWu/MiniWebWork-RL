#!/usr/bin/env bash
# Phase10-D: paired SFT4(D4) vs SFT35(D4) full-environment evaluation arm.
# The SFT35(D4) arm merges the audited formal adapter into Raw35 first because
# vLLM 0.17 has no compatible Qwen3.5 LoRA mapper; the merge is guarded by the
# same expected-LoRA/expected-target gates as Job2378.
#SBATCH --job-name=m6-p10d-same-corpus-eval
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/m6_phase10d_same_corpus_eval_%j.out
#SBATCH --error=logs/m6_phase10d_same_corpus_eval_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE10D_EVAL_IDENTITY:?missing M6_PHASE10D_EVAL_IDENTITY}"
: "${M6_PHASE10D_EVAL_ROSTER:?missing M6_PHASE10D_EVAL_ROSTER}"
: "${M6_PHASE10D_ROSTER_PRODUCER_GIT_SHA:?missing M6_PHASE10D_ROSTER_PRODUCER_GIT_SHA}"
: "${M6_PHASE10D_EVAL_OUTPUT:?missing M6_PHASE10D_EVAL_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$M6_PHASE10D_EVAL_ROSTER"
test ! -e "$M6_PHASE10D_EVAL_OUTPUT/collection_report.json"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null

# This shared node's Slurm device indices do not always match physical GPU
# occupancy, so a submission-time verified device list may be pinned via
# M6_CUDA_VISIBLE_DEVICES.  Either way every device used below must pass a
# physical free-memory gate before any model memory is touched.
min_free_mib="${M6_MIN_FREE_MIB_PER_GPU:-60000}"
# sbatch --export cannot carry commas inside values, so colons or spaces in
# M6_CUDA_VISIBLE_DEVICES are normalized to the comma list vLLM expects.
selected_devices="$(printf '%s' "${M6_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}" | sed 's/[:, ]/,/g')"
for dev in ${selected_devices//,/ }; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$dev")"
  if [ "$free_mib" -lt "$min_free_mib" ]; then
    echo "refusing to start: GPU $dev has ${free_mib}MiB free (< ${min_free_mib}MiB)" >&2
    exit 3
  fi
done
export CUDA_VISIBLE_DEVICES="$selected_devices"

python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase10d_root="$study_root/phase10d_same_corpus_scale_v1"
adapter_args=()
case "$M6_PHASE10D_EVAL_IDENTITY" in
  sft4)
    base_model=/data/share/model/Qwen3.5-4B
    base_manifest="$repo_root/data/m4_long_horizon_base_model_manifest_v1.json"
    adapter_args=(--adapter "$study_root/mini/pilot_sft/final_adapter")
    tp=1
    ;;
  sft35_d4)
    adapter="$phase10d_root/s35_d4/formal_v1/final_adapter"
    merged_root="$phase10d_root/s35_d4/formal_v1/merged_model_v1"
    base_model="$merged_root/model"
    base_manifest="$merged_root/model_manifest.json"
    merge_report="$merged_root/merge_report.json"
    test -d "$adapter"
    if [ ! -d "$base_model" ]; then
      merge_device="$(echo "$selected_devices" | cut -d, -f1)"
      CUDA_VISIBLE_DEVICES="$merge_device" "$python_bin" scripts/m6_phase10c_merge_teacher_lora.py \
        --base-model /data/share/model/Qwen3.5-35B-A3B \
        --base-model-manifest "$study_root/phase10b_multi_specialist_opd_v1/base_model_manifests/S_match.json" \
        --adapter "$adapter" \
        --output "$base_model" \
        --manifest "$base_manifest" \
        --report "$merge_report" \
        --expected-lora-r 16 \
        --expected-lora-alpha 32 \
        --expected-target-count 310 \
        --expected-git-sha "$M6_EXPECTED_GIT_SHA"
    fi
    test -f "$base_manifest"
    test -f "$merge_report"
    tp=2
    ;;
  *)
    echo "invalid Phase10-D evaluation identity: $M6_PHASE10D_EVAL_IDENTITY" >&2
    exit 2
    ;;
esac
seed=20260867
test -f "$base_manifest"
test -d "$base_model"
mkdir -p "$M6_PHASE10D_EVAL_OUTPUT"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
echo "selected_devices=$selected_devices"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase10c_teacher_stage_evaluation --role train --k 4 --seed "$seed" \
  --split-lock "$study_root/locks/m6_webshop_split_v1.json" \
  --goals "$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json" \
  --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE10D_EVAL_ROSTER" \
  --task-roster-producer-git-sha "$M6_PHASE10D_ROSTER_PRODUCER_GIT_SHA" \
  --phase10c-evaluation-identity "$M6_PHASE10D_EVAL_IDENTITY" \
  --base-model "$base_model" --base-model-manifest "$base_manifest" \
  "${adapter_args[@]}" --tensor-parallel-size "$tp" \
  --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$M6_PHASE10D_EVAL_OUTPUT"
