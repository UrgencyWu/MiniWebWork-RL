#!/usr/bin/env bash
# Phase10-B: exact Student-token-prefix logits alignment across Student and three Specialists.
#SBATCH --job-name=m6-p10b-logits
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
#SBATCH --output=logs/m6_phase10b_logit_alignment_%A_%a.out
#SBATCH --error=logs/m6_phase10b_logit_alignment_%A_%a.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE10B_MODEL_MANIFEST:?missing M6_PHASE10B_MODEL_MANIFEST}"
: "${M6_PHASE10B_LOGIT_ROOT:?missing M6_PHASE10B_LOGIT_ROOT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$M6_PHASE10B_MODEL_MANIFEST"

student_model=/data/share/model/Qwen3.5-4B
student_adapter="$repo_root/outputs/m6_monotonic_posttraining_v1/mini/pilot_sft/final_adapter"
extra_pythonpath=""
case "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}" in
  0)
    identity=student
    model_path=/data/share/model/Qwen3.5-4B
    adapter_args=(--adapter "$student_adapter")
    ;;
  1)
    identity=S_nav
    model_path=/data/share/model/Qwen3.5-9B
    adapter_args=()
    ;;
  2)
    identity=S_match
    model_path=/data/share/model/Qwen3.5-35B-A3B
    adapter_args=()
    ;;
  3)
    identity=S_finish
    model_path=/data/share/model/Qwen3.6-35B-A3B-FP8
    extra_pythonpath="$repo_root/outputs/m6_monotonic_posttraining_v1/phase10b_multi_specialist_opd_v1/runtime_deps/kernels_0_15_2"
    adapter_args=()
    ;;
  *)
    echo "invalid Phase10-B logit probe array index" >&2
    exit 2
    ;;
esac

output="$M6_PHASE10B_LOGIT_ROOT/models/$identity.json"
test -d "$model_path"
test ! -e "$output"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
if [[ -n "$extra_pythonpath" ]]; then
  test -d "$extra_pythonpath/kernels"
  export PYTHONPATH="$extra_pythonpath:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
fi
"$python_bin" scripts/m6_phase10b_logit_alignment_probe.py model \
  --identity "$identity" \
  --model-path "$model_path" \
  --student-tokenizer "$student_model" \
  "${adapter_args[@]}" \
  --prompt "$repo_root/prompts/webshop_agent_v1_compact.txt" \
  --model-manifest "$M6_PHASE10B_MODEL_MANIFEST" \
  --output "$output"
