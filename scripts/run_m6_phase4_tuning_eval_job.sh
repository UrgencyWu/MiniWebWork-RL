#!/usr/bin/env bash
# Fresh paired K4/18-15 tuning evaluation for one Raw/SFT/RL identity.
#SBATCH --job-name=m6-p4-tuning-eval
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%1
#SBATCH --output=logs/m6_phase4_tuning_eval_%A_%a.out
#SBATCH --error=logs/m6_phase4_tuning_eval_%A_%a.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_TUNING_ROSTER_DIR:?set M6_TUNING_ROSTER_DIR}"
: "${M6_TUNING_OUTPUT:?set M6_TUNING_OUTPUT}"
: "${M6_EVAL_IDENTITY:?set raw, sft, or rl}"
: "${M6_SERVICE_BASE_URL:?set M6_SERVICE_BASE_URL}"
tuning_roster_producer_git_sha="${M6_TUNING_ROSTER_PRODUCER_GIT_SHA:-$M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

case "$M6_EVAL_IDENTITY" in
  raw) adapter_args=() ;;
  sft|rl)
    : "${M6_EVAL_ADAPTER:?non-Raw evaluation requires M6_EVAL_ADAPTER}"
    test -d "$M6_EVAL_ADAPTER"
    adapter_args=(--adapter "$M6_EVAL_ADAPTER")
    ;;
  *) echo "unsupported identity: $M6_EVAL_IDENTITY" >&2; exit 2 ;;
esac
case "${SLURM_ARRAY_TASK_ID:?missing array id}" in
  0) partition=a; seed=20260829 ;;
  1) partition=b; seed=20260830 ;;
  *) echo "unexpected tuning array index" >&2; exit 2 ;;
esac

python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
roster="$M6_TUNING_ROSTER_DIR/roster_${partition}.json"
output="$M6_TUNING_OUTPUT/$M6_EVAL_IDENTITY/partition_${partition}"
test -f "$roster"
test ! -e "$output/collection_report.json"
mkdir -p "$output"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase4_tuning_evaluation --role train --k 4 --seed "$seed" \
  --split-lock "$split_lock" --goals "$goals" --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$roster" --task-roster-producer-git-sha "$tuning_roster_producer_git_sha" \
  --max-model-turns 18 --max-environment-steps 15 --concurrent-groups 4 \
  --output-dir "$output" "${adapter_args[@]}"
