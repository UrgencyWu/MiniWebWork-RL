#!/usr/bin/env bash
# Recoverable M6 Raw/K4/K8 inference stage. This script never updates weights.
#SBATCH --job-name=m6-rollout
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m6_rollout_%j.out
#SBATCH --error=logs/m6_rollout_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_ROLLOUT_MODE:?set M6_ROLLOUT_MODE}"
: "${M6_ROLLOUT_ROLE:?set M6_ROLLOUT_ROLE}"
: "${M6_ROLLOUT_K:?set M6_ROLLOUT_K}"
: "${M6_ROLLOUT_OUTPUT:?set M6_ROLLOUT_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
mkdir -p "$M6_ROLLOUT_OUTPUT"

submit_timeout_successor() {
  trap - USR1
  if test -n "${SLURM_JOB_ID:-}" && test "${M6_DISABLE_SUCCESSOR:-0}" != "1"; then
    successor_job_id="$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
      --export="ALL,M6_EXPECTED_GIT_SHA=$M6_EXPECTED_GIT_SHA,M6_SERVICE_BASE_URL=${base_url:-${M6_SERVICE_BASE_URL:-http://127.0.0.1:44151}},M6_SPLIT_LOCK=${M6_SPLIT_LOCK:-},M6_ROLLOUT_MODE=$M6_ROLLOUT_MODE,M6_ROLLOUT_ROLE=$M6_ROLLOUT_ROLE,M6_ROLLOUT_K=$M6_ROLLOUT_K,M6_ROLLOUT_OUTPUT=$M6_ROLLOUT_OUTPUT,M6_ROLLOUT_ADAPTER=${M6_ROLLOUT_ADAPTER:-},M6_ROLLOUT_ITERATION=${M6_ROLLOUT_ITERATION:-0},M6_ROLLOUT_SEED=${M6_ROLLOUT_SEED:-20260812},M6_EVAL_IDENTITY=${M6_EVAL_IDENTITY:-},M6_TASK_ROSTER=${M6_TASK_ROSTER:-},M6_TASK_OFFSET=${M6_TASK_OFFSET:-0},M6_ROLLOUT_MAXIMUM_TOKENS=${M6_ROLLOUT_MAXIMUM_TOKENS:-}" \
      scripts/run_m6_rollout_job.sh)"
    printf '%s\n' "$successor_job_id" > "$M6_ROLLOUT_OUTPUT/successor_job_id"
    echo "timeout_successor_job_id=$successor_job_id"
  fi
  exit 99
}
trap submit_timeout_successor USR1

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTORCH_CUDA_ALLOC_CONF
adapter_args=()
test -z "${M6_ROLLOUT_ADAPTER:-}" || adapter_args=(--adapter "$M6_ROLLOUT_ADAPTER")
mode_args=()
if test "$M6_ROLLOUT_MODE" = "rl_collection"; then
  : "${M6_TASK_ROSTER:?set M6_TASK_ROSTER for RL collection}"
  : "${M6_ROLLOUT_MAXIMUM_TOKENS:?set M6_ROLLOUT_MAXIMUM_TOKENS for RL collection}"
  mode_args=(
    --task-roster "$M6_TASK_ROSTER"
    --task-offset "${M6_TASK_OFFSET:-0}"
    --maximum-tasks 1
    --max-model-turns 6
    --max-environment-steps 6
    --maximum-action-tokens "$M6_ROLLOUT_MAXIMUM_TOKENS"
    --concurrent-groups 1
  )
fi
if test -n "${M6_SERVICE_BASE_URL:-}"; then
  base_url="$M6_SERVICE_BASE_URL"
elif test -f "$repo_root/outputs/m6_monotonic_posttraining_v1/service_base_url"; then
  IFS= read -r base_url < "$repo_root/outputs/m6_monotonic_posttraining_v1/service_base_url"
else
  base_url="http://127.0.0.1:44151"
fi
split_lock="${M6_SPLIT_LOCK:-$repo_root/outputs/m6_monotonic_posttraining_v1/locks/m6_webshop_split_v1.json}"
curl --fail --silent --show-error --max-time 30 "$base_url/health" >/dev/null
curl --fail --silent --show-error --max-time 120 -X POST \
  -H 'Content-Type: application/json' -d '{"goal_index":1000}' "$base_url/reset" >/dev/null

/opt/slurm/slurm.25.05/bin/srun --ntasks=1 "$python_bin" scripts/m6_collect_policy_success.py \
  --mode "$M6_ROLLOUT_MODE" --role "$M6_ROLLOUT_ROLE" --k "$M6_ROLLOUT_K" \
  --split-lock "$split_lock" \
  --goals outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json \
  --output-dir "$M6_ROLLOUT_OUTPUT" --iteration-index "${M6_ROLLOUT_ITERATION:-0}" \
  --seed "${M6_ROLLOUT_SEED:-20260812}" \
  --base-url "$base_url" \
  "${mode_args[@]}" \
  "${adapter_args[@]}"

if test "$M6_ROLLOUT_MODE" = "evaluation"; then
  : "${M6_EVAL_IDENTITY:?set raw, mini_sft, or mini_rl for evaluation}"
  /opt/slurm/slurm.25.05/bin/srun --ntasks=1 "$python_bin" scripts/m6_closed_loop_eval.py \
    --identity "$M6_EVAL_IDENTITY" --groups-dir "$M6_ROLLOUT_OUTPUT/groups" \
    --collection-report "$M6_ROLLOUT_OUTPUT/collection_report.json" --split-lock "$split_lock" \
    --output "$M6_ROLLOUT_OUTPUT/identity_report.json"
fi
