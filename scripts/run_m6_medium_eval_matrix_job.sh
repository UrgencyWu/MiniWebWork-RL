#!/usr/bin/env bash
# Frozen formal-dev K4 evaluation for Raw, shared SFT and six medium RL adapters.
#SBATCH --job-name=m6-medium-eval
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%6
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m6_medium_eval_%A_%a.out
#SBATCH --error=logs/m6_medium_eval_%A_%a.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_EVAL_ROOT:?set M6_EVAL_ROOT}"
: "${M6_EVAL_PLAN:?set M6_EVAL_PLAN}"
: "${M6_SERVICE_BASE_URL:?set M6_SERVICE_BASE_URL}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
sft_adapter="$study_root/mini/pilot_sft/final_adapter"
online_root="$study_root/medium/online_v1"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

case "${SLURM_ARRAY_TASK_ID:?missing Slurm array task id}" in
  0) label=raw; identity=raw; adapter="" ;;
  1) label=sft; identity=mini_sft; adapter="$sft_adapter" ;;
  2) label=multi_turn_grpo_seed_20260812; identity=mini_rl; method=multi_turn_grpo; seed=20260812 ;;
  3) label=anchor_gigpo_seed_20260812; identity=mini_rl; method=anchor_gigpo; seed=20260812 ;;
  4) label=multi_turn_grpo_seed_20260813; identity=mini_rl; method=multi_turn_grpo; seed=20260813 ;;
  5) label=anchor_gigpo_seed_20260813; identity=mini_rl; method=anchor_gigpo; seed=20260813 ;;
  6) label=multi_turn_grpo_seed_20260814; identity=mini_rl; method=multi_turn_grpo; seed=20260814 ;;
  7) label=anchor_gigpo_seed_20260814; identity=mini_rl; method=anchor_gigpo; seed=20260814 ;;
  *) echo "unsupported evaluation index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

output="$M6_EVAL_ROOT/$label"
mkdir -p "$output"

submit_timeout_successor() {
  trap - USR1
  if test -n "${SLURM_JOB_ID:-}" && test "${M6_DISABLE_SUCCESSOR:-0}" != "1"; then
    successor="$($slurm_bin/sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
      --array="$SLURM_ARRAY_TASK_ID" --cpus-per-task=4 --mem=24G --gres=gpu:1 --time=24:00:00 \
      --export="ALL,M6_EXPECTED_GIT_SHA=$M6_EXPECTED_GIT_SHA,M6_EVAL_ROOT=$M6_EVAL_ROOT,M6_EVAL_PLAN=$M6_EVAL_PLAN,M6_SERVICE_BASE_URL=$M6_SERVICE_BASE_URL" \
      scripts/run_m6_medium_eval_matrix_job.sh)"
    printf '%s\n' "$successor" > "$output/successor_job_id"
  fi
  exit 99
}
trap submit_timeout_successor USR1

if test "$identity" = mini_rl; then
  adapter="$($python_bin scripts/m6_resolve_medium_adapter.py \
    --run-root "$online_root/seed_${seed}/${method}" --method "$method")"
fi

"$python_bin" - "$M6_EVAL_PLAN" "$label" <<'PY'
import json, sys
from pathlib import Path
plan=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if plan.get("schema_version") != "m6_medium_eval_plan_v1" or plan.get("role") != "formal_dev":
    raise ValueError("M6 medium evaluation plan drift")
if sys.argv[2] not in plan.get("identities", []):
    raise ValueError("M6 medium evaluation identity is not frozen")
if plan.get("task_count") != 500 or plan.get("group_size") != 4 or plan.get("rollout_seed") != 20260815:
    raise ValueError("M6 medium evaluation budget drift")
if plan.get("isolation") != {"mini_dev_reuse_for_selection": False, "promotion_opened": False, "holdout_opened": False}:
    raise ValueError("M6 medium evaluation isolation drift")
PY

curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
curl --fail --silent --show-error --max-time 120 -X POST -H 'Content-Type: application/json' \
  -d '{"goal_index":1000}' "$M6_SERVICE_BASE_URL/reset" >/dev/null

adapter_args=()
test -z "$adapter" || adapter_args=(--adapter "$adapter")
"$slurm_bin/srun" --ntasks=1 "$python_bin" scripts/m6_collect_policy_success.py \
  --mode evaluation --role formal_dev --k 4 --seed 20260815 \
  --split-lock "$split_lock" --goals "$goals" --base-url "$M6_SERVICE_BASE_URL" \
  --max-model-turns 18 --max-environment-steps 15 --concurrent-groups 4 \
  --output-dir "$output" "${adapter_args[@]}"

"$python_bin" scripts/m6_closed_loop_eval.py \
  --identity "$identity" --role formal_dev --groups-dir "$output/groups" \
  --collection-report "$output/collection_report.json" --split-lock "$split_lock" \
  --output "$output/identity_report.json"
