#!/usr/bin/env bash
# Development-only K8 evaluation on the union of tasks that received M6.2 updates.
#SBATCH --job-name=m6-p1-seen
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%3
#SBATCH --output=logs/m6_phase1_seen_%A_%a.out
#SBATCH --error=logs/m6_phase1_seen_%A_%a.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE1_ROOT:?missing M6_PHASE1_ROOT}"
: "${M6_PHASE1_ROSTER:?missing M6_PHASE1_ROSTER}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_ROSTER_PRODUCER_GIT_SHA:?missing M6_ROSTER_PRODUCER_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
sft_adapter="$study_root/mini/pilot_sft/final_adapter"
online_root="$study_root/medium/online_v1"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

case "${SLURM_ARRAY_TASK_ID:?missing array id}" in
  0) label=sft; adapter="$sft_adapter" ;;
  1) label=multi_turn_grpo_seed_20260812; adapter="$($python_bin scripts/m6_resolve_medium_adapter.py --run-root "$online_root/seed_20260812/multi_turn_grpo" --method multi_turn_grpo)" ;;
  2) label=anchor_gigpo_seed_20260812; adapter="$($python_bin scripts/m6_resolve_medium_adapter.py --run-root "$online_root/seed_20260812/anchor_gigpo" --method anchor_gigpo)" ;;
  3) label=multi_turn_grpo_seed_20260813; adapter="$($python_bin scripts/m6_resolve_medium_adapter.py --run-root "$online_root/seed_20260813/multi_turn_grpo" --method multi_turn_grpo)" ;;
  4) label=anchor_gigpo_seed_20260813; adapter="$($python_bin scripts/m6_resolve_medium_adapter.py --run-root "$online_root/seed_20260813/anchor_gigpo" --method anchor_gigpo)" ;;
  5) label=multi_turn_grpo_seed_20260814; adapter="$($python_bin scripts/m6_resolve_medium_adapter.py --run-root "$online_root/seed_20260814/multi_turn_grpo" --method multi_turn_grpo)" ;;
  6) label=anchor_gigpo_seed_20260814; adapter="$($python_bin scripts/m6_resolve_medium_adapter.py --run-root "$online_root/seed_20260814/anchor_gigpo" --method anchor_gigpo)" ;;
  7) label=lr_1e_6; adapter="$M6_PHASE1_ROOT/gpu_probe/lr_1e_6/adapter" ;;
  8) label=lr_3e_6; adapter="$M6_PHASE1_ROOT/gpu_probe/lr_3e_6/adapter" ;;
  9) label=lr_1e_5; adapter="$M6_PHASE1_ROOT/gpu_probe/lr_1e_5/adapter" ;;
esac
if test "${SLURM_ARRAY_TASK_ID}" -ge 7; then
  test -f "$M6_PHASE1_ROOT/gpu_probe/report.json"
  expected_adapter_sha="$($python_bin - "$M6_PHASE1_ROOT/gpu_probe/report.json" "$label" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
rows={row["label"]: row for row in value["learning_rate_probe"]}
print(rows[sys.argv[2]]["output_adapter_sha256"])
PY
)"
  observed_adapter_sha="$($python_bin - "$adapter" <<'PY'
import sys
from pathlib import Path
from miniwebwork.long_horizon_rl.contracts import directory_sha256
print(directory_sha256(Path(sys.argv[1])))
PY
)"
  test "$expected_adapter_sha" = "$observed_adapter_sha"
fi
output="$M6_PHASE1_ROOT/seen_task_eval/$label"
mkdir -p "$output"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode diagnostic_evaluation --role mini_train --k 8 --seed 20260818 \
  --split-lock "$split_lock" --goals "$goals" --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$M6_PHASE1_ROSTER" --task-roster-producer-git-sha "$M6_ROSTER_PRODUCER_GIT_SHA" \
  --adapter "$adapter" \
  --max-model-turns 6 --max-environment-steps 6 --concurrent-groups 4 \
  --output-dir "$output"
"$python_bin" scripts/m6_phase1_seen_eval.py --label "$label" \
  --groups-dir "$output/groups" --collection-report "$output/collection_report.json" \
  --output "$output/seen_task_report.json"
