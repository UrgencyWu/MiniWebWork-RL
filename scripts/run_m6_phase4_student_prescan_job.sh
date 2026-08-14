#!/usr/bin/env bash
# Phase4: two independent SFT-student K4 full-horizon prescan partitions.
#SBATCH --job-name=m6-p4-student-prescan
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --output=logs/m6_phase4_student_prescan_%A_%a.out
#SBATCH --error=logs/m6_phase4_student_prescan_%A_%a.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
: "${M6_PHASE4_ROSTER_DIR:?missing M6_PHASE4_ROSTER_DIR}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase4_root="${M6_PHASE4_ROOT:-$study_root/phase4_rl_data_v1}"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
sft_adapter="$study_root/mini/pilot_sft/final_adapter"
case "${SLURM_ARRAY_TASK_ID:?missing array id}" in
  0) partition=a; seed=20260824 ;;
  1) partition=b; seed=20260825 ;;
  *) echo "unexpected Phase4 array index" >&2; exit 2 ;;
esac
roster="$M6_PHASE4_ROSTER_DIR/roster_${partition}.json"
output="$phase4_root/student_prescan/partition_${partition}"
test -f "$roster"
test ! -e "$output/collection_report.json"
mkdir -p "$output"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_collect_policy_success.py \
  --mode phase4_data_synthesis --role train --k 4 --seed "$seed" \
  --split-lock "$split_lock" --goals "$goals" --base-url "$M6_SERVICE_BASE_URL" \
  --task-roster "$roster" --task-roster-producer-git-sha "$M6_EXPECTED_GIT_SHA" \
  --adapter "$sft_adapter" \
  --max-model-turns 18 --max-environment-steps 15 \
  --concurrent-groups 4 --output-dir "$output"
