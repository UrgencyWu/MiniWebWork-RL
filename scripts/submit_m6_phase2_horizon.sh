#!/usr/bin/env bash
# Freeze the P1 roster and submit paired horizon arms plus one CPU audit.
set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
expected_git_sha="$(git rev-parse HEAD)"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase2_root="${M6_PHASE2_ROOT:-$study_root/phase2_causal_validation_v1}"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
goals="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json"
roster="$phase2_root/p1_horizon/task_roster_v1.json"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$roster"
test ! -e "$phase2_root/p1_horizon/analysis_report.json"
mkdir -p "$(dirname "$roster")"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase2_build_horizon_roster.py \
  --split-lock "$split_lock" --goals "$goals" --output "$roster"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
horizon_job="$($slurm_bin/sbatch --parsable \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE2_ROOT=$phase2_root,M6_PHASE2_ROSTER=$roster,M6_ROSTER_PRODUCER_GIT_SHA=$expected_git_sha,M6_SERVICE_BASE_URL=$M6_SERVICE_BASE_URL" \
  scripts/run_m6_phase2_horizon_job.sh)"
audit_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$horizon_job" \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha" \
  scripts/run_m6_phase2_horizon_analysis_job.sh)"
printf 'horizon_array_job=%s\nhorizon_audit_job=%s\nroster=%s\n' "$horizon_job" "$audit_job" "$roster"
