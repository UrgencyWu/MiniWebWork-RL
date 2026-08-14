#!/usr/bin/env bash
# Submit only the exact-prefix full-horizon successor and its CPU audit.
set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
expected_git_sha="$(git rev-parse HEAD)"
root="$repo_root/outputs/m6_monotonic_posttraining_v1/phase2_causal_validation_v1/p1_horizon"
roster="$root/task_roster_v1.json"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$roster"
test -f "$root/short_6_6/collection_report.json"
test ! -e "$root/full_18_15_prefix_replay_r2/collection_report.json"
test ! -e "$root/analysis_report_r2.json"
roster_producer="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_sha"])' "$roster")"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
replay_job="$($slurm_bin/sbatch --parsable \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE2_ROSTER=$roster,M6_ROSTER_PRODUCER_GIT_SHA=$roster_producer,M6_SERVICE_BASE_URL=$M6_SERVICE_BASE_URL" \
  scripts/run_m6_phase2_horizon_replay_job.sh)"
audit_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$replay_job" \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha" \
  scripts/run_m6_phase2_horizon_replay_analysis_job.sh)"
printf 'horizon_prefix_replay_job=%s\nhorizon_prefix_replay_audit_job=%s\n' "$replay_job" "$audit_job"
