#!/usr/bin/env bash
# Submit the bounded M6 phase-one diagnostics: probe -> paired seen eval -> final CPU report.
set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
expected_git_sha="$(git rev-parse HEAD)"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase1_root="$study_root/phase1_diagnostics_v1"
roster="$phase1_root/updated_task_roster.json"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$phase1_root/final_report.json"
mkdir -p "$phase1_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase1_build_roster.py \
  --online-root "$study_root/medium/online_v1" \
  --split-lock "$study_root/locks/m6_webshop_split_v1.json" \
  --output "$roster"
roster_producer_git_sha="$expected_git_sha"

probe_job="$($slurm_bin/sbatch --parsable \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE1_ROOT=$phase1_root" \
  scripts/run_m6_phase1_gpu_probe_job.sh)"
seen_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$probe_job" \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE1_ROOT=$phase1_root,M6_PHASE1_ROSTER=$roster,M6_ROSTER_PRODUCER_GIT_SHA=$roster_producer_git_sha,M6_SERVICE_BASE_URL=$M6_SERVICE_BASE_URL" \
  scripts/run_m6_phase1_seen_eval_job.sh)"
final_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$seen_job" \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE1_ROOT=$phase1_root" \
  scripts/run_m6_phase1_finalize_job.sh)"
printf 'probe_job=%s\nseen_eval_job=%s\nfinal_job=%s\n' "$probe_job" "$seen_job" "$final_job"
