#!/usr/bin/env bash
# Submit independent P0a GPU dropout and P0b CPU buy-readiness diagnostics.
set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
expected_git_sha="$(git rev-parse HEAD)"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase2_root="${M6_PHASE2_ROOT:-$study_root/phase2_causal_validation_v1}"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$phase2_root/p0a_dropout/report.json"
test ! -e "$phase2_root/p0b_buy_readiness/report.json"
mkdir -p "$phase2_root"

dropout_job="$($slurm_bin/sbatch --parsable \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE2_ROOT=$phase2_root" \
  scripts/run_m6_phase2_dropout_probe_job.sh)"
readiness_job="$($slurm_bin/sbatch --parsable \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE2_ROOT=$phase2_root" \
  scripts/run_m6_phase2_buy_readiness_job.sh)"
printf 'dropout_job=%s\nbuy_readiness_job=%s\n' "$dropout_job" "$readiness_job"
