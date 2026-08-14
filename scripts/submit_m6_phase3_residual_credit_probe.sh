#!/usr/bin/env bash
set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
expected_git_sha="$(git rev-parse HEAD)"
phase3_root="$repo_root/outputs/m6_monotonic_posttraining_v1/phase3_residual_credit_v1"
p2_report="$repo_root/outputs/m6_monotonic_posttraining_v1/phase2_causal_validation_v1/p2_reward_counterfactual/report.json"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$p2_report"
test ! -e "$phase3_root/p0_same_batch/report.json"
grep -q '"p2_same_batch_reward_probe_passed":false' "$p2_report"
job_id="$($slurm_bin/sbatch --parsable \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE3_ROOT=$phase3_root" \
  scripts/run_m6_phase3_residual_credit_probe_job.sh)"
printf 'phase3_residual_credit_probe_job=%s\n' "$job_id"
