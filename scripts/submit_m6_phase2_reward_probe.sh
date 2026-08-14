#!/usr/bin/env bash
set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
expected_git_sha="$(git rev-parse HEAD)"
phase2_root="$repo_root/outputs/m6_monotonic_posttraining_v1/phase2_causal_validation_v1"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$phase2_root/p2_reward_counterfactual/report.json"
job_id="$($slurm_bin/sbatch --parsable \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE2_ROOT=$phase2_root" \
  scripts/run_m6_phase2_reward_probe_job.sh)"
printf 'reward_probe_job=%s\n' "$job_id"
