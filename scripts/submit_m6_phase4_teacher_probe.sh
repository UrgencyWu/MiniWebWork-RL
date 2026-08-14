#!/usr/bin/env bash
set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
expected_git_sha="$(git rev-parse HEAD)"
phase4_root="$repo_root/outputs/m6_monotonic_posttraining_v1/phase4_rl_data_v1"
teacher_root="${M6_TEACHER_ROOT:-$phase4_root/teacher_probe_qwen35_9b_v1}"
teacher_model="${M6_TEACHER_MODEL:-/data/share/model/Qwen3.5-9B}"
student_root="$repo_root/outputs/m6_monotonic_posttraining_v1/phase4_rl_data_v1/student_prescan/audit"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$student_root/dataset_audit.json"
test -f "$student_root/teacher_candidates.json"
test ! -e "$teacher_root/collection/collection_report.json"
test ! -e "$teacher_root/audit/teacher_probe_audit.json"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
prep_job="$($slurm_bin/sbatch --parsable \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_TEACHER_ROOT=$teacher_root,M6_TEACHER_MODEL=$teacher_model" \
  scripts/run_m6_phase4_teacher_prep_job.sh)"
probe_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$prep_job" \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_SERVICE_BASE_URL=$M6_SERVICE_BASE_URL,M6_TEACHER_ROOT=$teacher_root,M6_TEACHER_MODEL=$teacher_model" \
  scripts/run_m6_phase4_teacher_probe_job.sh)"
audit_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$probe_job" \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_SERVICE_BASE_URL=$M6_SERVICE_BASE_URL,M6_TEACHER_ROOT=$teacher_root" \
  scripts/run_m6_phase4_teacher_probe_audit_job.sh)"
printf 'teacher_prep_job=%s\nteacher_probe_job=%s\nteacher_audit_job=%s\n' \
  "$prep_job" "$probe_job" "$audit_job"
