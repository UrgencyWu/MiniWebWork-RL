#!/usr/bin/env bash
set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
expected_git_sha="$(git rev-parse HEAD)"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase4_root="$study_root/phase4_rl_data_v1"
roster_dir="$phase4_root/rosters"
corpus_root="$study_root/mini/corpus_v2"
p1_roster="$study_root/phase2_causal_validation_v1/p1_horizon/task_roster_v1.json"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$roster_dir/manifest.json"
test ! -e "$phase4_root/student_prescan/audit/dataset_audit.json"
mkdir -p "$roster_dir"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase4_build_data_rosters.py \
  --split-lock "$study_root/locks/m6_webshop_split_v1.json" \
  --goals "$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json" \
  --sft-jsonl "$corpus_root/train.jsonl" --sft-jsonl "$corpus_root/dev.jsonl" \
  --exclude-roster "$p1_roster" --output-dir "$roster_dir"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
prescan_job="$($slurm_bin/sbatch --parsable \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE4_ROOT=$phase4_root,M6_PHASE4_ROSTER_DIR=$roster_dir,M6_SERVICE_BASE_URL=$M6_SERVICE_BASE_URL" \
  scripts/run_m6_phase4_student_prescan_job.sh)"
audit_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$prescan_job" \
  --export="ALL,M6_EXPECTED_GIT_SHA=$expected_git_sha,M6_PHASE4_ROOT=$phase4_root" \
  scripts/run_m6_phase4_student_prescan_audit_job.sh)"
printf 'student_prescan_array_job=%s\nstudent_prescan_audit_job=%s\nroster_manifest=%s\n' \
  "$prescan_job" "$audit_job" "$roster_dir/manifest.json"
