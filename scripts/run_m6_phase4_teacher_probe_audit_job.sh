#!/usr/bin/env bash
# Phase4: CPU-only strict replay verification and bounded teacher supplement admission.
#SBATCH --job-name=m6-p4-teacher-audit
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/m6_phase4_teacher_audit_%j.out
#SBATCH --error=logs/m6_phase4_teacher_audit_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_SERVICE_BASE_URL:?missing M6_SERVICE_BASE_URL}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
phase4_root="$repo_root/outputs/m6_monotonic_posttraining_v1/phase4_rl_data_v1"
student_root="$phase4_root/student_prescan/audit"
teacher_root="${M6_TEACHER_ROOT:-$phase4_root/teacher_probe_qwen35_9b_v1}"
output="$teacher_root/audit"
test ! -e "$output/teacher_probe_audit.json"
mkdir -p "$output"
curl --fail --silent --show-error --max-time 30 "$M6_SERVICE_BASE_URL/health" >/dev/null
export PYTHONPATH="$repo_root/src:$repo_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase4_audit_teacher_probe.py \
  --teacher-roster "$teacher_root/prep/teacher_roster.json" \
  --student-audit "$student_root/dataset_audit.json" \
  --teacher-candidates "$student_root/teacher_candidates.json" \
  --collection-root "$teacher_root/collection" \
  --base-url "$M6_SERVICE_BASE_URL" \
  --output-dir "$output"
