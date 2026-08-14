#!/usr/bin/env bash
# Phase4: CPU-only teacher roster freeze and Qwen3.5-9B functional manifest.
#SBATCH --job-name=m6-p4-teacher-prep
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/m6_phase4_teacher_prep_%j.out
#SBATCH --error=logs/m6_phase4_teacher_prep_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
phase4_root="$repo_root/outputs/m6_monotonic_posttraining_v1/phase4_rl_data_v1"
student_root="$phase4_root/student_prescan/audit"
teacher_root="$phase4_root/teacher_probe_qwen35_9b_v1"
mkdir -p "$teacher_root/prep"
export PYTHONPATH="$repo_root/src:$repo_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase4_build_teacher_roster.py \
  --student-audit "$student_root/dataset_audit.json" \
  --teacher-candidates "$student_root/teacher_candidates.json" \
  --split-lock "$repo_root/outputs/m6_monotonic_posttraining_v1/locks/m6_webshop_split_v1.json" \
  --output-dir "$teacher_root/prep"
