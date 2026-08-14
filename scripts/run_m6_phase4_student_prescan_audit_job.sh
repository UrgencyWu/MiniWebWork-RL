#!/usr/bin/env bash
# Phase4: CPU-only student exploration-support and contrast audit.
#SBATCH --job-name=m6-p4-prescan-audit
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/m6_phase4_student_prescan_audit_%j.out
#SBATCH --error=logs/m6_phase4_student_prescan_audit_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
phase4_root="${M6_PHASE4_ROOT:-$study_root/phase4_rl_data_v1}"
roster_dir="$phase4_root/rosters"
corpus_root="$study_root/mini/corpus_v2"
p1_roster="$study_root/phase2_causal_validation_v1/p1_horizon/task_roster_v1.json"
output="$phase4_root/student_prescan/audit"
test ! -e "$output/dataset_audit.json"
mkdir -p "$output"
export PYTHONPATH="$repo_root/src:$repo_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase4_audit_student_prescan.py \
  --roster-manifest "$roster_dir/manifest.json" \
  --collection "a=$phase4_root/student_prescan/partition_a" \
  --collection "b=$phase4_root/student_prescan/partition_b" \
  --split-lock "$study_root/locks/m6_webshop_split_v1.json" \
  --goals "$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json" \
  --sft-jsonl "$corpus_root/train.jsonl" --sft-jsonl "$corpus_root/dev.jsonl" \
  --exclude-roster "$p1_roster" \
  --sft-adapter "$study_root/mini/pilot_sft/final_adapter" \
  --output-dir "$output"
