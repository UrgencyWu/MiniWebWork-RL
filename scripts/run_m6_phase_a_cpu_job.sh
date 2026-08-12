#!/usr/bin/env bash
# CPU-only Phase A0: exposure, prospective power, and immutable M6 split.
#SBATCH --job-name=m6-phase-a
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/m6_phase_a_%j.out
#SBATCH --error=logs/m6_phase_a_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
locks_root="$study_root/locks"
mkdir -p logs "$study_root/power_inputs" "$locks_root"
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"$python_bin" scripts/m6_build_exposure_registry.py \
  --evidence sft_corpus=outputs/m5_webshop_credit_assignment_v1/preflight/sft_corpus \
  --evidence online_training=outputs/m5_webshop_credit_assignment_v1/formal/online \
  --evidence frozen_evaluation=outputs/m5_webshop_credit_assignment_v1/formal/frozen_test \
  --evidence technical_report=docs/M5_FINAL_TECHNICAL_REPORT.md \
  --output "$locks_root/m5_goal_exposure_registry_v1.json"

"$python_bin" scripts/m6_extract_m5_power_inputs.py \
  --raw-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/raw_base_model/groups \
  --sft-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/shared_verified_sft/groups \
  --rl-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/anchor_gigpo_seed_20260801/groups \
  --rl-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/anchor_gigpo_seed_20260802/groups \
  --rl-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/anchor_gigpo_seed_20260803/groups \
  --output-dir "$study_root/power_inputs"

"$python_bin" scripts/m6_plan_statistical_power.py \
  --input raw_sft="$study_root/power_inputs/raw_sft.csv" \
  --input sft_rl="$study_root/power_inputs/sft_rl.csv" \
  --input raw_rl="$study_root/power_inputs/raw_rl.csv" \
  --output "$study_root/power_report.json"

n_eval="$($python_bin -c 'import json,sys; value=json.load(open(sys.argv[1])); assert value["passed"] is True; print(value["selected_n_eval"])' "$study_root/power_report.json")"
"$python_bin" scripts/m6_build_split.py \
  --goals outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json \
  --exposure-registry "$locks_root/m5_goal_exposure_registry_v1.json" \
  --power-report "$study_root/power_report.json" \
  --n-eval "$n_eval" --output "$locks_root/m6_webshop_split_v1.json"
