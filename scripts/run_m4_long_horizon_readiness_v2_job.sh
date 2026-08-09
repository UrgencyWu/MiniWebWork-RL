#!/usr/bin/env bash
# Compute readiness-v2 after the final clean regression; this never trains.
#SBATCH --job-name=m4-lh-readiness-v2
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m4_lh_readiness_v2_%j.out
#SBATCH --error=logs/m4_lh_readiness_v2_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
: "${M4_EXPECTED_GIT_SHA:?set the frozen formal commit SHA}"
: "${M4_CLEAN_REGRESSION_JOB_ID:?set the completed clean-regression JobID}"
export M4_EXPECTED_GIT_SHA
source scripts/m4_assert_frozen_git.sh
/home/wushaohua/miniconda3/bin/conda run -n miniwebwork \
  python scripts/m4_long_horizon_readiness_v2.py \
    --expected-git-sha "$M4_EXPECTED_GIT_SHA" \
    --clean-regression-job-id "$M4_CLEAN_REGRESSION_JOB_ID" \
    --clean-regression-stdout "logs/m4_lh_formal_clean_${M4_CLEAN_REGRESSION_JOB_ID}.out" \
    --clean-regression-stderr "logs/m4_lh_formal_clean_${M4_CLEAN_REGRESSION_JOB_ID}.err"
