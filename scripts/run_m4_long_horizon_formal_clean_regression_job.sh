#!/usr/bin/env bash
# Final frozen-SHA CPU regression used by readiness-v2.
#SBATCH --job-name=m4-lh-formal-clean
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m4_lh_formal_clean_%j.out
#SBATCH --error=logs/m4_lh_formal_clean_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
: "${M4_EXPECTED_GIT_SHA:?set the frozen formal commit SHA}"
export M4_EXPECTED_GIT_SHA
source scripts/m4_assert_frozen_git.sh
/home/wushaohua/miniconda3/bin/conda run -n miniwebwork \
  pytest -q -m "not browser and not gpu and not slurm"
