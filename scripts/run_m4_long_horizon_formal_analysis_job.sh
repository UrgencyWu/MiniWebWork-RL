#!/usr/bin/env bash
# CPU-only final analysis after all seven frozen evaluations validate.
#SBATCH --job-name=m4-lh-formal-analysis
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m4_lh_formal_analysis_%j.out
#SBATCH --error=logs/m4_lh_formal_analysis_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
: "${M4_EXPECTED_GIT_SHA:?set the frozen formal commit SHA}"
export M4_EXPECTED_GIT_SHA
source scripts/m4_assert_frozen_git.sh
/home/wushaohua/miniconda3/bin/conda run -n miniwebwork \
  python scripts/m4_long_horizon_formal_analyze.py
