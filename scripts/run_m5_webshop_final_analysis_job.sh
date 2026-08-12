#!/usr/bin/env bash
# CPU-only post-hoc analysis of the completed frozen evaluation matrix.
#SBATCH --job-name=m5-final-analysis
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/m5_webshop_final_analysis_%j.out
#SBATCH --error=logs/m5_webshop_final_analysis_%j.err

set -euo pipefail
repo_root="${M5_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M5_EXPECTED_GIT_SHA:?set the M5 analysis Git SHA}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
mkdir -p logs
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
/home/wushaohua/miniconda3/envs/miniwebwork/bin/python \
  scripts/m5_webshop_analyze_final.py \
  --expected-git-sha "$M5_EXPECTED_GIT_SHA"
