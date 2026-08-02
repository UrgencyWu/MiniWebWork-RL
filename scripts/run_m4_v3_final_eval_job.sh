#!/usr/bin/env bash
# One recoverable v3 frozen-test collection, bounded by 24 hours.
#SBATCH --job-name=m4-v3-test
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ALGORITHM STUDY_SEED" >&2
  exit 2
fi
algorithm="$1"
study_seed="$2"

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
M4_EXPECTED_GIT_SHA="${M4_EXPECTED_GIT_SHA:?set the frozen v3 commit SHA}" \
  source scripts/m4_assert_frozen_git.sh

exec python scripts/m4_v3_run_final_eval.py \
  --algorithm "$algorithm" \
  --seed "$study_seed" \
  --training-root outputs/m4_v3_runs \
  --output-dir "outputs/m4_v3_final_eval/${algorithm}/seed_${study_seed}"
