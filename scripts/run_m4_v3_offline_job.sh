#!/usr/bin/env bash
# One v3 SFT/RSFT offline update, bounded by one 24h Slurm allocation.
#SBATCH --job-name=m4-v3-offline
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ALGORITHM STUDY_SEED TRAIN_DATA VALID_DATA OUTPUT_DIR" >&2
  exit 2
fi

algorithm="$1"
study_seed="$2"
train_data="$3"
valid_data="$4"
output_dir="$5"
case "$algorithm" in sft|rsft) ;; *) echo "algorithm must be sft or rsft" >&2; exit 2 ;; esac
case "$study_seed" in 20260801|20260802|20260803) ;; *) echo "invalid study seed" >&2; exit 2 ;; esac

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
M4_EXPECTED_GIT_SHA="${M4_EXPECTED_GIT_SHA:?set the frozen v3 commit SHA}" \
  source scripts/m4_assert_frozen_git.sh

exec python scripts/m4_v3_train_offline.py \
  --algorithm "$algorithm" \
  --seed "$study_seed" \
  --train-data-dir "$train_data" \
  --validation-data-dir "$valid_data" \
  --initial-adapter outputs/m2_2r/seed_42/final_adapter \
  --output-dir "$output_dir"
