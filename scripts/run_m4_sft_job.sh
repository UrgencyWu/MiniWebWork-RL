#!/usr/bin/env bash
# Slurm entrypoint for one auditable M4 SFT seed.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 STUDY_SEED" >&2
  exit 2
fi

study_seed="$1"
case "$study_seed" in
  20260801|20260802|20260803) ;;
  *) echo "unsupported M4 study seed: $study_seed" >&2; exit 2 ;;
esac

repo_root="/home/wushaohua/data/MiniWebWork-RL"
source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork
cd "$repo_root"
source scripts/m4_assert_frozen_git.sh

python scripts/m4_train_offline.py \
  --algorithm sft \
  --seed "$study_seed" \
  --train-data-dir outputs/m4_sft_corpus_v1 \
  --initial-adapter outputs/m2_2r/seed_42/final_adapter \
  --output-dir "outputs/m4_runs/sft/seed_${study_seed}"
