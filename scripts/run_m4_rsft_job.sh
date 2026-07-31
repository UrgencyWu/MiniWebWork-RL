#!/usr/bin/env bash
# Run one complete M4 RSFT seed: two fixed-policy collections, corpus build, and train.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 STUDY_SEED" >&2
  exit 2
fi

study_seed="$1"
case "$study_seed" in
  20260801|20260802|20260803) ;;
  *)
    echo "STUDY_SEED must be one of: 20260801, 20260802, 20260803" >&2
    exit 2
    ;;
esac

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"

exec python scripts/m4_run_rsft.py \
  --seed "$study_seed" \
  --initial-adapter outputs/m2_2r/seed_42/final_adapter \
  --sft-validation-data-dir outputs/m4_sft_corpus_v1 \
  --output-dir "outputs/m4_runs/rsft/seed_${study_seed}"
