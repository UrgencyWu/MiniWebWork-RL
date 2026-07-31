#!/usr/bin/env bash
# Collect the frozen M4 test split for one completed method/seed only.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ALGORITHM STUDY_SEED" >&2
  exit 2
fi

algorithm="$1"
study_seed="$2"
case "$algorithm" in
  sft|rsft|rloo|grpo|gspo) ;;
  *)
    echo "ALGORITHM must be one of: sft, rsft, rloo, grpo, gspo" >&2
    exit 2
    ;;
esac
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

exec python scripts/m4_run_final_eval.py \
  --algorithm "$algorithm" \
  --seed "$study_seed" \
  --training-root outputs/m4_runs \
  --output-dir "outputs/m4_final_eval/${algorithm}/seed_${study_seed}"
