#!/usr/bin/env bash
# Run one complete, two-pass M4 online-RL seed on a single Slurm GPU allocation.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ALGORITHM STUDY_SEED" >&2
  exit 2
fi

algorithm="$1"
study_seed="$2"
case "$algorithm" in
  rloo|grpo|gspo) ;;
  *)
    echo "ALGORITHM must be one of: rloo, grpo, gspo" >&2
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

exec python scripts/m4_run_online.py \
  --algorithm "$algorithm" \
  --seed "$study_seed" \
  --initial-adapter outputs/m2_2r/seed_42/final_adapter \
  --output-dir "outputs/m4_runs/${algorithm}/seed_${study_seed}"
