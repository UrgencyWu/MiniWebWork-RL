#!/usr/bin/env bash
# One v3 collection/update pass. Resubmitting the same arguments resumes only
# complete K=4 groups from the immutable incremental artifact.
#SBATCH --job-name=m4-v3-pass
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ALGORITHM STUDY_SEED PASS_INDEX INPUT_ADAPTER OUTPUT_DIR" >&2
  exit 2
fi

algorithm="$1"
study_seed="$2"
pass_index="$3"
input_adapter="$4"
output_dir="$5"

case "$algorithm" in
  rloo|grpo|gspo|rsft) ;;
  *) echo "unsupported algorithm: $algorithm" >&2; exit 2 ;;
esac
case "$study_seed" in
  20260801|20260802|20260803) ;;
  *) echo "unsupported study seed: $study_seed" >&2; exit 2 ;;
esac
case "$pass_index" in
  1|2) ;;
  *) echo "PASS_INDEX must be 1 or 2" >&2; exit 2 ;;
esac

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
M4_EXPECTED_GIT_SHA="${M4_EXPECTED_GIT_SHA:?set the frozen v3 commit SHA}" \
  source scripts/m4_assert_frozen_git.sh

exec python scripts/m4_v3_pass_job.py \
  --algorithm "$algorithm" \
  --seed "$study_seed" \
  --pass-index "$pass_index" \
  --input-adapter "$input_adapter" \
  --output-dir "$output_dir"
