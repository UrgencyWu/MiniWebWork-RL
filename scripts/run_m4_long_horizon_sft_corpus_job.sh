#!/usr/bin/env bash
# Build the unique verified SFT corpus and exact token audit. This is a CPU
# pre-training data job, not a formal model-training job.
#SBATCH --job-name=m4-lh-sft-corpus
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m4_lh_sft_corpus_%j.out
#SBATCH --error=logs/m4_lh_sft_corpus_%j.err

set -euo pipefail

repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork

M4_EXPECTED_GIT_SHA="${M4_EXPECTED_GIT_SHA:?set the frozen preflight commit SHA}" \
  source scripts/m4_assert_frozen_git.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TOKENIZERS_PARALLELISM=false

output_dir="$repo_root/data/sft/m4_long_horizon_verified_v2"
slurm_srun="/opt/slurm/slurm.25.05/bin/srun"
echo "study=m4_long_horizon_credit_v1"
echo "phase=preflight_sft_corpus"
echo "git_sha=$(git rev-parse HEAD)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown}"
echo "output_dir=$output_dir"
echo "started=$(date -Is)"

"$slurm_srun" --ntasks=1 python scripts/build_m4_long_horizon_sft_corpus.py \
  --output-dir "$output_dir"
"$slurm_srun" --ntasks=1 python scripts/audit_m4_long_horizon_sft_tokens.py \
  --data-dir "$output_dir" \
  --max-length 6144
"$slurm_srun" --ntasks=1 python scripts/build_m4_long_horizon_sft_corpus.py \
  --output-dir "$output_dir" \
  --validate-only

echo "finished=$(date -Is)"
