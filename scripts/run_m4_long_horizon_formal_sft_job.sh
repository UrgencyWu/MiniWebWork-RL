#!/usr/bin/env bash
# Recoverable shared formal SFT. Re-submit the same script after a 24h timeout.
#SBATCH --job-name=m4-lh-formal-sft
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m4_lh_formal_sft_%j.out
#SBATCH --error=logs/m4_lh_formal_sft_%j.err

set -euo pipefail

repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork

: "${M4_EXPECTED_GIT_SHA:?set the frozen formal commit SHA}"
export M4_EXPECTED_GIT_SHA
source scripts/m4_assert_frozen_git.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

job_id="${SLURM_JOB_ID:-manual}"
telemetry_path="$repo_root/logs/m4_lh_formal_sft_${job_id}_gpu.csv"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose exactly one GPU}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* && "$CUDA_VISIBLE_DEVICES" != "-1" ]]
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits --loop=5 > "$telemetry_path" &
telemetry_pid=$!
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT

echo "study=m4_long_horizon_credit_v1"
echo "phase=formal_shared_sft"
echo "formal_training=true"
echo "git_sha=$(git rev-parse HEAD)"
echo "job_id=$job_id"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

/opt/slurm/slurm.25.05/bin/srun --ntasks=1 python scripts/m4_long_horizon_formal_sft.py \
  --expected-git-sha "$M4_EXPECTED_GIT_SHA" \
  --readiness outputs/m4_long_horizon_credit_v1/readiness/readiness_manifest_v2.json \
  --base-model /data/share/model/Qwen3.5-4B \
  --data-dir data/sft/m4_long_horizon_verified_v2 \
  --output-root outputs/m4_long_horizon_credit_v1/formal/shared_sft/seed_20260801 \
  --dataloader-workers 2

cleanup
trap - EXIT
echo "telemetry_samples=$(wc -l < "$telemetry_path")"
