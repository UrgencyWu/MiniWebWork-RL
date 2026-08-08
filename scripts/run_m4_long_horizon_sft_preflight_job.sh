#!/usr/bin/env bash
# Disposable GPU benchmark plus 20-update SFT smoke. Never a formal training job.
#SBATCH --job-name=m4-lh-sft-preflight
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m4_lh_sft_preflight_%j.out
#SBATCH --error=logs/m4_lh_sft_preflight_%j.err

set -euo pipefail

repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs outputs/m4_long_horizon_credit_v1/preflight

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork

M4_EXPECTED_GIT_SHA="${M4_EXPECTED_GIT_SHA:?set the frozen preflight commit SHA}" \
  source scripts/m4_assert_frozen_git.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

job_id="${SLURM_JOB_ID:-manual}"
output_root="$repo_root/outputs/m4_long_horizon_credit_v1/preflight/sft_${job_id}"
mkdir "$output_root"

echo "study=m4_long_horizon_credit_v1"
echo "phase=preflight_sft_gpu"
echo "formal_training=false"
echo "git_sha=$(git rev-parse HEAD)"
echo "job_id=$job_id"
echo "host=$(hostname)"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "started=$(date -Is)"
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader

telemetry_path="$output_root/gpu_telemetry.csv"
nvidia-smi \
  --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits \
  --loop=5 > "$telemetry_path" &
telemetry_pid=$!
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT

/opt/slurm/slurm.25.05/bin/srun --ntasks=1 python scripts/m4_long_horizon_sft.py \
  --mode preflight \
  --base-model /data/share/model/Qwen3.5-4B \
  --data-dir data/sft/m4_long_horizon_verified_v2 \
  --output-dir "$output_root/run" \
  --dataloader-workers 2

cleanup
trap - EXIT
echo "telemetry_samples=$(wc -l < "$telemetry_path")"
echo "finished=$(date -Is)"
