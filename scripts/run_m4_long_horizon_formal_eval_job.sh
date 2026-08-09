#!/usr/bin/env bash
# One recoverable frozen-test model. Run no more than four such jobs concurrently.
#SBATCH --job-name=m4-lh-formal-eval
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m4_lh_formal_eval_%j.out
#SBATCH --error=logs/m4_lh_formal_eval_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork
: "${M4_EXPECTED_GIT_SHA:?set the frozen formal commit SHA}"
export M4_EXPECTED_GIT_SHA
source scripts/m4_assert_frozen_git.sh
method="${M4_METHOD:?set verified_sft, multi_turn_grpo, or step_aware_gpo}"
seed="${M4_SEED:?set the model seed}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF

job_id="${SLURM_JOB_ID:-manual}"
telemetry_path="$repo_root/logs/m4_lh_formal_eval_${job_id}_gpu.csv"
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

/opt/slurm/slurm.25.05/bin/srun --ntasks=1 python scripts/m4_long_horizon_formal_eval.py \
  --method "$method" --seed "$seed" \
  --expected-git-sha "$M4_EXPECTED_GIT_SHA" \
  --readiness outputs/m4_long_horizon_credit_v1/readiness/readiness_manifest_v2.json

cleanup
trap - EXIT
echo "telemetry_samples=$(wc -l < "$telemetry_path")"
