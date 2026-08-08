#!/usr/bin/env bash
# Disposable vLLM/browser/PEFT online smoke. This entrypoint cannot submit formal training.
#SBATCH --job-name=m4-lh-online-preflight
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m4_lh_online_preflight_%j.out
#SBATCH --error=logs/m4_lh_online_preflight_%j.err

set -euo pipefail

repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs outputs/m4_long_horizon_credit_v1/preflight

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork

M4_EXPECTED_GIT_SHA="${M4_EXPECTED_GIT_SHA:?set the frozen preflight commit SHA}" \
  source scripts/m4_assert_frozen_git.sh

run_name="${M4_PREFLIGHT_RUN_NAME:?set one stable preflight run name for resume}"
initial_adapter="${M4_INITIAL_ADAPTER:?set the frozen disposable SFT adapter path}"
browser_workers="${M4_BROWSER_WORKERS:?set 1, 2, 4, or 8 browser workers}"
maximum_tasks="${M4_MAXIMUM_TASKS:?set a preflight task count within 1..32}"
learner_microbatch="${M4_LEARNER_MICROBATCH_SIZE:-1}"
method="${M4_METHOD:-multi_turn_grpo}"
seed="${M4_SEED:-20260801}"
collection_only="${M4_COLLECTION_ONLY:-0}"
if [[ ! "$run_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "unsafe M4_PREFLIGHT_RUN_NAME" >&2
  exit 2
fi
if [[ "$collection_only" != "0" && "$collection_only" != "1" ]]; then
  echo "M4_COLLECTION_ONLY must be 0 or 1" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
# vLLM sleep mode uses CuMemAllocator; expandable segments are incompatible
# with that memory pool. Keep both PyTorch allocator aliases unset for the
# engine and the later same-process learner.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

job_id="${SLURM_JOB_ID:-manual}"
output_root="$repo_root/outputs/m4_long_horizon_credit_v1/preflight/online_${run_name}"
mkdir -p "$output_root"

echo "study=m4_long_horizon_credit_v1"
echo "phase=preflight_online_single_gpu"
echo "formal_training=false"
echo "git_sha=$(git rev-parse HEAD)"
echo "job_id=$job_id"
echo "run_name=$run_name"
echo "host=$(hostname)"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown}"
echo "browser_workers=$browser_workers"
echo "maximum_tasks=$maximum_tasks"
echo "collection_only=$collection_only"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "started=$(date -Is)"
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader

telemetry_path="$output_root/gpu_telemetry_job_${job_id}.csv"
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

extra_args=()
if [[ "$collection_only" == "1" ]]; then
  extra_args+=(--collection-only)
fi

/opt/slurm/slurm.25.05/bin/srun --ntasks=1 python scripts/m4_long_horizon_online_preflight.py \
  --output-dir "$output_root" \
  --method "$method" \
  --seed "$seed" \
  --initial-adapter "$initial_adapter" \
  --browser-workers "$browser_workers" \
  --maximum-tasks "$maximum_tasks" \
  --learner-microbatch-size "$learner_microbatch" \
  "${extra_args[@]}"

cleanup
trap - EXIT
echo "telemetry_samples=$(wc -l < "$telemetry_path")"
echo "finished=$(date -Is)"
