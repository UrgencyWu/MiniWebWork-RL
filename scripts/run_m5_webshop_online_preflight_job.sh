#!/usr/bin/env bash
# Real K4 collection + two-learner GPU preflight. Never a formal training entry.
# The USR1 timeout warning submits one same-root successor. Deterministic
# failures stop immediately instead of creating an unbounded retry chain.
#SBATCH --job-name=m5-webshop-online-pf
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m5_webshop_online_preflight_%j.out
#SBATCH --error=logs/m5_webshop_online_preflight_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 preflight commit SHA}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

output_root="$repo_root/outputs/m5_webshop_credit_assignment_v1/preflight/online_gpu"
mkdir -p "$output_root"
if test -f "$output_root/preflight_report.json"; then
  echo "M5 online GPU preflight already complete"
  exit 0
fi

submit_timeout_successor() {
  trap - USR1
  if test -n "${SLURM_JOB_ID:-}" && test "${M5_DISABLE_SUCCESSOR:-0}" != "1"; then
    successor_job_id="$(sbatch --parsable \
      --dependency="afterany:${SLURM_JOB_ID}" \
      --export="ALL,M5_EXPECTED_GIT_SHA=${M5_EXPECTED_GIT_SHA}" \
      scripts/run_m5_webshop_online_preflight_job.sh)"
    printf '%s\n' "$successor_job_id" > "$output_root/successor_job_id"
    echo "timeout_successor_job_id=$successor_job_id"
  fi
  exit 99
}
trap submit_timeout_successor USR1

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "phase=m5_online_gpu_preflight"
echo "formal_training=false"
echo "git_sha=$(git rev-parse HEAD)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "cpus=${SLURM_CPUS_PER_TASK:-8}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader

curl --fail --silent --show-error --max-time 30 \
  -X POST -H 'content-type: application/json' \
  -d '{"goal_index":1000}' http://127.0.0.1:44151/reset >/dev/null

/opt/slurm/slurm.25.05/bin/srun --ntasks=1 "$python_bin" scripts/m5_webshop_online_preflight.py \
  --base-model /data/share/model/Qwen3.5-4B \
  --initial-adapter outputs/m5_webshop_credit_assignment_v1/preflight/sft_gpu/training/final_adapter_epoch_1 \
  --output-dir "$output_root" \
  --base-url http://127.0.0.1:44151

echo "finished=$(date -Is)"
