#!/usr/bin/env bash
# Resumable full-epoch SFT GPU preflight. This is not a formal training entry.
# One logical run may span multiple 24-hour allocations in the same output root.
#SBATCH --job-name=m5-webshop-sft-pf
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m5_webshop_sft_preflight_%j.out
#SBATCH --error=logs/m5_webshop_sft_preflight_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 preflight commit SHA}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

output_root="$repo_root/outputs/m5_webshop_credit_assignment_v1/preflight/sft_gpu"
mkdir -p "$output_root"
if test -f "$output_root/preflight_report.json"; then
  echo "M5 SFT GPU preflight already complete"
  exit 0
fi

# Pre-submit one afterany successor. On clean completion it is cancelled; on
# timeout or node loss it reopens this exact output root and the last committed
# optimizer-boundary checkpoint.
successor_job_id=""
if test -n "${SLURM_JOB_ID:-}" && test "${M5_DISABLE_SUCCESSOR:-0}" != "1"; then
  successor_job_id="$(sbatch --parsable \
    --dependency="afterany:${SLURM_JOB_ID}" \
    --export="ALL,M5_EXPECTED_GIT_SHA=${M5_EXPECTED_GIT_SHA}" \
    scripts/run_m5_webshop_sft_preflight_job.sh)"
  printf '%s\n' "$successor_job_id" > "$output_root/successor_job_id"
  echo "successor_job_id=$successor_job_id"
fi

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "phase=m5_sft_gpu_preflight"
echo "formal_training=false"
echo "git_sha=$(git rev-parse HEAD)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "cpus=${SLURM_CPUS_PER_TASK:-8}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader

telemetry="$output_root/gpu_telemetry_${SLURM_JOB_ID:-manual}.csv"
nvidia-smi \
  --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits --loop=5 > "$telemetry" &
telemetry_pid=$!
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT

/opt/slurm/slurm.25.05/bin/srun --ntasks=1 "$python_bin" scripts/m5_webshop_sft_preflight.py \
  --base-model /data/share/model/Qwen3.5-4B \
  --data-dir outputs/m5_webshop_credit_assignment_v1/preflight/sft_corpus \
  --output-dir "$output_root"

cleanup
trap - EXIT
if test -f "$output_root/preflight_report.json" && test -n "$successor_job_id"; then
  scancel "$successor_job_id" || true
fi
echo "telemetry_samples=$(wc -l < "$telemetry")"
echo "finished=$(date -Is)"
