#!/usr/bin/env bash
# One authorized, recoverable M5 online method/seed. Never submits without the
# readiness and user-authorization artifacts bound to the frozen Git SHA.
#SBATCH --job-name=m5-webshop-formal-rl
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m5_webshop_formal_online_%j.out
#SBATCH --error=logs/m5_webshop_formal_online_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 formal Git SHA}"
: "${M5_METHOD:?set multi_turn_grpo or anchor_gigpo}"
: "${M5_SEED:?set 20260801, 20260802, or 20260803}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

readiness="${M5_READINESS_PATH:-$repo_root/outputs/m5_webshop_credit_assignment_v1/readiness/readiness_manifest_v1.json}"
authorization="${M5_AUTHORIZATION_PATH:-$repo_root/outputs/m5_webshop_credit_assignment_v1/readiness/formal_authorization_v1.json}"
output_root="$repo_root/outputs/m5_webshop_credit_assignment_v1/formal/online/$M5_METHOD/seed_$M5_SEED"
mkdir -p "$output_root"
if test -f "$output_root/run_report.json"; then
  echo "M5 formal online run already complete"
  exit 0
fi

submit_timeout_successor() {
  trap - USR1
  if test -n "${SLURM_JOB_ID:-}" && test "${M5_DISABLE_SUCCESSOR:-0}" != "1"; then
    successor_job_id="$(sbatch --parsable \
      --dependency="afterany:${SLURM_JOB_ID}" \
      --export="ALL,M5_EXPECTED_GIT_SHA=$M5_EXPECTED_GIT_SHA,M5_METHOD=$M5_METHOD,M5_SEED=$M5_SEED,M5_READINESS_PATH=$readiness,M5_AUTHORIZATION_PATH=$authorization" \
      scripts/run_m5_webshop_formal_online_job.sh)"
    printf '%s\n' "$successor_job_id" > "$output_root/successor_job_id"
    echo "timeout_successor_job_id=$successor_job_id"
  fi
  exit 99
}
trap submit_timeout_successor USR1

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTORCH_CUDA_ALLOC_CONF
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose exactly one GPU}"
case "$CUDA_VISIBLE_DEVICES" in
  *,*|-1) echo "M5 formal run requires exactly one visible GPU" >&2; exit 2 ;;
esac

echo "study=m5_webshop_credit_assignment_v1"
echo "phase=formal_online"
echo "formal_training=true"
echo "method=$M5_METHOD"
echo "seed=$M5_SEED"
echo "git_sha=$(git rev-parse HEAD)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "cpus=${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader
curl --fail --silent --show-error --max-time 30 \
  -X POST -H 'content-type: application/json' \
  -d '{"goal_index":1000}' http://127.0.0.1:44151/reset >/dev/null

/opt/slurm/slurm.25.05/bin/srun --ntasks=1 "$python_bin" scripts/m5_webshop_formal_online.py \
  --method "$M5_METHOD" \
  --seed "$M5_SEED" \
  --readiness "$readiness" \
  --authorization "$authorization"

echo "finished=$(date -Is)"
