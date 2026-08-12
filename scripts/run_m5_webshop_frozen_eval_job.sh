#!/usr/bin/env bash
# One authorized, recoverable and evaluation-only M5 inference identity.
#SBATCH --job-name=m5-webshop-eval
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m5_webshop_frozen_eval_%j.out
#SBATCH --error=logs/m5_webshop_frozen_eval_%j.err

set -euo pipefail
repo_root="${M5_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 evaluation Git SHA}"
: "${M5_EVAL_IDENTITY:?set one frozen M5 evaluation identity}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

authorization="${M5_EVAL_AUTHORIZATION_PATH:-$repo_root/outputs/m5_webshop_credit_assignment_v1/readiness/frozen_eval_authorization_v1.json}"
output_root="$repo_root/outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/$M5_EVAL_IDENTITY"
mkdir -p "$output_root"
if test -f "$output_root/run_report.json"; then
  echo "M5 frozen evaluation identity already complete"
  exit 0
fi

submit_timeout_successor() {
  trap - USR1
  if test -n "${SLURM_JOB_ID:-}" && test "${M5_DISABLE_SUCCESSOR:-0}" != "1"; then
    successor_job_id="$(/opt/slurm/slurm.25.05/bin/sbatch --parsable \
      --dependency="afterany:${SLURM_JOB_ID}" \
      --export="ALL,M5_REPO_ROOT=$repo_root,M5_EXPECTED_GIT_SHA=$M5_EXPECTED_GIT_SHA,M5_EVAL_IDENTITY=$M5_EVAL_IDENTITY,M5_EVAL_AUTHORIZATION_PATH=$authorization" \
      scripts/run_m5_webshop_frozen_eval_job.sh)"
    printf '%s\n' "$successor_job_id" > "$output_root/successor_job_id"
    echo "timeout_successor_job_id=$successor_job_id"
  fi
  exit 99
}
trap submit_timeout_successor USR1

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTORCH_CUDA_ALLOC_CONF
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose exactly one GPU}"
case "$CUDA_VISIBLE_DEVICES" in
  *,*|-1) echo "M5 frozen evaluation requires exactly one visible GPU" >&2; exit 2 ;;
esac

echo "study=m5_webshop_credit_assignment_v1"
echo "phase=frozen_evaluation"
echo "formal_evaluation=true"
echo "training_updates_allowed=false"
echo "identity=$M5_EVAL_IDENTITY"
echo "repo_root=$repo_root"
echo "git_sha=$(git rev-parse HEAD)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "cpus=${SLURM_CPUS_PER_TASK:-6}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader
curl --fail --silent --show-error --max-time 30 \
  -X POST -H 'content-type: application/json' \
  -d '{"goal_index":1000}' http://127.0.0.1:44151/reset >/dev/null

/opt/slurm/slurm.25.05/bin/srun --ntasks=1 "$python_bin" scripts/m5_webshop_frozen_eval.py \
  --identity "$M5_EVAL_IDENTITY" \
  --expected-git-sha "$M5_EXPECTED_GIT_SHA" \
  --authorization "$authorization"

echo "finished=$(date -Is)"
