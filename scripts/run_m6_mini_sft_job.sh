#!/usr/bin/env bash
# Recoverable development-only mini-SFT. It cannot be reused as a formal checkpoint.
#SBATCH --job-name=m6-mini-sft
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m6_mini_sft_%j.out
#SBATCH --error=logs/m6_mini_sft_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
output="$repo_root/outputs/m6_monotonic_posttraining_v1/mini/sft"
mkdir -p "$output"

submit_timeout_successor() {
  trap - USR1
  if test -n "${SLURM_JOB_ID:-}" && test "${M6_DISABLE_SUCCESSOR:-0}" != "1"; then
    successor_job_id="$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,M6_EXPECTED_GIT_SHA=$M6_EXPECTED_GIT_SHA" scripts/run_m6_mini_sft_job.sh)"
    printf '%s\n' "$successor_job_id" > "$output/successor_job_id"
    echo "timeout_successor_job_id=$successor_job_id"
  fi
  exit 99
}
trap submit_timeout_successor USR1

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
/opt/slurm/slurm.25.05/bin/srun --ntasks=1 "$python_bin" scripts/m6_sft_train.py \
  --data-dir outputs/m6_monotonic_posttraining_v1/mini/corpus \
  --output-dir "$output"
