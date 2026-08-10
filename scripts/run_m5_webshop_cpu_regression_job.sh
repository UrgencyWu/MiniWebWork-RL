#!/usr/bin/env bash
# Scheduled clean-SHA regression. This consumes CPU only and submits no model work.
# The same entrypoint is rerun at the final readiness SHA.
#SBATCH --job-name=m5-webshop-cpu
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m5_webshop_cpu_%j.out
#SBATCH --error=logs/m5_webshop_cpu_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 preflight commit SHA}"
actual_sha="$(git rev-parse HEAD)"
test "$actual_sha" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork

echo "study=m5_webshop_credit_assignment_v1"
echo "phase=clean_sha_cpu_regression"
echo "git_sha=$actual_sha"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "python=$(command -v python)"
python --version
python -m pytest -q -m "not gpu and not slurm"
