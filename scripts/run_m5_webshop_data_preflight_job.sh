#!/usr/bin/env bash
# Download and byte-audit the pinned WebShop runtime. This job never trains or
# opens the frozen test outcomes to a model-selection path.
#SBATCH --job-name=m5-webshop-data
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/m5_webshop_data_%j.out
#SBATCH --error=logs/m5_webshop_data_%j.err

set -euo pipefail

repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 preflight commit SHA}"
actual_sha="$(git rev-parse HEAD)"
test "$actual_sha" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

runtime_root="$repo_root/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full"
audit_path="$repo_root/outputs/m5_webshop_credit_assignment_v1/preflight/data/runtime_audit.json"
python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"

echo "study=m5_webshop_credit_assignment_v1"
echo "phase=pinned_data_preflight"
echo "git_sha=$actual_sha"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown}"
echo "runtime_root=$runtime_root"
echo "started=$(date -Is)"

"$python_bin" scripts/m5_webshop_data_preflight.py download \
  --runtime-root "$runtime_root"
"$python_bin" scripts/m5_webshop_data_preflight.py audit \
  --runtime-root "$runtime_root" \
  --output "$audit_path"

echo "audit_path=$audit_path"
echo "finished=$(date -Is)"
