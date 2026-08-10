#!/usr/bin/env bash
# Repair and audit the existing learner/rollout environment without using a GPU.
# The sole tolerated pip-check conflict is frozen in the M5 machine contract.
#SBATCH --job-name=m5-train-env
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m5_training_env_%j.out
#SBATCH --error=logs/m5_training_env_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 preflight commit SHA}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export CUDA_VISIBLE_DEVICES=""
"$python_bin" -m pip install --disable-pip-version-check -r requirements.m5-training-runtime.txt
"$python_bin" scripts/m5_training_runtime_preflight.py \
  --base-model /data/share/model/Qwen3.5-4B \
  --output outputs/m5_webshop_credit_assignment_v1/preflight/training_runtime/runtime_audit.json
