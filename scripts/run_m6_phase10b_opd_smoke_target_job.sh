#!/usr/bin/env bash
# Phase10-B: query one qualified Specialist on frozen Student behavior tokens.
#SBATCH --job-name=m6-p10b-opd-target
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase10b_opd_target_%j.out
#SBATCH --error=logs/m6_phase10b_opd_target_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?missing M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE10B_ROUTER_MANIFEST:?missing M6_PHASE10B_ROUTER_MANIFEST}"
: "${M6_PHASE10B_TARGET_IDENTITY:?missing M6_PHASE10B_TARGET_IDENTITY}"
: "${M6_PHASE10B_TARGET_OUTPUT:?missing M6_PHASE10B_TARGET_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$M6_PHASE10B_ROUTER_MANIFEST"
test ! -e "$M6_PHASE10B_TARGET_OUTPUT"
case "$M6_PHASE10B_TARGET_IDENTITY" in
  S_nav) model_path=/data/share/model/Qwen3.5-9B ;;
  S_match) model_path=/data/share/model/Qwen3.5-35B-A3B ;;
  S_finish) model_path=/data/share/model/Qwen3.6-35B-A3B-FP8 ;;
  *) echo "invalid Phase10-B target identity" >&2; exit 2 ;;
esac
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
model_manifest="$repo_root/outputs/m6_monotonic_posttraining_v1/phase10b_multi_specialist_opd_v1/base_model_manifests/$M6_PHASE10B_TARGET_IDENTITY.json"
test -f "$model_manifest"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/m6_phase10b_opd_smoke.py specialist \
  --router-manifest "$M6_PHASE10B_ROUTER_MANIFEST" \
  --identity "$M6_PHASE10B_TARGET_IDENTITY" \
  --model-path "$model_path" \
  --base-model-manifest "$model_manifest" \
  --output "$M6_PHASE10B_TARGET_OUTPUT"
