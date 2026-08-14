#!/usr/bin/env bash
#SBATCH --job-name=m6-p6-preterminal
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase6_preterminal_%j.out
#SBATCH --error=logs/m6_phase6_preterminal_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
study=outputs/m6_monotonic_posttraining_v1
online="$study/phase4_online_v1/seed_20260827"
reference="$study/mini/pilot_sft/final_adapter"
output="${M6_PHASE6_OUTPUT:-$study/phase6_preterminal_credit_v1/report.json}"
test ! -e "$output"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
"$python_bin" scripts/m6_phase6_preterminal_credit_probe.py \
  --panel-collection-root "$online/step_00/collection" \
  --panel-input-adapter "$reference" \
  --panel-collection-root "$online/step_01/collection" \
  --panel-input-adapter "$online/step_00/learner/adapter" \
  --reference-sft-adapter "$reference" \
  --output "$output"
