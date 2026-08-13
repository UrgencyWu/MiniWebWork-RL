#!/usr/bin/env bash
# Rebuild paired Raw/SFT reports under evaluation-semantics v2, then run Gate.
# Existing collections, identity reports, and the failed Gate remain immutable.
#SBATCH --job-name=m6-sft-gate-v2
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m6_sft_gate_v2_%j.out
#SBATCH --error=logs/m6_sft_gate_v2_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_RAW_EVAL_ROOT:?set M6_RAW_EVAL_ROOT}"
: "${M6_SFT_EVAL_ROOT:?set M6_SFT_EVAL_ROOT}"
: "${M6_RAW_EVAL_REPORT_OUTPUT:?set M6_RAW_EVAL_REPORT_OUTPUT}"
: "${M6_SFT_EVAL_REPORT_OUTPUT:?set M6_SFT_EVAL_REPORT_OUTPUT}"
: "${M6_RAW_EVAL_BINDING_OUTPUT:?set M6_RAW_EVAL_BINDING_OUTPUT}"
: "${M6_SFT_EVAL_BINDING_OUTPUT:?set M6_SFT_EVAL_BINDING_OUTPUT}"
: "${M6_CORPUS_AUDIT:?set M6_CORPUS_AUDIT}"
: "${M6_PILOT_AUTHORIZATION:?set M6_PILOT_AUTHORIZATION}"
: "${M6_SFT_GATE_OUTPUT:?set M6_SFT_GATE_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
split_lock="${M6_SPLIT_LOCK:-$repo_root/outputs/m6_monotonic_posttraining_v1/locks/m6_webshop_split_v1.json}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

for required in \
  "$M6_RAW_EVAL_ROOT/groups" "$M6_RAW_EVAL_ROOT/collection_report.json" \
  "$M6_SFT_EVAL_ROOT/groups" "$M6_SFT_EVAL_ROOT/collection_report.json" \
  "$split_lock" "$M6_CORPUS_AUDIT" "$M6_PILOT_AUTHORIZATION"; do
  test -e "$required"
done
for absent in \
  "$M6_RAW_EVAL_REPORT_OUTPUT" "$M6_SFT_EVAL_REPORT_OUTPUT" \
  "$M6_RAW_EVAL_BINDING_OUTPUT" "$M6_SFT_EVAL_BINDING_OUTPUT" \
  "$M6_SFT_GATE_OUTPUT"; do
  test ! -e "$absent"
  mkdir -p "$(dirname "$absent")"
done

"$python_bin" scripts/m6_closed_loop_eval.py \
  --identity raw \
  --groups-dir "$M6_RAW_EVAL_ROOT/groups" \
  --collection-report "$M6_RAW_EVAL_ROOT/collection_report.json" \
  --split-lock "$split_lock" \
  --pilot-authorization "$M6_PILOT_AUTHORIZATION" \
  --pilot-binding-output "$M6_RAW_EVAL_BINDING_OUTPUT" \
  --output "$M6_RAW_EVAL_REPORT_OUTPUT"

"$python_bin" scripts/m6_closed_loop_eval.py \
  --identity mini_sft \
  --groups-dir "$M6_SFT_EVAL_ROOT/groups" \
  --collection-report "$M6_SFT_EVAL_ROOT/collection_report.json" \
  --split-lock "$split_lock" \
  --pilot-authorization "$M6_PILOT_AUTHORIZATION" \
  --pilot-binding-output "$M6_SFT_EVAL_BINDING_OUTPUT" \
  --output "$M6_SFT_EVAL_REPORT_OUTPUT"

M6_RAW_EVAL_REPORT="$M6_RAW_EVAL_REPORT_OUTPUT" \
M6_SFT_EVAL_REPORT="$M6_SFT_EVAL_REPORT_OUTPUT" \
M6_RAW_EVAL_BINDING="$M6_RAW_EVAL_BINDING_OUTPUT" \
M6_SFT_EVAL_BINDING="$M6_SFT_EVAL_BINDING_OUTPUT" \
bash scripts/run_m6_sft_gate_job.sh
