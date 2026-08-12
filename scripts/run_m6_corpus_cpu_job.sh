#!/usr/bin/env bash
# CPU-only success replay, SFT corpus audit, and frozen Raw-K8 curriculum.
#SBATCH --job-name=m6-corpus
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/m6_corpus_%j.out
#SBATCH --error=logs/m6_corpus_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
split_lock="${M6_SPLIT_LOCK:-$study_root/locks/m6_webshop_split_v1.json}"
if test -n "${M6_SERVICE_BASE_URL:-}"; then
  base_url="$M6_SERVICE_BASE_URL"
elif test -f "$study_root/service_base_url"; then
  IFS= read -r base_url < "$study_root/service_base_url"
else
  base_url="http://127.0.0.1:44151"
fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

curl --fail --silent --show-error --max-time 30 "$base_url/health" >/dev/null
curl --fail --silent --show-error --max-time 120 -X POST \
  -H 'Content-Type: application/json' -d '{"goal_index":1000}' "$base_url/reset" >/dev/null

collection_args=()
IFS=':' read -r -a collection_roots <<< "${M6_RAW_COLLECTION_ROOTS:-$study_root/mini/raw_collection}"
test "${#collection_roots[@]}" -ge 1 && test "${#collection_roots[@]}" -le 2
for collection_root in "${collection_roots[@]}"; do
  collection_args+=(--collection-root "$collection_root")
done
"$python_bin" scripts/m6_build_sft_corpus.py \
  "${collection_args[@]}" \
  --split-lock "$split_lock" \
  --goals outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json \
  --output-dir "$study_root/mini/corpus" --base-url "$base_url"

"$python_bin" scripts/m6_build_rl_curriculum.py \
  --groups-dir "$study_root/mini/raw_collection/groups" \
  --split-lock "$split_lock" \
  --output "$study_root/mini/rl_curriculum.json"
