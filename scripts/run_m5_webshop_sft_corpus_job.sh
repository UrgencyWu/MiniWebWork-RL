#!/usr/bin/env bash
# Build verified public-action WebShop SFT labels against the shared service.
# This CPU job never submits model training and never opens the frozen test set.
#SBATCH --job-name=m5-webshop-sft-data
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/m5_webshop_sft_data_%j.out
#SBATCH --error=logs/m5_webshop_sft_data_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 preflight commit SHA}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

study_root="$repo_root/outputs/m5_webshop_credit_assignment_v1"
runtime_root="$study_root/upstream/webshop_full"
output_root="$study_root/preflight/sft_corpus"
python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
base_url="${WEBSHOP_ENV_BASE_URL:-http://127.0.0.1:44151}"
service_workers="${M5_WEBSHOP_WORKERS:-16}"
case "$service_workers" in 8|16) ;; *) exit 2 ;; esac
health_audit="$study_root/preflight/server/health_workers_${service_workers}.json"
export CUDA_VISIBLE_DEVICES=""

"$python_bin" -c 'import sys,urllib.request; url=sys.argv[1] + "/health"; print(urllib.request.urlopen(url, timeout=30).read().decode())' "$base_url"
"$python_bin" scripts/build_m5_webshop_sft_corpus.py \
  --goals "$runtime_root/goals.json" \
  --output-dir "$output_root" \
  --base-url "$base_url" \
  --workers 16 \
  --health-audit "$health_audit"
"$python_bin" scripts/audit_m5_webshop_sft_tokens.py \
  --data-dir "$output_root" \
  --base-model /data/share/model/Qwen3.5-4B
