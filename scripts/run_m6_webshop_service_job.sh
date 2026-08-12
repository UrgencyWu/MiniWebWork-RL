#!/usr/bin/env bash
# Shared CPU-only WebShop service for M6 mini. One service is reused by every
# collection/evaluation stage and renews only on a genuine 24h allocation edge.
#SBATCH --job-name=m6-webshop-env
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m6_webshop_env_%j.out
#SBATCH --error=logs/m6_webshop_env_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
mkdir -p outputs/m6_monotonic_posttraining_v1
: "${M6_EXPECTED_GIT_SHA:?set the frozen M6 commit SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

study_root="$repo_root/outputs/m5_webshop_credit_assignment_v1"
public_study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
environment_root="$study_root/server_environment"
upstream_root="$study_root/upstream/Agent-R1"
runtime_root="$study_root/upstream/webshop_full"
workers="${M6_WEBSHOP_WORKERS:-8}"
case "$workers" in 8|16) ;; *) echo "M6_WEBSHOP_WORKERS must be 8 or 16" >&2; exit 2 ;; esac

export CUDA_VISIBLE_DEVICES=""
export JAVA_HOME="$environment_root"
export JVM_PATH="$environment_root/lib/jvm/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$repo_root/src:$upstream_root"
export WEBSHOP_DATASET_MODE=full
export WEBSHOP_DATA_DIR="$runtime_root"
export WEBSHOP_INDEX_DIR="$runtime_root"
export WEBSHOP_SEARCH_TOP_K=50
export WEBSHOP_ENV_LOG_SEARCH=0
export WEBSHOP_ENV_LOG_STEPS=0
export WEBSHOP_ENV_WORKERS="$workers"
export WEBSHOP_ENV_PORT=44151
bind_address="${M6_WEBSHOP_BIND_ADDRESS:-0.0.0.0}"
service_host="$(hostname -f 2>/dev/null || hostname)"
service_base_url="http://${service_host}:44151"

cd "$upstream_root"
if ss -ltn | awk '{print $4}' | grep -Eq '(^|:|\])44151$'; then
  echo "TCP port 44151 is already in use" >&2
  exit 2
fi

service_pid=""
renew_service() {
  trap - USR1 TERM INT
  test -z "$service_pid" || kill -TERM "$service_pid" 2>/dev/null || true
  test -z "$service_pid" || wait "$service_pid" 2>/dev/null || true
  if test -n "${SLURM_JOB_ID:-}" && test "${M6_DISABLE_SUCCESSOR:-0}" != "1"; then
    cd "$repo_root"
    successor_job_id="$(sbatch --parsable \
      --dependency="afterany:${SLURM_JOB_ID}" \
      --export="ALL,M6_REPO_ROOT=$repo_root,M6_EXPECTED_GIT_SHA=$M6_EXPECTED_GIT_SHA,M6_WEBSHOP_WORKERS=$workers" \
      scripts/run_m6_webshop_service_job.sh)"
    printf '%s\n' "$successor_job_id" > "$repo_root/outputs/m6_monotonic_posttraining_v1/service_successor_job_id"
    echo "timeout_successor_job_id=$successor_job_id"
  fi
  exit 99
}
stop_service() {
  trap - USR1 TERM INT
  test -z "$service_pid" || kill -TERM "$service_pid" 2>/dev/null || true
  test -z "$service_pid" || wait "$service_pid" 2>/dev/null || true
  exit 143
}
trap renew_service USR1
trap stop_service TERM INT

service_url_tmp="$public_study_root/.service_base_url.${SLURM_JOB_ID:-$$}.tmp"
service_host_tmp="$public_study_root/.service_host.${SLURM_JOB_ID:-$$}.tmp"
printf '%s\n' "$service_base_url" > "$service_url_tmp"
printf '%s\n' "$service_host" > "$service_host_tmp"
mv "$service_url_tmp" "$public_study_root/service_base_url"
mv "$service_host_tmp" "$public_study_root/service_host"
"$environment_root/bin/gunicorn" -w "$workers" -k uvicorn.workers.UvicornWorker \
  miniwebwork.webshop_rl.serialized_service:app -b "$bind_address:44151" \
  --timeout 120 --graceful-timeout 60 --log-level info &
service_pid="$!"
wait "$service_pid"
