#!/usr/bin/env bash
# Shared WebShop service for SFT verification, preflight and formal rollouts.
# One 8-CPU service avoids multiplying CPU requests across six GPU jobs.
#SBATCH --job-name=m5-webshop-env
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/m5_webshop_env_%j.out
#SBATCH --error=logs/m5_webshop_env_%j.err

set -euo pipefail
repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
mkdir -p logs
: "${M5_EXPECTED_GIT_SHA:?set the frozen M5 preflight commit SHA}"
test "$(git rev-parse HEAD)" = "$M5_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

study_root="$repo_root/outputs/m5_webshop_credit_assignment_v1"
environment_root="$study_root/server_environment"
upstream_root="$study_root/upstream/Agent-R1"
runtime_root="$study_root/upstream/webshop_full"
data_audit="$study_root/preflight/data/runtime_audit.json"
environment_audit="$study_root/preflight/server/environment_audit.json"
launch_audit_root="$study_root/preflight/server_launch/${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}"
python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
workers="${M5_WEBSHOP_WORKERS:-4}"
export CUDA_VISIBLE_DEVICES=""

case "$workers" in
  2|4|8) ;;
  *) echo "M5_WEBSHOP_WORKERS must be one of 2, 4, 8" >&2; exit 2 ;;
esac

test -s "$data_audit"
test -s "$environment_audit"
test -x "$environment_root/bin/python"
test "$(git -C "$upstream_root" rev-parse HEAD)" = "b124aa46534cbf2fb8bc8af11405774984c42ac7"
mkdir -p "$launch_audit_root"

# Revalidate the exact bytes and interpreter at every 24h service allocation.
"$python_bin" scripts/m5_webshop_data_preflight.py audit \
  --runtime-root "$runtime_root" \
  --output "$launch_audit_root/runtime_audit.json" \
  --reference-audit "$data_audit"
"$environment_root/bin/python" scripts/m5_webshop_server_preflight.py environment \
  --upstream-root "$upstream_root" \
  --output "$launch_audit_root/environment_audit.json" \
  --reference-audit "$environment_audit"

export JAVA_HOME="$environment_root"
export JVM_PATH="$environment_root/lib/jvm/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$upstream_root"
export WEBSHOP_DATASET_MODE=full
export WEBSHOP_DATA_DIR="$runtime_root"
export WEBSHOP_INDEX_DIR="$runtime_root"
export WEBSHOP_SEARCH_TOP_K=50
export WEBSHOP_ENV_LOG_SEARCH=0
export WEBSHOP_ENV_LOG_STEPS=0
export WEBSHOP_ENV_WORKERS="$workers"
export WEBSHOP_ENV_PORT=44151

cd "$upstream_root"

if ss -ltn | awk '{print $4}' | grep -Eq '(^|:|\])44151$'; then
  echo "refusing to start: TCP port 44151 is already in use" >&2
  exit 2
fi

service_pid=""
renew_service() {
  trap - USR1 TERM INT
  if test -n "$service_pid"; then
    kill -TERM "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
  if test -n "${SLURM_JOB_ID:-}"; then
    cd "$repo_root"
    successor_job_id="$(sbatch --parsable \
      --dependency="afterany:${SLURM_JOB_ID}" \
      --export="ALL,M5_EXPECTED_GIT_SHA=${M5_EXPECTED_GIT_SHA},M5_WEBSHOP_WORKERS=${workers}" \
      "$repo_root/scripts/run_m5_webshop_service_job.sh")"
    echo "renewal_parent_job_id=$SLURM_JOB_ID"
    echo "renewal_successor_job_id=$successor_job_id"
  fi
  exit 0
}
stop_service() {
  trap - USR1 TERM INT
  if test -n "$service_pid"; then
    kill -TERM "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
  exit 143
}
trap renew_service USR1
trap stop_service TERM INT

"$environment_root/bin/gunicorn" \
  -w "$workers" \
  -k uvicorn.workers.UvicornWorker \
  recipes.webshop.env.server:app \
  -b 127.0.0.1:44151 \
  --timeout 120 \
  --graceful-timeout 60 \
  --log-level info &
service_pid="$!"
wait "$service_pid"
