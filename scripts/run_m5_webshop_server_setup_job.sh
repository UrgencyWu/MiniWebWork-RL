#!/usr/bin/env bash
# Create the isolated Java/Pyserini server environment and pin Agent-R1 code.
# No GPU is requested and this job does not start model training.
#SBATCH --job-name=m5-webshop-setup
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/m5_webshop_setup_%j.out
#SBATCH --error=logs/m5_webshop_setup_%j.err

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
audit_path="$study_root/preflight/server/environment_audit.json"
conda_bin="/home/wushaohua/miniconda3/bin/conda"
revision="b124aa46534cbf2fb8bc8af11405774984c42ac7"

if test ! -x "$environment_root/bin/python"; then
  "$conda_bin" create -y -p "$environment_root" -c conda-forge python=3.12 openjdk=21 pip
else
  # Repair an environment whose earlier 24h allocation ended during creation.
  "$conda_bin" install -y -p "$environment_root" -c conda-forge python=3.12 openjdk=21 pip
fi
"$environment_root/bin/python" -m pip install --disable-pip-version-check -r requirements.m5-webshop-server.txt

if test ! -d "$upstream_root/.git"; then
  mkdir -p "$upstream_root"
  git -C "$upstream_root" init
  git -C "$upstream_root" remote add origin https://github.com/AgentR1/Agent-R1.git
fi
git -C "$upstream_root" fetch --depth 1 origin "$revision"
git -C "$upstream_root" checkout --detach "$revision"
test -z "$(git -C "$upstream_root" status --porcelain --untracked-files=no)"

export JAVA_HOME="$environment_root"
export JVM_PATH="$environment_root/lib/jvm/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
"$environment_root/bin/python" scripts/m5_webshop_server_preflight.py environment \
  --upstream-root "$upstream_root" \
  --output "$audit_path"
