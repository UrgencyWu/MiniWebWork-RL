#!/usr/bin/env bash
#SBATCH --job-name=m6-p6-preterminal-rl
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase6_preterminal_rl_%j.out
#SBATCH --error=logs/m6_phase6_preterminal_rl_%j.err

set -euo pipefail
: "${M6_PHASE6_ROSTER:?set M6_PHASE6_ROSTER}"
: "${M6_PHASE6_ROSTER_PRODUCER_GIT_SHA:?set M6_PHASE6_ROSTER_PRODUCER_GIT_SHA}"
: "${M6_PHASE6_OUTPUT:?set M6_PHASE6_OUTPUT}"
export M6_PHASE5_ROSTER="$M6_PHASE6_ROSTER"
export M6_PHASE5_ROSTER_PRODUCER_GIT_SHA="$M6_PHASE6_ROSTER_PRODUCER_GIT_SHA"
export M6_PHASE5_OUTPUT="$M6_PHASE6_OUTPUT"
export M6_POLICY_CREDIT_WINDOW=preterminal1
exec bash scripts/run_m6_phase5_tail2_online_rl_job.sh
