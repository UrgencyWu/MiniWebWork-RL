#!/usr/bin/env bash
# Six development-only medium RL tasks, throttled to the frozen four-GPU limit.
#SBATCH --job-name=m6-medium-rl
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%6
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m6_medium_rl_%A_%a.out
#SBATCH --error=logs/m6_medium_rl_%A_%a.err

set -euo pipefail
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_MEDIUM_ROOT:?set M6_MEDIUM_ROOT}"
: "${M6_MEDIUM_AUTHORIZATION:?set M6_MEDIUM_AUTHORIZATION}"
: "${M6_PILOT_AUTHORIZATION:?set M6_PILOT_AUTHORIZATION}"
: "${M6_CURRICULUM:?set M6_CURRICULUM}"

case "${SLURM_ARRAY_TASK_ID:?missing Slurm array task id}" in
  0) method=multi_turn_grpo; seed=20260812 ;;
  1) method=anchor_gigpo; seed=20260812 ;;
  2) method=multi_turn_grpo; seed=20260813 ;;
  3) method=anchor_gigpo; seed=20260813 ;;
  4) method=multi_turn_grpo; seed=20260814 ;;
  5) method=anchor_gigpo; seed=20260814 ;;
  *) echo "unsupported matrix index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export M6_METHOD="$method"
export M6_RUN_SEED="$seed"
export M6_RL_OUTPUT="$M6_MEDIUM_ROOT/seed_${seed}/${method}"
export M6_MAXIMUM_ITERATIONS=32
export M6_TARGET_MIXED_ITERATIONS=20
export M6_TOTAL_ACTION_TOKEN_CAP=50000
export M6_DISABLE_SUCCESSOR=0

exec bash scripts/run_m6_mini_rl_loop_job.sh
