#!/usr/bin/env bash
# Five-step single-variable follow-up: only policy credit changes from full to tail2.
#SBATCH --job-name=m6-p5-tail2-rl
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/m6_phase5_tail2_rl_%j.out
#SBATCH --error=logs/m6_phase5_tail2_rl_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_PHASE5_ROSTER:?set M6_PHASE5_ROSTER}"
: "${M6_PHASE5_ROSTER_PRODUCER_GIT_SHA:?set M6_PHASE5_ROSTER_PRODUCER_GIT_SHA}"
: "${M6_PHASE5_OUTPUT:?set M6_PHASE5_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

python_bin=/home/wushaohua/miniconda3/envs/miniwebwork/bin/python
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
sft_adapter="${M6_SFT_ADAPTER:-$study_root/mini/pilot_sft/final_adapter}"
split_lock="${M6_SPLIT_LOCK:-$study_root/locks/m6_webshop_split_v1.json}"
goals="${M6_GOALS:-outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json}"
base_url="${M6_SERVICE_BASE_URL:-http://127.0.0.1:44151}"
run_seed="${M6_RUN_SEED:-20260827}"
policy_credit_window="${M6_POLICY_CREDIT_WINDOW:-tail2}"
case "$policy_credit_window" in
  tail2|preterminal1) ;;
  *) echo "unsupported policy credit window: $policy_credit_window" >&2; exit 2 ;;
esac
steps=5
test -d "$sft_adapter"
test -f "$M6_PHASE5_ROSTER"
test -f "$split_lock"
test -f "$goals"
test ! -f "$M6_PHASE5_OUTPUT/run_report.json"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTORCH_CUDA_ALLOC_CONF

curl --fail --silent --show-error --max-time 30 "$base_url/health" >/dev/null
"$python_bin" - "$M6_PHASE5_ROSTER" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
if value.get("task_count") != 40 or len(value.get("task_ids", [])) != 40:
    raise ValueError("Phase5 online roster must contain exactly 40 tasks")
PY

mkdir -p "$M6_PHASE5_OUTPUT"
current_adapter="$sft_adapter"
current_optimizer=""
step=0
while test "$step" -lt "$steps"; do
  step_root="$M6_PHASE5_OUTPUT/step_$(printf '%02d' "$step")"
  collection_root="$step_root/collection"
  learner_root="$step_root/learner"
  if ! test -f "$collection_root/collection_report.json"; then
    "$slurm_bin/srun" --ntasks=1 "$python_bin" scripts/m6_collect_policy_success.py \
      --mode phase4_online_rl_collection --role train --k 4 \
      --split-lock "$split_lock" --goals "$goals" \
      --task-roster "$M6_PHASE5_ROSTER" \
      --task-roster-producer-git-sha "$M6_PHASE5_ROSTER_PRODUCER_GIT_SHA" \
      --task-offset "$((step * 4))" --maximum-tasks 4 \
      --max-model-turns 18 --max-environment-steps 15 --maximum-action-tokens 75000 \
      --concurrent-groups 4 --base-url "$base_url" \
      --output-dir "$collection_root" --iteration-index "$step" --seed "$run_seed" \
      --adapter "$current_adapter"
  fi
  optimizer_args=()
  test -z "$current_optimizer" || optimizer_args=(--input-optimizer "$current_optimizer")
  if ! test -f "$learner_root/learner_report.json"; then
    "$slurm_bin/srun" --ntasks=1 "$python_bin" scripts/m6_phase4_online_update.py \
      --collection-root "$collection_root" \
      --input-adapter "$current_adapter" --reference-sft-adapter "$sft_adapter" \
      --output-dir "$learner_root" --iteration-index "$step" --seed "$run_seed" \
      --microbatch-size 4 --policy-credit-window "$policy_credit_window" "${optimizer_args[@]}"
  fi
  current_adapter="$("$python_bin" -c 'import json,sys;print(json.load(open(sys.argv[1]))["output_adapter"])' "$learner_root/learner_report.json")"
  current_optimizer="$("$python_bin" -c 'import json,sys;print(json.load(open(sys.argv[1]))["output_optimizer"])' "$learner_root/learner_report.json")"
  step=$((step + 1))
done

"$python_bin" - "$M6_PHASE5_OUTPUT" "$current_adapter" "$current_optimizer" "$run_seed" "$sft_adapter" "$policy_credit_window" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
policy_credit_window=sys.argv[6]
reports=[json.load(open(root/f"step_{i:02d}"/"learner"/"learner_report.json")) for i in range(5)]
if any(row.get("policy_credit_window") != policy_credit_window for row in reports):
    raise ValueError("Phase5 learner policy-credit drift")
value={
    "schema_version":"m6_policy_window_online_run_v1",
    "complete":True,
    "method":f"strict_binary_{policy_credit_window}_trajectory_group_normalized_policy_gradient_with_sft_kl",
    "reward":"strict_binary",
    "policy_credit_window":policy_credit_window,
    "seed":int(sys.argv[4]),
    "optimizer_steps":sum(row["optimizer_steps"] for row in reports),
    "attempted_unique_tasks":len({group["task_id"] for row in reports for group in row["group_rows"]}),
    "active_mixed_task_groups":sum(row["active_mixed_task_groups"] for row in reports),
    "final_adapter":sys.argv[2],
    "final_optimizer":sys.argv[3],
    "maximum_reference_kl":max(row["reference_kl"] for row in reports),
    "maximum_initial_replay_clip_fraction":max(row["initial_replay_clip_fraction"] for row in reports),
    "checkpoints":{"step_0":sys.argv[5],"step_5":reports[4]["output_adapter"]},
}
(root/"run_report.json").write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
print(json.dumps(value,indent=2,sort_keys=True))
PY
