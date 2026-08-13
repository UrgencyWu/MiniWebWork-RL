#!/usr/bin/env bash
# Recoverable development-only online loop: K8 collect -> selected GRPO-family update.
# One GPU is reused sequentially by vLLM collection and HF learning.
#SBATCH --job-name=m6-mini-rl-loop
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/m6_mini_rl_loop_%j.out
#SBATCH --error=logs/m6_mini_rl_loop_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_METHOD:?set multi_turn_grpo or anchor_gigpo}"
: "${M6_PILOT_AUTHORIZATION:?set M6_PILOT_AUTHORIZATION}"
case "$M6_METHOD" in
  multi_turn_grpo|anchor_gigpo) ;;
  *) echo "unsupported M6_METHOD=$M6_METHOD" >&2; exit 2 ;;
esac
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
rl_root="${M6_RL_OUTPUT:-$study_root/mini/pilot_rl/$M6_METHOD}"
sft_adapter="${M6_SFT_ADAPTER:-$study_root/mini/pilot_sft/final_adapter}"
curriculum="${M6_CURRICULUM:-$study_root/mini/rl_curriculum_v2.json}"
curriculum_producer_git_sha="${M6_CURRICULUM_PRODUCER_GIT_SHA:-}"
learner_microbatch_size="${M6_LEARNER_MICROBATCH_SIZE:-4}"
replay_parity_calibration="${M6_REPLAY_PARITY_CALIBRATION:-$repo_root/data/m6_mini_replay_parity_calibration_v1.json}"
sft_gate="${M6_SFT_GATE:-$study_root/mini/pilot_sft_gate.json}"
raw_eval_report="${M6_RAW_EVAL_REPORT:-$study_root/mini/pilot_raw_eval/identity_report.json}"
sft_eval_report="${M6_SFT_EVAL_REPORT:-$study_root/mini/pilot_sft_eval/identity_report.json}"
corpus_audit="${M6_CORPUS_AUDIT:-$study_root/mini/corpus_v2/corpus_audit.json}"
split_lock="${M6_SPLIT_LOCK:-$study_root/locks/m6_webshop_split_v1.json}"
case "$learner_microbatch_size" in
  1|2|4|8) ;;
  *) echo "unsupported M6_LEARNER_MICROBATCH_SIZE=$learner_microbatch_size" >&2; exit 2 ;;
esac
if test -n "${M6_SERVICE_BASE_URL:-}"; then
  base_url="$M6_SERVICE_BASE_URL"
elif test -f "$study_root/service_base_url"; then
  IFS= read -r base_url < "$study_root/service_base_url"
else
  base_url="http://127.0.0.1:44151"
fi
mkdir -p "$rl_root"

submit_timeout_successor() {
  trap - USR1
  if test -n "${SLURM_JOB_ID:-}" && test "${M6_DISABLE_SUCCESSOR:-0}" != "1"; then
    successor_job_id="$("$slurm_bin/sbatch" --parsable --dependency="afterany:${SLURM_JOB_ID}" \
      --export="ALL,M6_EXPECTED_GIT_SHA=$M6_EXPECTED_GIT_SHA,M6_METHOD=$M6_METHOD,M6_PILOT_AUTHORIZATION=$M6_PILOT_AUTHORIZATION,M6_SERVICE_BASE_URL=$base_url,M6_SPLIT_LOCK=$split_lock,M6_RL_OUTPUT=$rl_root,M6_SFT_ADAPTER=$sft_adapter,M6_CURRICULUM=$curriculum,M6_CURRICULUM_PRODUCER_GIT_SHA=$curriculum_producer_git_sha,M6_LEARNER_MICROBATCH_SIZE=$learner_microbatch_size,M6_REPLAY_PARITY_CALIBRATION=$replay_parity_calibration,M6_SFT_GATE=$sft_gate,M6_RAW_EVAL_REPORT=$raw_eval_report,M6_SFT_EVAL_REPORT=$sft_eval_report,M6_CORPUS_AUDIT=$corpus_audit" \
      scripts/run_m6_mini_rl_loop_job.sh)"
    printf '%s\n' "$successor_job_id" > "$rl_root/successor_job_id"
    echo "timeout_successor_job_id=$successor_job_id"
  fi
  exit 99
}
trap submit_timeout_successor USR1

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTORCH_CUDA_ALLOC_CONF

curl --fail --silent --show-error --max-time 30 "$base_url/health" >/dev/null
curl --fail --silent --show-error --max-time 120 -X POST \
  -H 'Content-Type: application/json' -d '{"goal_index":1000}' "$base_url/reset" >/dev/null
test -d "$sft_adapter"
test -f "$curriculum"
test -f "$sft_gate"
test -f "$replay_parity_calibration"
"$python_bin" - "$sft_gate" "$raw_eval_report" "$sft_eval_report" "$corpus_audit" "$M6_PILOT_AUTHORIZATION" "$M6_METHOD" <<'PY'
import json
import sys
from pathlib import Path
from miniwebwork.m6_mini import validate_closed_loop_identity, validate_mini_sft_gate
from miniwebwork.m6_pilot import validate_pilot_authorization, validate_pilot_method
from miniwebwork.webshop_rl.m6_corpus import validate_conditional_learnability_audit

gate = validate_mini_sft_gate(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
raw = validate_closed_loop_identity(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
sft = validate_closed_loop_identity(json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")))
corpus = validate_conditional_learnability_audit(json.loads(Path(sys.argv[4]).read_text(encoding="utf-8")))
pilot = validate_pilot_authorization(json.loads(Path(sys.argv[5]).read_text(encoding="utf-8")), corpus_audit=corpus)
validate_pilot_method(sys.argv[6], pilot)
if gate.get("passed") is not True or gate.get("decision") != "ALLOW_MINI_RL":
    raise ValueError("M6 mini SFT gate did not authorize RL")
if gate["raw_content_sha256"] != raw["content_sha256"]:
    raise ValueError("M6 SFT gate Raw evaluation binding drift")
if gate["sft_content_sha256"] != sft["content_sha256"]:
    raise ValueError("M6 SFT gate SFT evaluation binding drift")
if gate["corpus_audit_content_sha256"] != corpus["content_sha256"]:
    raise ValueError("M6 SFT gate corpus binding drift")
if gate.get("pilot_authorization_content_sha256") != pilot["content_sha256"]:
    raise ValueError("M6 SFT gate pilot-authorization binding drift")
PY

identity="$rl_root/sft_adapter_identity.json"
# Revalidation is cheap and proves the existing view still represents the
# canonical SFT adapter before every 24h allocation/resume.
"$slurm_bin/srun" --ntasks=1 "$python_bin" scripts/m6_adapter_identity.py \
  --adapter "$sft_adapter" --view "$rl_root/sft_rollout_adapter" --output "$identity"

curriculum_tasks="$($python_bin -c 'import json,sys; value=json.load(open(sys.argv[1])); print(value["task_count"])' "$curriculum")"
curriculum_producer_args=()
test -z "$curriculum_producer_git_sha" || curriculum_producer_args=(--task-roster-producer-git-sha "$curriculum_producer_git_sha")
"$python_bin" - "$curriculum" "$M6_PILOT_AUTHORIZATION" <<'PY'
import json
import sys
from pathlib import Path
from miniwebwork.long_horizon_rl.contracts import sha256_json
from miniwebwork.m6_pilot import validate_pilot_authorization

curriculum = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = dict(curriculum)
observed = expected.pop("content_sha256", None)
if observed != sha256_json(expected):
    raise ValueError("M6 RL curriculum self-hash drift")
pilot = validate_pilot_authorization(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
if curriculum.get("pilot_authorization_content_sha256") != pilot["content_sha256"]:
    raise ValueError("M6 RL curriculum pilot-authorization binding drift")
PY
maximum_iterations=12
test "$curriculum_tasks" -ge "$maximum_iterations" || maximum_iterations="$curriculum_tasks"

iteration=0
mixed_iterations=0
total_tokens=0
latest_learner_report=""
while test -f "$rl_root/iteration_${iteration}/collection/collection_report.json"; do
  collection_report="$rl_root/iteration_${iteration}/collection/collection_report.json"
  learner_report="$rl_root/iteration_${iteration}/learner/learner_report.json"
  skip_reason="$rl_root/iteration_${iteration}/skip_reason"
  if ! test -f "$learner_report" && ! test -f "$skip_reason"; then
    break
  fi
  tokens="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["all_attempt_generated_action_tokens"])' "$collection_report")"
  total_tokens=$((total_tokens + tokens))
  if test -f "$learner_report"; then
    latest_learner_report="$learner_report"
    mixed_iterations=$((mixed_iterations + 1))
  fi
  iteration=$((iteration + 1))
done
while test "$iteration" -lt "$maximum_iterations" && test "$mixed_iterations" -lt 5; do
  remaining_tokens=$((50000 - total_tokens))
  # A complete K8 attempt is 8 rollouts * 6 turns * 128 tokens = 6,144.
  # The 12,288 per-iteration ceiling leaves room for one infrastructure retry,
  # but the final global-budget slice may safely allow just one full attempt.
  if test "$remaining_tokens" -lt 6144; then
    echo "M6 cumulative RL budget cannot reserve another complete K8 iteration" >&2
    exit 2
  fi
  iteration_token_cap=12288
  test "$remaining_tokens" -ge "$iteration_token_cap" || iteration_token_cap="$remaining_tokens"
  iteration_root="$rl_root/iteration_${iteration}"
  collection_root="$iteration_root/collection"
  learner_root="$iteration_root/learner"
  mkdir -p "$iteration_root"
  if test -z "$latest_learner_report"; then
    input_adapter="$sft_adapter"
    input_semantic="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter_semantic_sha256"])' "$identity")"
    reference_sft_semantic="$input_semantic"
    input_optimizer=""
  else
    input_adapter="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_adapter"])' "$latest_learner_report")"
    input_semantic="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_adapter_semantic_sha256"])' "$latest_learner_report")"
    reference_sft_semantic="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["reference_sft_adapter_semantic_sha256"])' "$latest_learner_report")"
    input_optimizer="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_optimizer"])' "$latest_learner_report")"
  fi

  # The curriculum is ranked by Raw K8 pass rate, but SFT can make a task
  # homogeneous.  Such a group is valid diagnostic evidence, yet it must not
  # consume an optimizer iteration: verifier-TD on homogeneous outcomes is
  # not the pre-specified strict mixed-signal test.  Move to the next frozen
  # task with the same input policy.
  if test -f "$collection_root/collection_report.json"; then
    collection_mixed="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["mixed_strict_reward_group_count"])' "$collection_root/collection_report.json")"
  else
    collection_mixed=""
  fi

  if ! test -f "$collection_root/collection_report.json"; then
    "$slurm_bin/srun" --ntasks=1 "$python_bin" scripts/m6_collect_policy_success.py \
      --mode rl_collection --role mini_train --k 8 \
      --split-lock "$split_lock" \
      --goals outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json \
      --task-roster "$curriculum" "${curriculum_producer_args[@]}" \
      --task-offset "$iteration" --maximum-tasks 1 \
      --max-model-turns 6 --max-environment-steps 6 --maximum-action-tokens "$iteration_token_cap" \
      --concurrent-groups 1 --base-url "$base_url" \
      --output-dir "$collection_root" --iteration-index "$iteration" --adapter "$input_adapter"
    collection_mixed="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["mixed_strict_reward_group_count"])' "$collection_root/collection_report.json")"
  fi

  tokens="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["all_attempt_generated_action_tokens"])' "$collection_root/collection_report.json")"
  total_tokens=$((total_tokens + tokens))
  if test "$total_tokens" -gt 50000; then
    echo "M6 cumulative RL token cap exceeded" >&2
    exit 2
  fi
  if test "$collection_mixed" -eq 0; then
    printf '%s\n' "homogeneous_strict_reward_no_update" > "$iteration_root/skip_reason"
    iteration=$((iteration + 1))
    continue
  fi

  optimizer_args=()
  test -z "$input_optimizer" || optimizer_args=(--input-optimizer "$input_optimizer")
  if ! test -f "$learner_root/learner_report.json"; then
    "$slurm_bin/srun" --ntasks=1 "$python_bin" scripts/m6_online_rl.py \
      --groups-dir "$collection_root/groups" --collection-report "$collection_root/collection_report.json" \
      --input-adapter "$input_adapter" --input-adapter-semantic-sha256 "$input_semantic" \
      --reference-sft-adapter "$sft_adapter" --reference-sft-adapter-semantic-sha256 "$reference_sft_semantic" \
      --replay-parity-calibration "$replay_parity_calibration" --output-dir "$learner_root" \
      --iteration-index "$mixed_iterations" --seed 20260812 --method "$M6_METHOD" \
      --pilot-authorization "$M6_PILOT_AUTHORIZATION" --microbatch-size "$learner_microbatch_size" \
      "${optimizer_args[@]}"
  fi
  iteration_report="$learner_root/learner_report.json"
  mixed="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["collection_audit"]["mixed_strict_reward_group_count"] > 0))' "$iteration_report")"
  mixed_iterations=$((mixed_iterations + mixed))
  latest_learner_report="$iteration_report"
  iteration=$((iteration + 1))
done
test "$mixed_iterations" -ge 5

audit_args=()
collection_args=()
index=0
while test "$index" -lt "$iteration"; do
  collection_args+=(--collection-report "$rl_root/iteration_${index}/collection/collection_report.json")
  if test -f "$rl_root/iteration_${index}/learner/learner_report.json"; then
    audit_args+=(--iteration "$rl_root/iteration_${index}/learner/learner_report.json")
    audit_args+=(--groups-dir "$rl_root/iteration_${index}/collection/groups")
  fi
  index=$((index + 1))
done
"$slurm_bin/srun" --ntasks=1 "$python_bin" scripts/m6_finalize_rl_audit.py \
  "${audit_args[@]}" "${collection_args[@]}" --method "$M6_METHOD" \
  --pilot-authorization "$M6_PILOT_AUTHORIZATION" --output "$rl_root/rl_audit.json"
