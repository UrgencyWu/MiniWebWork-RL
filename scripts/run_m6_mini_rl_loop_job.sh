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
medium_authorization="${M6_MEDIUM_AUTHORIZATION:-}"
run_seed="${M6_RUN_SEED:-20260812}"
maximum_iterations="${M6_MAXIMUM_ITERATIONS:-12}"
target_mixed_iterations="${M6_TARGET_MIXED_ITERATIONS:-5}"
total_action_token_cap="${M6_TOTAL_ACTION_TOKEN_CAP:-50000}"
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
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --mem="${M6_JOB_MEMORY:-24G}" --gres=gpu:1 --time=24:00:00 \
      --export="ALL,M6_EXPECTED_GIT_SHA=$M6_EXPECTED_GIT_SHA,M6_METHOD=$M6_METHOD,M6_PILOT_AUTHORIZATION=$M6_PILOT_AUTHORIZATION,M6_MEDIUM_AUTHORIZATION=$medium_authorization,M6_RUN_SEED=$run_seed,M6_MAXIMUM_ITERATIONS=$maximum_iterations,M6_TARGET_MIXED_ITERATIONS=$target_mixed_iterations,M6_TOTAL_ACTION_TOKEN_CAP=$total_action_token_cap,M6_SERVICE_BASE_URL=$base_url,M6_SPLIT_LOCK=$split_lock,M6_RL_OUTPUT=$rl_root,M6_SFT_ADAPTER=$sft_adapter,M6_CURRICULUM=$curriculum,M6_CURRICULUM_PRODUCER_GIT_SHA=$curriculum_producer_git_sha,M6_LEARNER_MICROBATCH_SIZE=$learner_microbatch_size,M6_REPLAY_PARITY_CALIBRATION=$replay_parity_calibration,M6_SFT_GATE=$sft_gate,M6_RAW_EVAL_REPORT=$raw_eval_report,M6_SFT_EVAL_REPORT=$sft_eval_report,M6_CORPUS_AUDIT=$corpus_audit" \
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
if test -n "$medium_authorization"; then
  "$python_bin" - "$medium_authorization" "$run_seed" "$maximum_iterations" "$target_mixed_iterations" "$total_action_token_cap" "$curriculum" "$rl_root" "$M6_METHOD" <<'PY'
import json
import sys
from pathlib import Path
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_file, sha256_json
from miniwebwork.m6_pilot import load_medium_rl_authorization

authorization = load_medium_rl_authorization(Path(sys.argv[1]))
contract = authorization["payload"]
seed = int(sys.argv[2])
maximum_iterations = int(sys.argv[3])
target_updates = int(sys.argv[4])
token_cap = int(sys.argv[5])
curriculum = json.loads(Path(sys.argv[6]).read_text(encoding="utf-8"))
root = Path(sys.argv[7])
method = sys.argv[8]
controls = contract["shared_controls"]
if method not in contract["approved_methods"] or seed not in contract["paired_seeds"]:
    raise ValueError("M6 medium method/seed is not authorized")
if maximum_iterations != controls["maximum_iterations"]:
    raise ValueError("M6 medium maximum-iteration drift")
if target_updates != controls["target_mixed_optimizer_updates"]:
    raise ValueError("M6 medium mixed-update target drift")
if token_cap != controls["generated_action_token_cap_per_run"]:
    raise ValueError("M6 medium token-budget drift")
if curriculum.get("task_count") != controls["curriculum_candidate_tasks"]:
    raise ValueError("M6 medium curriculum size drift")
if curriculum.get("medium_authorization_file_sha256") != authorization["sha256"]:
    raise ValueError("M6 medium curriculum authorization binding drift")
run = {
    "schema_version": "m6_medium_rl_run_v1",
    "development_only": True,
    "formal_training": False,
    "method": method,
    "seed": seed,
    "maximum_iterations": maximum_iterations,
    "target_mixed_optimizer_updates": target_updates,
    "generated_action_token_cap": token_cap,
    "curriculum_content_sha256": curriculum["content_sha256"],
    "medium_authorization_file_sha256": sha256_file(Path(sys.argv[1])),
}
run["content_sha256"] = sha256_json(run)
destination = root / "medium_run_manifest.json"
if destination.is_file():
    if json.loads(destination.read_text(encoding="utf-8")) != run:
        raise ValueError("M6 recovered medium run contract drift")
else:
    atomic_write_json(destination, run)
PY
else
  test "$run_seed" = "20260812"
  test "$maximum_iterations" = "12"
  test "$target_mixed_iterations" = "5"
  test "$total_action_token_cap" = "50000"
fi
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
while test "$iteration" -lt "$maximum_iterations" && test "$mixed_iterations" -lt "$target_mixed_iterations"; do
  remaining_tokens=$((total_action_token_cap - total_tokens))
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
      --output-dir "$collection_root" --iteration-index "$iteration" --seed "$run_seed" --adapter "$input_adapter"
    collection_mixed="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["mixed_strict_reward_group_count"])' "$collection_root/collection_report.json")"
  fi

  tokens="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["all_attempt_generated_action_tokens"])' "$collection_root/collection_report.json")"
  total_tokens=$((total_tokens + tokens))
  if test "$total_tokens" -gt "$total_action_token_cap"; then
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
  collection_bridge_args=()
  collection_producer_git_sha="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_sha"])' "$collection_root/collection_report.json")"
  if test "$collection_producer_git_sha" != "$M6_EXPECTED_GIT_SHA"; then
    collection_bridge_args=(--collection-producer-git-sha "$collection_producer_git_sha")
  fi
  input_state_bridge_args=()
  if test -n "$latest_learner_report"; then
    input_state_producer_git_sha="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_sha"])' "$latest_learner_report")"
    if test "$input_state_producer_git_sha" != "$M6_EXPECTED_GIT_SHA"; then
      input_state_bridge_args=(--input-state-producer-git-sha "$input_state_producer_git_sha")
    fi
  fi
  if ! test -f "$learner_root/learner_report.json"; then
    set +e
    "$slurm_bin/srun" --ntasks=1 "$python_bin" scripts/m6_online_rl.py \
      --groups-dir "$collection_root/groups" --collection-report "$collection_root/collection_report.json" \
      --input-adapter "$input_adapter" --input-adapter-semantic-sha256 "$input_semantic" \
      --reference-sft-adapter "$sft_adapter" --reference-sft-adapter-semantic-sha256 "$reference_sft_semantic" \
      --replay-parity-calibration "$replay_parity_calibration" --output-dir "$learner_root" \
      --iteration-index "$mixed_iterations" --seed "$run_seed" --method "$M6_METHOD" \
      --pilot-authorization "$M6_PILOT_AUTHORIZATION" --microbatch-size "$learner_microbatch_size" \
      --parity-rejection-report "$iteration_root/parity_rejection.json" \
      "${optimizer_args[@]}" "${collection_bridge_args[@]}" "${input_state_bridge_args[@]}"
    learner_status=$?
    set -e
    if test "$learner_status" -eq 42; then
      test -f "$iteration_root/parity_rejection.json"
      printf '%s\n' "behavior_hf_replay_parity_rejected_no_update" > "$iteration_root/skip_reason"
      iteration=$((iteration + 1))
      continue
    fi
    test "$learner_status" -eq 0 || exit "$learner_status"
  fi
  iteration_report="$learner_root/learner_report.json"
  mixed="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["collection_audit"]["mixed_strict_reward_group_count"] > 0))' "$iteration_report")"
  mixed_iterations=$((mixed_iterations + mixed))
  latest_learner_report="$iteration_report"
  iteration=$((iteration + 1))
done
test "$mixed_iterations" -ge "$target_mixed_iterations"

audit_args=()
collection_args=()
parity_rejection_args=()
index=0
while test "$index" -lt "$iteration"; do
  collection_args+=(--collection-report "$rl_root/iteration_${index}/collection/collection_report.json")
  if test -f "$rl_root/iteration_${index}/parity_rejection.json"; then
    parity_rejection_args+=(--parity-rejection "$rl_root/iteration_${index}/parity_rejection.json")
  fi
  if test -f "$rl_root/iteration_${index}/learner/learner_report.json"; then
    audit_args+=(--iteration "$rl_root/iteration_${index}/learner/learner_report.json")
    audit_args+=(--groups-dir "$rl_root/iteration_${index}/collection/groups")
  fi
  index=$((index + 1))
done
"$slurm_bin/srun" --ntasks=1 "$python_bin" scripts/m6_finalize_rl_audit.py \
  "${audit_args[@]}" "${collection_args[@]}" "${parity_rejection_args[@]}" --method "$M6_METHOD" \
  --pilot-authorization "$M6_PILOT_AUTHORIZATION" --output "$rl_root/rl_audit.json"
