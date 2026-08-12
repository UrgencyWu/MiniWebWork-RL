#!/usr/bin/env bash
# Submit the complete fail-closed M6.1 development pilot DAG exactly once.

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
slurm_bin="${M6_SLURM_BIN:-/opt/slurm/slurm.25.05/bin}"
python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
cd "$repo_root"
: "${M6_EXPECTED_GIT_SHA:?set the frozen M6.1 consumer Git SHA}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
raw_root_a="$study_root/mini/raw_collection"
raw_root_b="$study_root/mini/raw_collection_seed_20260813"
corpus="$study_root/mini/corpus_v2"
curriculum="$study_root/mini/rl_curriculum_v2.json"
sft="$study_root/mini/pilot_sft"
raw_eval="$study_root/mini/pilot_raw_eval"
sft_eval="$study_root/mini/pilot_sft_eval"
sft_gate="$study_root/mini/pilot_sft_gate.json"
rl_root="$study_root/mini/pilot_rl"
pilot_waiver="$repo_root/data/m6_mini_pilot_waiver_v1.json"
pilot_authorization="$corpus/pilot_authorization.json"
submission="$study_root/mini/pilot_submission.json"
if test -n "${M6_SERVICE_BASE_URL:-}"; then
  base_url="$M6_SERVICE_BASE_URL"
else
  IFS= read -r base_url < "$study_root/service_base_url"
fi

for required in "$split_lock" "$pilot_waiver" \
  "$raw_root_a/collection_report.json" "$raw_root_a/invocation.json" \
  "$raw_root_b/collection_report.json" "$raw_root_b/invocation.json"; do
  test -f "$required"
done
for absent in "$corpus" "$curriculum" "$sft" "$raw_eval" "$sft_eval" \
  "$sft_gate" "$rl_root" "$submission"; do
  if test -e "$absent"; then
    echo "M6.1 output already exists: $absent" >&2
    exit 2
  fi
done
curl --fail --silent --show-error --max-time 30 "$base_url/health" >/dev/null
curl --fail --silent --show-error --max-time 120 -X POST \
  -H 'Content-Type: application/json' -d '{"goal_index":1000}' "$base_url/reset" >/dev/null

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" - "$pilot_waiver" "$split_lock" "$raw_root_a" "$raw_root_b" <<'PY'
import json
import sys
from pathlib import Path
from miniwebwork.long_horizon_rl.contracts import sha256_json
from miniwebwork.m6_pilot import load_pilot_waiver
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock

waiver = load_pilot_waiver(Path(sys.argv[1]))["payload"]
split = validate_split_lock(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
protocol = load_protocol()
if split["protocol_sha256"] != protocol["sha256"]:
    raise ValueError("M6.1 split/protocol drift")
observed_hashes = []
observed_seeds = []
for root_name in sys.argv[3:]:
    root = Path(root_name)
    report = json.loads((root / "collection_report.json").read_text(encoding="utf-8"))
    invocation = json.loads((root / "invocation.json").read_text(encoding="utf-8"))
    for value, label in ((report, "report"), (invocation, "invocation")):
        expected = dict(value)
        observed = expected.pop("content_sha256", None)
        if observed != sha256_json(expected):
            raise ValueError(f"M6.1 Raw {label} self-hash drift")
    if report.get("complete") is not True or report.get("mode") != "raw_collection" or report.get("K") != 8:
        raise ValueError("M6.1 Raw collection is incomplete")
    if report.get("git_sha") != waiver["source_producer_git_sha"] or report.get("protocol_sha256") != waiver["source_protocol_sha256"]:
        raise ValueError("M6.1 Raw producer lineage drift")
    if report.get("split_lock_content_sha256") != split["content_sha256"]:
        raise ValueError("M6.1 Raw split lineage drift")
    if report.get("invocation_content_sha256") != invocation.get("content_sha256"):
        raise ValueError("M6.1 Raw invocation/report binding drift")
    observed_hashes.append(report["content_sha256"])
    observed_seeds.append(invocation["seed"])
if observed_hashes != waiver["source_collection_report_content_sha256"]:
    raise ValueError("M6.1 Raw collection roster drift")
if observed_seeds != waiver["source_collection_seeds"]:
    raise ValueError("M6.1 Raw collection seed drift")
PY

common="ALL,M6_EXPECTED_GIT_SHA=$M6_EXPECTED_GIT_SHA,M6_SERVICE_BASE_URL=$base_url,M6_SPLIT_LOCK=$split_lock"
submitted_jobs=()
cancel_partial_submission() {
  status=$?
  trap - ERR INT TERM
  if test "$status" -ne 0 && test "${#submitted_jobs[@]}" -gt 0; then
    "$slurm_bin/scancel" "${submitted_jobs[@]}" || true
    echo "cancelled partial M6.1 submission: ${submitted_jobs[*]}" >&2
  fi
  exit "$status"
}
trap cancel_partial_submission ERR INT TERM

corpus_job="$($slurm_bin/sbatch --parsable \
  --export="$common,M6_RAW_COLLECTION_ROOTS=$raw_root_a:$raw_root_b,M6_CORPUS_OUTPUT=$corpus,M6_CURRICULUM_OUTPUT=$curriculum,M6_CURRICULUM_GROUPS=$raw_root_a/groups,M6_PILOT_WAIVER=$pilot_waiver" \
  scripts/run_m6_corpus_cpu_job.sh)"
submitted_jobs+=("$corpus_job")
sft_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$corpus_job" \
  --export="$common,M6_CORPUS_DIR=$corpus,M6_SFT_OUTPUT=$sft,M6_PILOT_AUTHORIZATION=$pilot_authorization" \
  scripts/run_m6_mini_sft_job.sh)"
submitted_jobs+=("$sft_job")
raw_eval_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$corpus_job" \
  --export="$common,M6_PILOT_AUTHORIZATION=$pilot_authorization,M6_ROLLOUT_MODE=evaluation,M6_ROLLOUT_ROLE=mini_dev,M6_ROLLOUT_K=4,M6_ROLLOUT_OUTPUT=$raw_eval,M6_ROLLOUT_SEED=20260812,M6_ROLLOUT_ITERATION=0,M6_EVAL_IDENTITY=raw" \
  scripts/run_m6_rollout_job.sh)"
submitted_jobs+=("$raw_eval_job")
sft_eval_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$sft_job" \
  --export="$common,M6_PILOT_AUTHORIZATION=$pilot_authorization,M6_ROLLOUT_MODE=evaluation,M6_ROLLOUT_ROLE=mini_dev,M6_ROLLOUT_K=4,M6_ROLLOUT_OUTPUT=$sft_eval,M6_ROLLOUT_ADAPTER=$sft/final_adapter,M6_ROLLOUT_SEED=20260812,M6_ROLLOUT_ITERATION=0,M6_EVAL_IDENTITY=mini_sft" \
  scripts/run_m6_rollout_job.sh)"
submitted_jobs+=("$sft_eval_job")
gate_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$raw_eval_job:$sft_eval_job" \
  --export="ALL,M6_EXPECTED_GIT_SHA=$M6_EXPECTED_GIT_SHA,M6_RAW_EVAL_REPORT=$raw_eval/identity_report.json,M6_SFT_EVAL_REPORT=$sft_eval/identity_report.json,M6_CORPUS_AUDIT=$corpus/corpus_audit.json,M6_PILOT_AUTHORIZATION=$pilot_authorization,M6_SFT_GATE_OUTPUT=$sft_gate" \
  scripts/run_m6_sft_gate_job.sh)"
submitted_jobs+=("$gate_job")
grpo_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$gate_job" \
  --export="$common,M6_METHOD=multi_turn_grpo,M6_PILOT_AUTHORIZATION=$pilot_authorization,M6_RL_OUTPUT=$rl_root/multi_turn_grpo,M6_SFT_ADAPTER=$sft/final_adapter,M6_CURRICULUM=$curriculum,M6_SFT_GATE=$sft_gate,M6_RAW_EVAL_REPORT=$raw_eval/identity_report.json,M6_SFT_EVAL_REPORT=$sft_eval/identity_report.json,M6_CORPUS_AUDIT=$corpus/corpus_audit.json" \
  scripts/run_m6_mini_rl_loop_job.sh)"
submitted_jobs+=("$grpo_job")
anchor_job="$($slurm_bin/sbatch --parsable --dependency="afterok:$gate_job" \
  --export="$common,M6_METHOD=anchor_gigpo,M6_PILOT_AUTHORIZATION=$pilot_authorization,M6_RL_OUTPUT=$rl_root/anchor_gigpo,M6_SFT_ADAPTER=$sft/final_adapter,M6_CURRICULUM=$curriculum,M6_SFT_GATE=$sft_gate,M6_RAW_EVAL_REPORT=$raw_eval/identity_report.json,M6_SFT_EVAL_REPORT=$sft_eval/identity_report.json,M6_CORPUS_AUDIT=$corpus/corpus_audit.json" \
  scripts/run_m6_mini_rl_loop_job.sh)"
submitted_jobs+=("$anchor_job")

"$python_bin" - "$submission" "$M6_EXPECTED_GIT_SHA" "$base_url" \
  "$corpus_job" "$sft_job" "$raw_eval_job" "$sft_eval_job" "$gate_job" \
  "$grpo_job" "$anchor_job" <<'PY'
import sys
from pathlib import Path
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json

keys = ("corpus", "sft", "raw_eval", "sft_eval", "sft_gate", "multi_turn_grpo", "anchor_gigpo")
jobs = dict(zip(keys, sys.argv[4:]))
report = {
    "schema_version": "m6_mini_pilot_submission_v1",
    "study_id": "m6_monotonic_posttraining_v1",
    "development_only": True,
    "formal_training_allowed": False,
    "consumer_git_sha": sys.argv[2],
    "service_base_url": sys.argv[3],
    "jobs": jobs,
    "dependencies": {
        "sft": [jobs["corpus"]],
        "raw_eval": [jobs["corpus"]],
        "sft_eval": [jobs["sft"]],
        "sft_gate": [jobs["raw_eval"], jobs["sft_eval"]],
        "multi_turn_grpo": [jobs["sft_gate"]],
        "anchor_gigpo": [jobs["sft_gate"]],
    },
    "resources": {
        "corpus": {"gpu": 0, "cpu": 8, "memory_gib": 24, "hours": 24},
        "sft": {"gpu": 1, "cpu": 4, "memory_gib": 24, "hours": 24},
        "evaluation": {"gpu": 1, "cpu": 4, "memory_gib": 24, "hours": 24},
        "sft_gate": {"gpu": 0, "cpu": 2, "memory_gib": 8, "hours": 24},
        "rl_each": {"gpu": 1, "cpu": 4, "memory_gib": 24, "hours": 24},
    },
}
report["content_sha256"] = sha256_json(report)
atomic_write_json(Path(sys.argv[1]), report)
PY
trap - ERR INT TERM

printf 'corpus=%s\nsft=%s\nraw_eval=%s\nsft_eval=%s\nsft_gate=%s\nmulti_turn_grpo=%s\nanchor_gigpo=%s\n' \
  "$corpus_job" "$sft_job" "$raw_eval_job" "$sft_eval_job" "$gate_job" \
  "$grpo_job" "$anchor_job"
