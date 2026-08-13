#!/usr/bin/env bash
# Rebuild compatible baseline reports, then compare both completed M6 RL
# policies on the same frozen 200-task mini-dev K4 roster.
#SBATCH --job-name=m6-rl-eval-finalize
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m6_rl_eval_finalize_%j.out
#SBATCH --error=logs/m6_rl_eval_finalize_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_GRPO_EVAL_ROOT:?set M6_GRPO_EVAL_ROOT}"
: "${M6_ANCHOR_EVAL_ROOT:?set M6_ANCHOR_EVAL_ROOT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
study_root="$repo_root/outputs/m6_monotonic_posttraining_v1"
mini_root="$study_root/mini"
split_lock="$study_root/locks/m6_webshop_split_v1.json"
analysis_root="$mini_root/pilot_eval_v7/analysis"
mkdir -p "$analysis_root"

rebuild_identity() {
  identity="$1"
  source_root="$2"
  output="$3"
  "$python_bin" scripts/m6_closed_loop_eval.py \
    --identity "$identity" \
    --groups-dir "$source_root/groups" \
    --collection-report "$source_root/collection_report.json" \
    --source-identity-report "$source_root/identity_report.json" \
    --split-lock "$split_lock" \
    --output "$output"
}

# The old baseline reports predate evaluation-semantics v2. Re-summarization
# is CPU-only and preserves the exact already-frozen trajectories.
rebuild_identity raw "$mini_root/pilot_raw_eval" "$analysis_root/raw_identity_report.json"
rebuild_identity mini_sft "$mini_root/pilot_sft_eval_v2" "$analysis_root/sft_identity_report.json"

for required in \
  "$M6_GRPO_EVAL_ROOT/identity_report.json" \
  "$M6_ANCHOR_EVAL_ROOT/identity_report.json" \
  "$mini_root/pilot_rl_v7/multi_turn_grpo/rl_audit.json" \
  "$mini_root/pilot_rl_v7/anchor_gigpo/rl_audit.json" \
  "$mini_root/corpus_v2/corpus_audit.json"; do
  test -f "$required"
done

"$python_bin" scripts/m6_run_mini_chain.py \
  --raw-eval "$analysis_root/raw_identity_report.json" \
  --sft-eval "$analysis_root/sft_identity_report.json" \
  --rl-eval "$M6_GRPO_EVAL_ROOT/identity_report.json" \
  --corpus-audit "$mini_root/corpus_v2/corpus_audit.json" \
  --rl-audit "$mini_root/pilot_rl_v7/multi_turn_grpo/rl_audit.json" \
  --bootstrap-samples 20000 --allow-nonpassing \
  --output "$analysis_root/multi_turn_grpo_chain_report.json"

"$python_bin" scripts/m6_run_mini_chain.py \
  --raw-eval "$analysis_root/raw_identity_report.json" \
  --sft-eval "$analysis_root/sft_identity_report.json" \
  --rl-eval "$M6_ANCHOR_EVAL_ROOT/identity_report.json" \
  --corpus-audit "$mini_root/corpus_v2/corpus_audit.json" \
  --rl-audit "$mini_root/pilot_rl_v7/anchor_gigpo/rl_audit.json" \
  --bootstrap-samples 20000 --allow-nonpassing \
  --output "$analysis_root/anchor_gigpo_chain_report.json"

"$python_bin" - "$analysis_root" <<'PY'
import json
import sys
from pathlib import Path
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json

root = Path(sys.argv[1])
raw = json.loads((root / "raw_identity_report.json").read_text(encoding="utf-8"))
sft = json.loads((root / "sft_identity_report.json").read_text(encoding="utf-8"))
grpo = json.loads((root / "multi_turn_grpo_chain_report.json").read_text(encoding="utf-8"))
anchor = json.loads((root / "anchor_gigpo_chain_report.json").read_text(encoding="utf-8"))
summary = {
    "schema_version": "m6_mini_two_method_comparison_v1",
    "development_only": True,
    "formal_training": False,
    "raw_strict_success_rate": raw["strict_success_rate"],
    "sft_strict_success_rate": sft["strict_success_rate"],
    "methods": {
        "multi_turn_grpo": {
            "passed": grpo["passed"],
            "decision": grpo["decision"],
            "strict_success_rate": grpo["mini_gate"]["metrics"]["rl"]["strict_success_rate"],
            "rl_minus_sft_pp": grpo["mini_gate"]["deltas_pp"]["rl_minus_sft"],
            "bootstrap_positive_fraction_rl_minus_sft": grpo["mini_gate"]["bootstrap_positive_fraction"]["rl_minus_sft"],
            "mini_gate": grpo["mini_gate"],
        },
        "anchor_gigpo": {
            "passed": anchor["passed"],
            "decision": anchor["decision"],
            "strict_success_rate": anchor["mini_gate"]["metrics"]["rl"]["strict_success_rate"],
            "rl_minus_sft_pp": anchor["mini_gate"]["deltas_pp"]["rl_minus_sft"],
            "bootstrap_positive_fraction_rl_minus_sft": anchor["mini_gate"]["bootstrap_positive_fraction"]["rl_minus_sft"],
            "mini_gate": anchor["mini_gate"],
        },
    },
}
summary["content_sha256"] = sha256_json(summary)
atomic_write_json(root / "comparison_report.json", summary)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
