#!/usr/bin/env bash
# CPU-only paired Raw/SFT K4 promotion gate for the approved M6 mini pilot.
#SBATCH --job-name=m6-sft-gate
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/m6_sft_gate_%j.out
#SBATCH --error=logs/m6_sft_gate_%j.err

set -euo pipefail
repo_root="${M6_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL}"
cd "$repo_root"
mkdir -p logs
: "${M6_EXPECTED_GIT_SHA:?set M6_EXPECTED_GIT_SHA}"
: "${M6_RAW_EVAL_REPORT:?set M6_RAW_EVAL_REPORT}"
: "${M6_SFT_EVAL_REPORT:?set M6_SFT_EVAL_REPORT}"
: "${M6_CORPUS_AUDIT:?set M6_CORPUS_AUDIT}"
: "${M6_SFT_GATE_OUTPUT:?set M6_SFT_GATE_OUTPUT}"
test "$(git rev-parse HEAD)" = "$M6_EXPECTED_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
python_bin="/home/wushaohua/miniconda3/envs/miniwebwork/bin/python"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
pilot_args=()
: "${M6_PILOT_AUTHORIZATION:?set M6_PILOT_AUTHORIZATION}"
pilot_args=(--pilot-authorization "$M6_PILOT_AUTHORIZATION")
"$python_bin" - "$M6_RAW_EVAL_REPORT" "$M6_SFT_EVAL_REPORT" "$M6_PILOT_AUTHORIZATION" <<'PY'
import json
import sys
from pathlib import Path
from miniwebwork.long_horizon_rl.contracts import sha256_json
from miniwebwork.m6_pilot import validate_pilot_authorization

authorization = validate_pilot_authorization(json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")))
for report_name in sys.argv[1:3]:
    report = Path(report_name)
    identity = json.loads(report.read_text(encoding="utf-8"))
    binding = json.loads((report.parent / "pilot_binding.json").read_text(encoding="utf-8"))
    expected = dict(binding)
    observed = expected.pop("content_sha256", None)
    if observed != sha256_json(expected):
        raise ValueError("M6 pilot evaluation binding self-hash drift")
    if binding.get("identity_report_content_sha256") != identity.get("content_sha256"):
        raise ValueError("M6 pilot evaluation binding/report drift")
    if binding.get("pilot_authorization_content_sha256") != authorization["content_sha256"]:
        raise ValueError("M6 pilot evaluation authorization drift")
PY
"$python_bin" scripts/m6_gate_mini_sft.py \
  --raw-eval "$M6_RAW_EVAL_REPORT" \
  --sft-eval "$M6_SFT_EVAL_REPORT" \
  --corpus-audit "$M6_CORPUS_AUDIT" \
  "${pilot_args[@]}" \
  --output "$M6_SFT_GATE_OUTPUT"
