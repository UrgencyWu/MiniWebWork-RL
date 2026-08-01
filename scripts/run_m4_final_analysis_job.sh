#!/usr/bin/env bash
# Build the complete frozen M4 5x3 analysis from exactly one final artifact per cell.
set -euo pipefail

repo_root="/home/wushaohua/data/MiniWebWork-RL"
cd "$repo_root"
source scripts/m4_assert_frozen_git.sh

single_artifact() {
  local collector_dir="$1"
  local files=()
  mapfile -t files < <(find "$collector_dir" -maxdepth 1 -type f -name 'single_probe_*.json' -print | sort)
  if [[ ${#files[@]} -ne 1 ]]; then
    echo "expected exactly one final collector artifact in $collector_dir; got ${#files[@]}" >&2
    exit 1
  fi
  printf '%s' "${files[0]}"
}

exec /home/wushaohua/miniconda3/envs/miniwebwork/bin/python scripts/m4_analyze_final.py \
  --input "sft:20260801:$(single_artifact outputs/m4_final_eval/sft/seed_20260801/collector)" \
  --input "sft:20260802:$(single_artifact outputs/m4_final_eval/sft/seed_20260802/collector)" \
  --input "sft:20260803:$(single_artifact outputs/m4_final_eval/sft/seed_20260803/collector)" \
  --input "rsft:20260801:$(single_artifact outputs/m4_final_eval/rsft/seed_20260801/collector)" \
  --input "rsft:20260802:$(single_artifact outputs/m4_final_eval/rsft/seed_20260802/collector)" \
  --input "rsft:20260803:$(single_artifact outputs/m4_final_eval/rsft/seed_20260803/collector)" \
  --input "rloo:20260801:$(single_artifact outputs/m4_final_eval/rloo/seed_20260801/collector)" \
  --input "rloo:20260802:$(single_artifact outputs/m4_final_eval/rloo/seed_20260802/collector)" \
  --input "rloo:20260803:$(single_artifact outputs/m4_final_eval/rloo/seed_20260803/collector)" \
  --input "grpo:20260801:$(single_artifact outputs/m4_final_eval/grpo/seed_20260801/collector)" \
  --input "grpo:20260802:$(single_artifact outputs/m4_final_eval/grpo/seed_20260802/collector)" \
  --input "grpo:20260803:$(single_artifact outputs/m4_final_eval/grpo/seed_20260803/collector)" \
  --input "gspo:20260801:$(single_artifact outputs/m4_final_eval/gspo/seed_20260801/collector)" \
  --input "gspo:20260802:$(single_artifact outputs/m4_final_eval/gspo/seed_20260802/collector)" \
  --input "gspo:20260803:$(single_artifact outputs/m4_final_eval/gspo/seed_20260803/collector)" \
  --training-root outputs/m4_runs \
  --output-dir outputs/m4_final_report
