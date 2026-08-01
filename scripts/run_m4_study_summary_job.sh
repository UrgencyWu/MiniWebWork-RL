#!/usr/bin/env bash
# Render the version-controlled evidence-only M4 final summary after analysis succeeds.
set -euo pipefail

cd /home/wushaohua/data/MiniWebWork-RL
source scripts/m4_assert_frozen_git.sh
source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate miniwebwork

test -s outputs/m4_final_report/m4_final_report.json
python scripts/m4_render_study_summary.py \
  --input outputs/m4_final_report/m4_final_report.json \
  --output outputs/m4_final_report/m4_final_study_report.md
