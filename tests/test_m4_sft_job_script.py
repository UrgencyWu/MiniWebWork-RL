from pathlib import Path


def test_m4_sft_job_script_declares_one_seed_argument_and_fixed_lineage():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_m4_sft_job.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "usage: $0 STUDY_SEED" in script
    assert "outputs/m4_sft_corpus_v1" in script
    assert "outputs/m2_2r/seed_42/final_adapter" in script
