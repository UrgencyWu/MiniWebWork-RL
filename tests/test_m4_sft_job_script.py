from pathlib import Path


def test_m4_sft_job_script_declares_one_seed_argument_and_fixed_lineage():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_m4_sft_job.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "usage: $0 STUDY_SEED" in script
    assert "outputs/m4_sft_corpus_v1" in script
    assert "outputs/m2_2r/seed_42/final_adapter" in script


def test_m4_rsft_job_script_declares_fixed_control_lineage():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_m4_rsft_job.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "usage: $0 STUDY_SEED" in script
    assert "outputs/m4_sft_corpus_v1" in script
    assert "outputs/m2_2r/seed_42/final_adapter" in script


def test_m4_online_job_script_limits_method_and_preserves_initial_adapter():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_m4_online_job.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "usage: $0 ALGORITHM STUDY_SEED" in script
    assert "rloo|grpo|gspo" in script
    assert "outputs/m2_2r/seed_42/final_adapter" in script
    assert 'outputs/m4_runs/${algorithm}/seed_${study_seed}' in script


def test_m4_final_eval_job_accepts_only_primary_matrix_methods():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_m4_final_eval_job.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "usage: $0 ALGORITHM STUDY_SEED" in script
    assert "sft|rsft|rloo|grpo|gspo" in script
    assert "--phase final_test" not in script  # The Python driver owns the frozen-split decision.
    assert "outputs/m4_final_eval/${algorithm}/seed_${study_seed}" in script
