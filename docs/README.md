# Documentation Index

## Authoritative current documents

1. [`M6_MINI_RESULT_AND_FAILURE_ANALYSIS.md`](M6_MINI_RESULT_AND_FAILURE_ANALYSIS.md) — completed M6-mini result: SFT exceeds Raw, both RL methods miss the SFT promotion gate, with paired statistics, failure trajectories, credit evidence and root-cause analysis.
2. [`TRAINING_FAILURE_LEDGER.md`](TRAINING_FAILURE_LEDGER.md) — append-only training/evaluation failure ledger with job, root cause, repair, successor and research disposition.
3. [`M6_MONOTONIC_POSTTRAINING_PLAN.md`](M6_MONOTONIC_POSTTRAINING_PLAN.md) — frozen Phase A/B contract for a genuine Raw → SFT → RL strict-success improvement.
4. [`M6_EXECUTION_RUNBOOK.md`](M6_EXECUTION_RUNBOOK.md) — development-only Phase A/B flow, resources, dependencies, recovery and hard stop gates; the completed mini result does not authorize formal training.
5. [`M5_FINAL_TECHNICAL_REPORT.md`](M5_FINAL_TECHNICAL_REPORT.md) — final M5 training report: 16,000 frozen trajectories, statistical results, SFT negative-transfer diagnosis, credit-assignment findings, failures, costs, artifacts, and limitations.
6. [`M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md`](M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md) — frozen historical M5 study contract.
7. [`M5_EXECUTION_READINESS.md`](M5_EXECUTION_READINESS.md) — M5 training, recovery, frozen-evaluation, and final-analysis execution record.
8. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — current implementation and historical stage evidence.
9. [`ARCHITECTURE_AND_CONTRACTS.md`](ARCHITECTURE_AND_CONTRACTS.md) — runtime boundaries and data contracts.
10. [`EXPERIMENT_GOVERNANCE.md`](EXPERIMENT_GOVERNANCE.md) — split, metric, seed, failure, and artifact governance.
11. [`M3_0_AGENTIC_RL_PLAN.md`](M3_0_AGENTIC_RL_PLAN.md) — historical foundation for the multi-turn GRPO implementation.
12. [`PROJECT_JOURNAL.md`](PROJECT_JOURNAL.md) — chronological decisions and lessons.
13. [`../reports/M3_0_DELIVERY_REPORT.md`](../reports/M3_0_DELIVERY_REPORT.md) — formal-update provenance and paired frozen-regression result.
14. [`INTERNSHIP_PROJECT_SUMMARY.md`](INTERNSHIP_PROJECT_SUMMARY.md) — mentor/recruiter-facing project summary and demo path.

The M4 scope/readiness documents and `M4_RLVR_STUDY_PROTOCOL.md` are retained as
historical evidence. M4's custom-data training results exposed a collapsed
signal regime and must not be resumed or mixed into the M5 formal matrix.

When historical reports conflict with these files, the authoritative current documents take precedence.

## Historical stage evidence

Files prefixed with `M1_` or `M2_` preserve the implementation state at the time they were written. They are useful for:

- reconstructing earlier decisions;
- reviewing commands and Slurm jobs;
- comparing stage metrics;
- understanding superseded implementations.

They must not be interpreted as the current architecture or execution entrypoint unless explicitly referenced by `CURRENT_STATUS.md`.

## Current executable entrypoints

```text
scripts/run_m6_phase_a_cpu_job.sh
scripts/run_m6_webshop_service_job.sh
scripts/run_m6_rollout_job.sh
scripts/run_m6_corpus_cpu_job.sh
scripts/run_m6_mini_sft_job.sh
scripts/run_m6_mini_rl_loop_job.sh
scripts/run_quality_checks.sh
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
```

Old M2.3 temperature-sweep and comparison runners were deleted. Git history retains them only as superseded evidence.
