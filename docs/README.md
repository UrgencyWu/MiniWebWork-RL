# Documentation Index

## Authoritative current documents

1. [`M6_MONOTONIC_POSTTRAINING_PLAN.md`](M6_MONOTONIC_POSTTRAINING_PLAN.md) — current planning-only proposal for a genuine Raw → SFT → RL strict-success improvement, with learnable data, closed-loop promotion gates, verifier-TD credit, and a new untouched holdout.
2. [`M5_FINAL_TECHNICAL_REPORT.md`](M5_FINAL_TECHNICAL_REPORT.md) — final M5 training report: 16,000 frozen trajectories, statistical results, SFT negative-transfer diagnosis, credit-assignment findings, failures, costs, artifacts, and limitations.
3. [`M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md`](M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md) — frozen historical M5 study contract.
4. [`M5_EXECUTION_READINESS.md`](M5_EXECUTION_READINESS.md) — M5 training, recovery, frozen-evaluation, and final-analysis execution record.
5. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — current implementation and historical stage evidence.
6. [`ARCHITECTURE_AND_CONTRACTS.md`](ARCHITECTURE_AND_CONTRACTS.md) — runtime boundaries and data contracts.
7. [`EXPERIMENT_GOVERNANCE.md`](EXPERIMENT_GOVERNANCE.md) — split, metric, seed, failure, and artifact governance.
8. [`M3_0_AGENTIC_RL_PLAN.md`](M3_0_AGENTIC_RL_PLAN.md) — historical foundation for the multi-turn GRPO implementation.
9. [`PROJECT_JOURNAL.md`](PROJECT_JOURNAL.md) — chronological decisions and lessons.
10. [`../reports/M3_0_DELIVERY_REPORT.md`](../reports/M3_0_DELIVERY_REPORT.md) — formal-update provenance and paired frozen-regression result.
11. [`INTERNSHIP_PROJECT_SUMMARY.md`](INTERNSHIP_PROJECT_SUMMARY.md) — mentor/recruiter-facing project summary and demo path.

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
scripts/run_quality_checks.sh
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
```

Old M2.3 temperature-sweep and comparison runners were deleted. Git history retains them only as superseded evidence.
