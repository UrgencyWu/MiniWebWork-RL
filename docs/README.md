# Documentation Index

## Authoritative current documents

1. [`M4_LONG_HORIZON_AGENT_RL_SCOPE.md`](M4_LONG_HORIZON_AGENT_RL_SCOPE.md) — approved interview-focused scope: shared SFT warm start, multi-turn GRPO baseline, step-aware credit assignment, GPU utilization and acceptance gates.
2. [`M4_FORMAL_TRAINING_READINESS.md`](M4_FORMAL_TRAINING_READINESS.md) — live, fail-closed checklist for every artifact required before any formal training job may be submitted.
3. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — current implementation and stage gates.
4. [`ARCHITECTURE_AND_CONTRACTS.md`](ARCHITECTURE_AND_CONTRACTS.md) — runtime boundaries and data contracts.
5. [`EXPERIMENT_GOVERNANCE.md`](EXPERIMENT_GOVERNANCE.md) — split, metric, seed, failure, and artifact governance.
6. [`M3_0_AGENTIC_RL_PLAN.md`](M3_0_AGENTIC_RL_PLAN.md) — historical foundation for the multi-turn GRPO implementation.
7. [`PROJECT_JOURNAL.md`](PROJECT_JOURNAL.md) — chronological decisions and lessons.
8. [`../reports/M3_0_DELIVERY_REPORT.md`](../reports/M3_0_DELIVERY_REPORT.md) — formal-update provenance and paired frozen-regression result.
9. [`INTERNSHIP_PROJECT_SUMMARY.md`](INTERNSHIP_PROJECT_SUMMARY.md) — mentor/recruiter-facing project summary and demo path.

`M4_RLVR_STUDY_PROTOCOL.md` is retained as the historical five-algorithm
preregistration. Its 15-run matrix was superseded on 2026-08-08 and must not be
resumed as the formal project matrix.

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
