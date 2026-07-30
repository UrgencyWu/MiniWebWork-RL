# Documentation Index

## Authoritative current documents

1. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — current implementation and stage gates.
2. [`ARCHITECTURE_AND_CONTRACTS.md`](ARCHITECTURE_AND_CONTRACTS.md) — runtime boundaries and data contracts.
3. [`EXPERIMENT_GOVERNANCE.md`](EXPERIMENT_GOVERNANCE.md) — split, metric, seed, failure, and artifact governance.
4. [`M3_0_AGENTIC_RL_PLAN.md`](M3_0_AGENTIC_RL_PLAN.md) — approved Agentic RL path.
5. [`PROJECT_JOURNAL.md`](PROJECT_JOURNAL.md) — chronological decisions and lessons.

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
