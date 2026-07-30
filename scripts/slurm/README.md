# Slurm Entry Points

## Current formal entrypoints

```text
m2_3_mini_single_probe.sbatch
```

This is the only formal M2.3/M3.0 rollout job. It runs one policy and one temperature per job and emits schema-v3 rollout evidence.

## Supported historical continuity jobs

```text
m1_1_procurement_e2e.sbatch
m1_2_rule_baseline.sbatch
m2_0_base_agent_eval.sbatch
```

These reproduce earlier environment/rule/Base baselines under current safety contracts. They do not participate in checkpoint selection or GRPO training.

## Stage-construction jobs

Other `m1_*`, `m2_1_*`, `m2_2*`, and `m2_3_mini_*` jobs preserve stage-specific construction, data, audit, or training workflows. Before reuse, verify that they explicitly declare:

- task source and split;
- Prompt Contract v2 where a model prompt is involved;
- adapter/base model path;
- output manifest/hash;
- Slurm-owned GPU visibility;
- scoped cleanup by owned PID/thread only.

The obsolete browser-agent-v1 SFT builder and legacy M2.3 probe/comparison jobs were deleted.

## Shared-node rules

Never use broad process cleanup such as:

```bash
pkill -f chromium
pkill -f uvicorn
```

Each job must terminate only the exact child processes it created. The canonical Environment owns its Uvicorn process and Playwright worker lifecycle.

Do not set `CUDA_VISIBLE_DEVICES` from Python after importing or initializing PyTorch. Slurm exposes the allocated logical device; code uses `cuda:0` within that allocation.
