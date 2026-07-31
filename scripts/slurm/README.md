# Slurm Entry Points

## Current formal entrypoint

```text
m2_3_mini_single_probe.sbatch
```

This is the only formal M2.3/M3.0 rollout job. It runs one policy under one explicit sampling distribution and emits schema-v3.2 rollout evidence.

Arguments:

```text
POLICY TEMPERATURE MASTER_SEED K [MAX_TASKS] [TOP_P]
```

Diagnostic readiness example:

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 8 "" 0.9
```

Strict first-update collection example:

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 1.0 20260731 8 "" 1.0
```

The diagnostic distribution may produce `has_learning_signal=true` but remains `valid_for_grpo_update=false`. The first strict distribution uses `temperature=1.0, top_p=1.0`; a group becomes update-valid only when it also contains mixed rewards and complete rollout evidence.

## Paired policy analysis

Use:

```bash
python scripts/analyze_probe_ab.py \
  --a outputs/m2_3_mini/<A_ARTIFACT>.json \
  --b outputs/m2_3_mini/<B_ARTIFACT>.json \
  --output outputs/m2_3_mini/paired_ab.json
```

Artifacts must have the same task source, split, temperature, top-p, K, master seed, and prompt contract. The analysis pairs `(task_id, rollout_index)` and excludes infrastructure-invalid pairs.

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
