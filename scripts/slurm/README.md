# Slurm Entry Points

## Canonical rollout

```text
scripts/slurm/m2_3_mini_single_probe.sbatch
```

Arguments:

```text
POLICY TEMPERATURE MASTER_SEED K [MAX_TASKS] [TOP_P] [TOP_K] [TASK_SOURCE]
```

Task aliases:

```text
no_solution → data/tasks/rollout_dev_no_solution_v1
feasible    → data/tasks/rollout_dev_feasible_v2
```

Feasible v2 is the only canonical feasible slice. It is a policy-selection and regression gate and must not enter gradient updates.

Diagnostic collection:

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 0.2 20260731 8 "" 0.9 0 no_solution
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 8 "" 0.9 0 no_solution
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 0.2 20260731 8 "" 0.9 0 feasible
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 8 "" 0.9 0 feasible
```

Strict collection:

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 1.0 20260731 8 "" 1.0 0 no_solution
```

The first strict distribution is `temperature=1.0, top_p=1.0, top_k=0`. Update-valid groups require complete token evidence, matching raw/sampling log-probabilities, zero infrastructure errors and non-zero reward variance.

## Feasible v2 generation

```bash
python scripts/build_rollout_dev_feasible_v2.py --output-dir data/tasks/rollout_dev_feasible_v2
```

The builder must reproduce byte-identical `valid_public.jsonl`, `valid_oracle.jsonl` and `dataset_manifest.json`.

## Paired A/B analysis

```bash
python scripts/analyze_probe_ab.py \
  --a outputs/m2_3_mini/runs/<A_RUN>/<A_ARTIFACT>.json \
  --b outputs/m2_3_mini/runs/<B_RUN>/<B_ARTIFACT>.json \
  --output outputs/m2_3_mini/paired_ab.json
```

Artifacts must share task source, model/prompt identity, distribution, K, seed and turn limits. Analysis pairs `(task_id, rollout_index)` and excludes infrastructure-invalid pairs.

## Single-batch optimizer smoke

```bash
sbatch scripts/slurm/m3_0_single_batch_smoke.sbatch \
  <STRICT_ARTIFACT_JSON> \
  outputs/m2_3_mini/seed_1234/final_adapter
```

The smoke performs one LoRA-only update with `AdamW(weight_decay=0.0)`, then validates replay consistency, gradients, parameter change, checkpoint save, reload and finite forward.

Slurm owns GPU visibility. Each job must clean up only the processes and threads it created.
