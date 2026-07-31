# MiniWebWork-RL Reproducibility Runbook

This runbook is the handoff path for a new contributor.  It deliberately
separates development, gradient-bearing rollout data, and frozen evaluation
data so a convenient command cannot silently invalidate an experiment.

## 1. Environment and CPU gate

Use Python 3.11 and the project Conda environment:

```bash
conda env create -f environment.yml
conda activate miniwebwork
pip install -e ".[test,training]"
python -m playwright install chromium

# Fast, non-GPU contract gate used before every experiment submission.
bash scripts/run_quality_checks.sh -m "not gpu and not slurm"
```

The gate compiles the source, initializes and validates the deterministic
SQLite seed, validates the task contracts, and runs the CPU test suite.  Do
not treat a passing GPU job as a substitute for this gate.

## 2. Data roles and immutability

| Source | Role | May enter a gradient batch? |
|---|---|---:|
| `data/tasks/rollout_dev_no_solution_v1` | RL development collection | Yes, only after strict collection passes |
| `data/tasks/rollout_dev_feasible_v2` | policy-selection and regression gate | No |
| `final_test_v2` | final one-time evaluation | No |

Before a run, validate every task source consumed by that process.  Rebuild
the feasible gate into a disposable directory and byte-compare it with the
frozen repository copy; never regenerate it in place.

```bash
python scripts/build_rollout_dev_feasible_v2.py --output-dir /tmp/miniwebwork-feasible-v2
cmp /tmp/miniwebwork-feasible-v2/valid_public.jsonl \
    data/tasks/rollout_dev_feasible_v2/valid_public.jsonl
cmp /tmp/miniwebwork-feasible-v2/valid_oracle.jsonl \
    data/tasks/rollout_dev_feasible_v2/valid_oracle.jsonl
cmp /tmp/miniwebwork-feasible-v2/dataset_manifest.json \
    data/tasks/rollout_dev_feasible_v2/dataset_manifest.json
```

## 3. Pre-update evidence

Start from the selected M2.2R adapter and preserve the paired A/B report used
to make that selection.  Diagnostic distributions such as `T=0.2, top_p=0.9`
are informative but never feed an update.

The only initial update-compatible distribution is:

```text
temperature = 1.0
top_p = 1.0
top_k = 0
use_cache = false
```

`use_cache=false` is intentional: the optimizer replays each prompt and
completion with no-cache teacher forcing, and the strict collector must use
the same numerical path.  The collector fails fast if the strict settings are
not exactly declared.  It also requests both raw generation logits and
post-processor scores, rejecting any hidden behavior-changing processor.

Run the log-probability audit on a previous strict artifact before changing
the collector or Transformers version:

```bash
sbatch scripts/slurm/m3_0b_logprob_audit.sbatch \
  outputs/m2_3_mini/runs/<STRICT_RUN>/<ARTIFACT>.json A 16
```

The report must show:

- no behavior difference between `sampling_logprobs` and generation raw
  logits;
- a raw/sampling difference at or below the artifact tolerance after the
  configured strict path; and
- a recorded model generation configuration and source prompt identity.

## 4. Strict collection and optimizer smoke

First submit a one-task smoke to confirm the environment, GPU and probability
contracts.  The argument order is `POLICY TEMPERATURE SEED K MAX_TASKS TOP_P
TOP_K TASK_SOURCE`.

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  A 1.0 20260731 8 1 1.0 0 no_solution
```

Inspect the final JSON, not only the Slurm exit code.  A group can enter the
optimizer only when all of these hold:

```text
complete = true
infrastructure_errors = 0
parameter_distribution_compatible = true
max_raw_sampling_logprob_abs_diff <= strict_logprob_match_tolerance
groups_valid_for_grpo >= 1
```

Then run exactly one LoRA-only optimizer smoke.  The batch script takes the
artifact and policy name; it binds the adapter path itself and writes a new,
disposable checkpoint.

```bash
sbatch scripts/slurm/m3_0_single_batch_smoke.sbatch \
  outputs/m2_3_mini/runs/<STRICT_RUN>/<ARTIFACT>.json A
```

The resulting `single_batch_smoke_report.json` must prove old/current
pre-update agreement, finite non-zero LoRA gradients, a non-zero parameter
delta, adapter hash, saved checkpoint, reload and finite forward pass.

## 5. Experiment ledger

For each submitted job, retain the artifact and record:

- Git SHA and clean/untracked source inventory;
- base model and adapter SHA-256;
- prompt-builder and chat-template SHA-256;
- task source SHA-256, split, task count and role;
- temperature, top-p, top-k, cache setting, `K`, seed and Slurm job ID;
- valid, infrastructure-invalid and successful trajectory counts;
- maximum raw/sampling log-probability difference; and
- the exact command, report path and output checkpoint hash.

Never train on an incomplete artifact, an artifact with `reward=null`, or an
artifact whose identities do not match the selected policy.

## 6. Stop and diagnose

Stop the run rather than relaxing a gate when any of the following occurs:

- raw/sampling log-probability mismatch exceeds the recorded tolerance;
- a hidden generation processor changes behavior probabilities;
- an infrastructure failure enters a reward or advantage;
- a group has no reward variance;
- old/current log-probabilities disagree before the first update;
- action format collapses or false no-solution grows on the feasible gate; or
- final/frozen evaluation data influences a training decision.

The expected response is a new diagnostic artifact and an explicit root-cause
report, not a tolerance increase or selective result deletion.
