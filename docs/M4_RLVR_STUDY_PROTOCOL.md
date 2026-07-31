# M4 RLVR Algorithm Study Protocol

Status: preregistered implementation protocol.  It is intentionally committed
before the M4 development jobs and before any final-test evaluation.

## Question and scope

M4 asks which post-training method most reliably improves a multi-turn Browser
Agent when reward is only available after a deterministic terminal verifier.
The setting also contains variable-length action traces and strict JSON action
parsing.  The deliverable is an auditable comparison, not a claim that a
particular method must win.

The primary question is:

> Under the same base policy, adapter capacity, task worlds, collection budget
> and random seeds, how do SFT, RSFT, RLOO, GRPO and GSPO differ in verified
> Browser-Agent task success, failure composition and cost?

## Dataset and freeze boundary

The checked dataset is `data/tasks/m4_rlvr_v1` with its matching versioned
catalogue in `data/seed_m4_rlvr_v1`.

| Split | Worlds | Tasks | Role | May update model |
| --- | ---: | ---: | --- | --- |
| `train` | 60 | 240 | optimizer source | yes |
| `dev` | 18 | 72 | selection and regression only | no |
| `test` | 30 | 120 | final frozen evaluation | no |

Every world owns four products and yields exactly one task from each family:
`exact_product`, `cheapest_feasible`, `highest_rating_supplier`, and
`no_feasible_product`.  Train, dev and test worlds have disjoint product IDs,
constraint signatures, selected-product answers and public instructions.  The
builder emits hashes for every public/oracle file; `validate_m4_rlvr_dataset`
recomputes every oracle against the M4 seed and audits all cross-split sets.
Every world keyword is unique, while categories remain one of the four values
exposed by the actual browser filter UI; this prevents a database-only task
contract that an agent could not execute through the site.

An official M4 runner must call `assert_m4_split_purpose` before loading a
source.  `test` permits only `final_evaluation`, never data creation, model
selection, offline training or online updates.  The selected configuration and
the exact git commit must be written before the first test job is submitted.

## Common model and rollout contract

All methods use the same locally cached `Qwen3.5-4B` base model, the same LoRA
target modules and rank, prompt contract, max environment steps, browser image,
task/seed manifest hashes and model-tokenizer revision.  The fixed study seeds
are `20260801`, `20260802`, and `20260803`.

Online collection is strictly on-policy:

```text
temperature = 1.0
top_p       = 1.0
top_k       = 0
use_cache   = false during replay/update audit
```

For every generated action token, the artifact must retain prompt token IDs,
completion token IDs, raw behavior-policy log-probabilities and sampling
log-probabilities.  A group is eligible only if it has no infrastructure error,
has at least two valid trajectories, has mixed terminal rewards, and its maximum
raw/sampling log-probability difference is at most `0.05`.  Invalid trajectories
are counted in cost/failure reporting but never converted into reward-zero
training examples.

## Methods

| Method | Regime | Training signal | Ratio / loss |
| --- | --- | --- | --- |
| SFT | offline control | oracle-expert action turns from train only | completion-only NLL |
| RSFT | offline control | only verifier-successful train rollouts, deterministic best-of-N tie-break | completion-only NLL |
| RLOO | online | reward minus mean reward of the other K-1 trajectories | tokenwise clipped ratio |
| GRPO | online | group-normalized terminal reward | tokenwise clipped ratio |
| GSPO | online | group-normalized terminal reward | whole trajectory sequence ratio |

RSFT samples and verifies candidates only in the train split.  It records the
candidate count, successes, selected trajectory IDs and discarded failures; a
seed with no usable RSFT examples is reported as a protocol outcome rather than
silently substituted with oracle data.

For online methods `K=4` trajectories are collected for each same-task group.
RLOO uses `r_i - mean(r_-i)`.  GRPO uses the population-standard-deviation
normalization already audited in M3.  GSPO sums the real action-token log-ratio
over all turns in one trajectory before clipping; prompts and padding are never
included.  The implementation limits sequence log-ratios only at ±30 to prevent
floating-point overflow and reports any saturation.

DAPO-style dynamic sampling and decoupled clipping are optional, separately
named mechanism ablations.  M4 must not call such an ablation “DAPO” or compare
it as an additional primary algorithm.

## Budget and run matrix

The unit of fairness is *collected action tokens*, not optimizer steps.  A
training seed runs two ordered passes over all 240 train tasks, with K=4
attempts per task per pass: 1,920 attempted trajectories per online method.
The per-method cap is the same `250,000` collected action-token budget; stopping
at that cap is recorded and applied to every online method.  The deterministic
task order is a seed-specific permutation fixed before collection.

SFT and RSFT use the same two-pass train-world roster and a completion-token
budget no larger than 250,000.  Their model-forward tokens, optimizer steps,
wall time and peak GPU memory are still logged separately because offline NLL
and browser rollouts do not have identical hardware costs.

The required primary matrix contains 15 runs:

```text
{SFT, RSFT, RLOO, GRPO, GSPO} × {20260801, 20260802, 20260803}
```

Development results may decide only operational safety settings declared in the
run configuration (for example a clearly documented OOM-safe microbatch size).
They may not select an algorithm, seed, checkpoint or test-time sampling rule.

## Evaluation and statistical analysis

After configurations are frozen, every selected seed/checkpoint is evaluated on
all 120 final-test tasks with four independent rollout seeds per task.  The
primary metric is the macro mean of each task's four terminal verifier rewards.
This gives 120 task-level paired observations per checkpoint and avoids treating
four rollouts from one task as independent tasks.

The final report must include:

- mean, standard deviation across training seeds, and task-cluster bootstrap
  95% confidence intervals;
- paired task-cluster bootstrap confidence intervals and a two-sided paired
  permutation test for each prespecified method comparison;
- raw attempt success, per-task success distribution, valid/infrastructure
  rollout counts, and no-solution-specific accuracy;
- a mutually exclusive failure taxonomy: JSON/schema parse, rejected browser
  action, environment/browser infrastructure, premature/max-step termination,
  verifier constraint/objective failure, and false/expected no-solution;
- collected and update action tokens, model forward tokens where available,
  trajectory turns, wall-clock seconds, GPU hours, peak memory and number of
  skipped no-signal groups.

No significance claim is made from a favorable point estimate alone.  Neutral,
negative or underpowered results remain first-class outcomes.

## Required audit artifacts

Each run directory must contain immutable JSON for the resolved protocol,
algorithm ID/formula version, git commit, base/adapter hashes, seed, task and
seed manifests, environment image, collection/update accounting, failure
taxonomy and checkpoint lineage.  A consolidated CSV/JSON table and a short
interpretation report must link every result row to those artifacts.
