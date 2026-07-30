# Experiment Governance

## 1. Purpose

This document prevents prompt drift, task leakage, test-set model selection, metric ambiguity, and infrastructure failures being mislabelled as policy failures.

## 2. Dataset Roles

| Dataset | Purpose | May update model? | May tune temperature/checkpoint? |
|---|---|---:|---:|
| SFT train | expert and recovery supervision | yes | no |
| SFT valid | loss/action quality and checkpoint selection | no | yes |
| rollout dev | stochastic rollout diagnostics and RL development | yes, after versioned collection | yes |
| legacy frozen test v1 | historical continuity only | no | no |
| final test v2 | final Base/SFT/RL evaluation | no | no |

The historical 15-task set has been observed repeatedly during debugging. It remains useful for continuity, but it is not a pristine final test.

## 3. Checkpoint Selection

Checkpoint selection is frozen before reading Frozen Test results.

Current rule:

```text
primary SFT checkpoint = lowest Canonical Valid Loss
```

This selects `seed_1234`. Frozen E2E success must not be combined with Valid Loss into a weighted checkpoint score.

A different policy may replace the primary checkpoint only when selected on a separately versioned development set with a written rule established before evaluation.

## 4. Prompt and Observation Governance

A result is comparable only when the following are identical:

- Prompt contract version;
- prompt-builder source hash;
- chat template;
- history window;
- Observation serializer and truncation rules;
- action schema and parser;
- maximum model/environment turns;
- generation settings;
- task source and verifier version.

A changed prompt contract creates a new experiment family. Base and trained policies must both be rerun.

## 5. Randomness

Each rollout seed is deterministically derived from:

```text
master_seed + task_id + rollout_index
```

A/B policies use the same master seed, task order, `K`, temperature, and top-p. The exact per-rollout seed is stored in the artifact.

Training runs report at least:

- seed;
- base model and adapter hashes;
- train/valid manifest hashes;
- optimizer and learning rate;
- epochs/steps;
- final and best checkpoint criteria.

## 6. Metric Definitions

### Task success

```text
number of verifier-success trajectories / valid policy trajectories
```

### Strict JSON rate

The complete raw output is one JSON object. Fenced or extracted JSON is not strict JSON.

### Schema Valid rate

```text
schema-valid model turns / all valid-rollout model turns
```

It is not the percentage of trajectories containing at least one valid action.

### Environment Action Success

```text
successful environment actions / environment actions attempted
```

Schema-invalid turns do not call `env.step` and are not in this denominator.

### Reward variance

Computed only from valid policy rollouts in one `(task, policy, temperature)` group. Infrastructure records with `reward=null` are excluded.

### GRPO-valid group

A group is valid for update only when:

- at least two valid policy trajectories exist;
- reward variance is greater than zero;
- prompt/model/task identities are consistent;
- every completion has aligned old-policy token log-probabilities.

## 7. Failure Taxonomy

### Policy failures

- non-JSON or Schema-invalid action;
- stale/wrong target selected by the model;
- premature `finish`;
- repeated actions or navigation loops;
- wrong product/no-solution decision;
- maximum turn truncation;
- valid submission rejected by the Verifier.

These receive reward 0.

### Infrastructure failures

- CUDA/OOM/device-side assertion;
- model or adapter load failure;
- Playwright/browser/service/database exception;
- task/Oracle source mismatch;
- malformed artifact or missing log-prob evidence;
- cleanup/lifecycle failure that invalidates the rollout.

These receive `reward=null` and never enter an RL update.

## 8. No-Solution Evaluation

Report separately:

- empty-result reached rate;
- no-solution action selected rate;
- submission persisted rate;
- verifier success rate;
- false no-solution count;
- missed no-solution count;
- premature finish count.

An increase in no-solution success is not acceptable if false no-solution on feasible tasks also increases materially.

## 9. Reward Governance

The first Agentic RL experiment uses only deterministic terminal outcome reward:

```text
success = 1
policy failure = 0
infrastructure failure = null
```

Do not initially reward legal clicks, no-solution declarations, page transitions, or form completion. Such shaping can optimize proxy behavior rather than task completion. Process rewards are allowed only as a documented ablation after the outcome-only baseline.

## 10. Artifact Requirements

Every formal result must contain:

- schema version;
- Git commit SHA;
- policy/adapter/base model identity and content hashes;
- task source and split hashes;
- prompt and tokenizer identity;
- generation parameters;
- requested and valid rollout counts;
- per-turn prompt/completion tokens and old log-probabilities;
- failure origin;
- verifier output;
- aggregate metrics.

Partial/incremental artifacts are marked `complete=false` and cannot be presented as final results.
