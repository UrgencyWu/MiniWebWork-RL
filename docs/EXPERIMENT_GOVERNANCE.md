# Experiment Governance

## 1. Purpose

This document prevents prompt drift, task leakage, test-set model selection, metric ambiguity, incomplete sampling identity, and infrastructure failures being mislabelled as policy failures.

## 2. Dataset Roles

| Dataset | Purpose | May update model? | May tune sampling/checkpoint? |
|---|---|---:|---:|
| SFT train | expert and recovery supervision | yes | no |
| SFT valid | loss/action quality and checkpoint selection | no | yes |
| no-solution rollout dev | rollout diagnostics and RL development | yes, after versioned collection | yes |
| feasible rollout dev | false-no-solution and general-capability checks | yes, after versioned collection | yes |
| legacy frozen test v1 | historical continuity only | no | no |
| final test v2 | final Base/SFT/RL evaluation | no | no |

The historical 15-task set has been observed repeatedly during debugging. It remains useful for continuity, but it is not a pristine final test.

## 3. Checkpoint and Starting-policy Selection

Checkpoint selection is frozen before reading Frozen Test results.

Canonical SFT checkpoint rule:

```text
primary SFT checkpoint = lowest Canonical Valid Loss
```

This selects `seed_1234`. Frozen E2E success must not be combined with Valid Loss into a weighted checkpoint score.

The M3.0 starting policy is chosen between versioned SFT candidates only on rollout development data using a rule declared before comparison:

1. false no-solution must not materially worsen;
2. feasible-task success must not materially regress;
3. after those gates, compare paired no-solution success;
4. when the difference remains unclear, keep the simpler earlier SFT policy.

A single aggregate success count cannot establish superiority or be dismissed as sampling variance.

## 4. Prompt and Observation Governance

A result is comparable only when the following are identical:

- Prompt contract version;
- prompt-builder source hash;
- chat template;
- history window;
- Observation serializer and truncation rules;
- action schema and parser;
- maximum model/environment turns;
- task source and verifier version;
- full sampling identity.

A changed prompt contract creates a new experiment family. Base and trained policies must both be rerun.

## 5. Sampling Identity and Randomness

Each rollout seed is deterministically derived from:

```text
master_seed + task_id + rollout_index
```

Every formal rollout artifact stores:

```text
temperature
top_p
top_k
master seed
per-rollout derived seed
```

A/B policies use the same master seed, task order, `K`, temperature, top-p, and top-k. Artifacts with different identities are not paired.

Historical readiness artifacts that did not explicitly record top-k may support infrastructure and capability conclusions, but they are not optimizer inputs.

Training runs report at least:

- seed;
- base model and adapter hashes;
- train/valid manifest hashes;
- optimizer and learning rate;
- epochs/steps;
- final and best checkpoint criteria;
- source rollout artifact hashes and policy version.

## 6. Metric Definitions

### Task success

```text
verifier-success trajectories / valid policy trajectories
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

Computed only from valid policy rollouts in one:

```text
(task, policy, temperature, top_p, top_k)
```

group. Infrastructure records with `reward=null` are excluded.

### Learning-signal group

A group has learning signal when at least two valid policy trajectories exist and valid rewards have non-zero variance.

### Update-compatible group

The first implementation permits update compatibility only for:

```text
temperature = 1.0
top_p = 1.0
top_k = 0
```

and only when:

- prompt/model/task/distribution identities are consistent;
- every generated turn has aligned prompt/completion tokens;
- raw-policy and sampling-distribution log-probabilities are complete and finite;
- their maximum absolute difference is within the predeclared artifact tolerance.

A caller cannot override these conditions by setting a boolean flag. Future scaled or truncated behavior distributions require a separate versioned probability contract.

### GRPO-valid group

A group is valid for update only when it has both:

```text
has_learning_signal = true
update_distribution_compatible = true
```

## 7. Paired Policy Comparison

A/B artifacts are paired by:

```text
(task_id, rollout_index)
```

Formal comparison reports:

- both-success, A-only, B-only, both-fail counts;
- comparable and infrastructure-invalid pairs;
- per-task success counts and differences;
- exact McNemar result;
- task-level bootstrap interval;
- termination-reason breakdown;
- results across multiple master seeds.

Infrastructure-invalid pairs are excluded from policy success comparison and reported separately.

## 8. Failure Taxonomy

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
- malformed artifact or missing/non-finite log-prob evidence;
- cleanup/lifecycle failure that invalidates the rollout.

These receive `reward=null` and never enter an RL update.

## 9. No-Solution Evaluation

Report separately:

- empty-result reached rate;
- no-solution action selected rate;
- submission persisted rate;
- verifier success rate;
- false no-solution count;
- missed no-solution count;
- premature finish count.

An increase in no-solution success is not acceptable if false no-solution on feasible tasks also increases materially.

## 10. Reward Governance

The first Agentic RL experiment uses only deterministic terminal outcome reward:

```text
success = 1
policy failure = 0
infrastructure failure = null
```

Do not initially reward legal clicks, no-solution declarations, page transitions, or form completion. Such shaping can optimize proxy behavior rather than task completion. Process rewards are allowed only as a documented ablation after the outcome-only baseline.

## 11. Artifact Requirements

Every formal result contains:

- schema version and `complete` flag;
- Git commit SHA;
- policy/adapter/base model identity and content hashes;
- task source and split hashes;
- prompt and tokenizer identity;
- temperature, top-p, top-k, K, and seeds;
- requested and valid rollout counts;
- per-turn prompt/completion tokens;
- raw-policy and sampling-distribution log-probabilities;
- strict-distribution probability-match diagnostics;
- failure origin and termination reason;
- Verifier output;
- aggregate and per-task metrics.

Partial/incremental artifacts are marked `complete=false` and cannot be presented as final results or used for optimization.

## 12. Test-set Governance

`legacy_frozen_test_v1` is used only for historical continuity. `final_test_v2` is evaluated once after:

- starting policy is selected;
- sampling and optimizer hyperparameters are frozen;
- training stopping rules are frozen;
- the final checkpoint-selection rule is frozen.

No final-test result may alter training, sampling, checkpoint, or stopping decisions.
