# Experiment Governance

## 1. Dataset roles

| Dataset | Purpose | Gradient allowed? |
|---|---|---:|
| SFT train | expert and recovery supervision | yes |
| SFT valid | checkpoint selection | no |
| `rollout_dev_no_solution_v1` | rollout diagnostics and RL development | yes, after strict versioned collection |
| `rollout_dev_feasible_v2` | policy selection, false-no-solution and general-capability gate | no |
| legacy frozen test v1 | historical continuity | no |
| final test v2 | final one-time evaluation | no |

`rollout_dev_feasible_v2` is the only canonical feasible slice. It is generated deterministically from a frozen task specification and seed products/suppliers. Its Public, Oracle and dataset manifest must be byte-identical to builder output. It never enters a gradient batch.

## 2. Checkpoint and starting-policy selection

The canonical SFT checkpoint is selected by the lowest Canonical Valid Loss. This selects `seed_1234`. Frozen E2E results are not combined with Valid Loss.

The M3.0 starting policy is selected on rollout development data using this predeclared order:

1. false no-solution must not materially worsen on feasible v2;
2. feasible-task success must not materially regress;
3. after those gates, compare paired no-solution success;
4. when evidence remains unclear, retain the simpler earlier SFT policy.

A single aggregate success count cannot establish superiority or be dismissed as sampling variance.

## 3. Experiment identity

Comparable artifacts must share:

- Prompt contract and source hash;
- chat template and history window;
- Observation serializer and action parser;
- base model and adapter identity;
- task source and split;
- maximum model/environment turns;
- `temperature`, `top_p`, `top_k`, `K` and master seed.

Each rollout seed is deterministically derived from:

```text
master_seed + task_id + rollout_index
```

A/B artifacts are paired by `(task_id, rollout_index)`.

## 4. Metrics

- Task success: verifier-success trajectories / valid policy trajectories.
- Strict JSON: complete raw output is one JSON object.
- Schema Valid: schema-valid model turns / valid-rollout model turns.
- Environment Action Success: successful environment actions / attempted environment actions.
- Infrastructure-invalid trajectories are excluded from policy denominators and reported separately.
- Feasible evaluation reports success and false no-solution.
- No-solution evaluation reports success, missed no-solution, empty-result reach, submission persistence and premature finish.

## 5. Reward and failure governance

```text
success = 1
valid policy failure = 0
infrastructure failure = null
```

Policy failures include malformed actions, wrong targets, premature finish, loops, wrong decisions and verifier rejection. Infrastructure failures include model/device, browser/service/database, task-source, artifact and probability-evidence errors. Infrastructure failures never enter advantages or gradients.

The first baseline does not reward JSON validity, clicks, transitions, no-solution declarations, form completion or shorter trajectories.

## 6. Strict update compatibility

The first optimizer-compatible behavior distribution is:

```text
temperature = 1.0
top_p = 1.0
top_k = 0
```

A group is update-compatible only when:

- all Prompt/model/task/distribution identities match;
- every generated turn has aligned Prompt and completion token IDs;
- raw-policy and sampling-distribution log-probabilities are complete and finite;
- their maximum absolute difference is within the declared tolerance.

The replay layer recomputes this condition. A caller cannot promote a diagnostic artifact with a boolean flag.

A group is GRPO-valid only when it is update-compatible and has at least two valid trajectories with non-zero reward variance.

## 7. Artifact requirements

Formal artifacts contain:

- schema version and `complete` flag;
- Git, model, adapter, Prompt and task-source identities;
- full sampling identity and seeds;
- requested, valid and infrastructure-invalid rollout counts;
- per-turn Prompt/completion token IDs;
- raw-policy and sampling-distribution log-probabilities;
- failure origin, termination reason and Verifier result;
- aggregate and per-task metrics.

Partial artifacts use `complete=false` and cannot be used for optimization.

## 8. Final-test governance

`legacy_frozen_test_v1` is historical only. `final_test_v2` is evaluated after the starting policy, sampling parameters, optimizer parameters, stopping rule and final checkpoint rule are frozen. Final-test results cannot alter training decisions.
