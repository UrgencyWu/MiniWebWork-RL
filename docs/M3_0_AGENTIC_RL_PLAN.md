# M3.0 Agentic RL Plan

## 1. Goal

M3.0 converts the current SFT browser policy into an online policy-improvement system. It is deliberately scoped as a minimal, auditable Agentic RL implementation rather than a large distributed training platform.

The first scientific/engineering question is:

> Can outcome-only group-relative policy optimization improve multi-turn procurement success beyond the Canonical SFT policy without degrading action validity or inducing false no-solution behavior?

## 2. Current Entry Point

Initial policy:

```text
Qwen3.5-4B + M2.3-mini seed_1234 LoRA
```

Selection reason: lowest Canonical Valid Loss. The historical Frozen Test result is not used for selection.

Reference policy:

```text
frozen copy of the same M2.3-mini starting policy
```

Task source:

```text
versioned rollout_dev task set
```

Legacy Frozen Test v1 is excluded from optimization. A new final_test_v2 is opened only after training and hyperparameters are frozen.

## 3. Required Rollout Record

For every model turn:

```text
prompt_token_ids
completion_token_ids
old_policy_logprobs
prompt_hash
raw_output
parsed_action
environment_result
```

For every trajectory:

```text
task_id
policy_version
rollout_seed
turns
terminal_reward
termination_reason
failure_origin
verifier_result
```

For every group:

```text
same task
K stochastic trajectories
reward sequence
mean/std
valid_for_grpo_update
```

Infrastructure failures carry `reward=null` and are excluded before advantage calculation.

## 4. First Reward

```text
Verifier success = 1.0
Valid policy failure = 0.0
Infrastructure failure = null
```

No handcrafted step reward is used in the first experiment. In particular, do not reward:

- valid JSON;
- legal browser actions;
- reaching an empty page;
- choosing no-solution;
- submitting a form.

These are diagnostics, not the objective.

## 5. Group Sampling

Each task generates `K` trajectories from the same current policy. Initial development range:

```text
K = 4 or 8
temperature ∈ {0.2, 0.4}
top_p = 0.9
```

Temperature selection is performed only on rollout_dev. A group is useful only when valid trajectories contain non-zero reward variance.

Normalized group-relative advantage:

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + epsilon)
```

All action completions in one trajectory initially receive the trajectory advantage. This is coarse sequence-level credit assignment and must be described as such; it is not a process reward.

## 6. Policy Objective

For each action completion token `t` in trajectory `i`:

```text
ratio_it = exp(logπθ(token_it) - logπold(token_it))
```

Use a clipped token-level objective aggregated over completion tokens, with trajectory/group advantage applied to every action completion in that trajectory.

A frozen reference-policy KL term may be added with a small coefficient. It is a stabilization term, not the reward:

```text
loss = clipped_policy_loss + beta_kl * KL(current || reference)
```

First pilot configuration should keep the design minimal:

- LoRA parameters trainable;
- base model frozen;
- no value model;
- no GAE;
- no replay buffer;
- on-policy rollout batch only;
- gradient clipping;
- one optimizer update per collected batch;
- reference policy fixed during the pilot.

## 7. Stage Sequence

### M3.0B-0: Canonical rollout qualification

Pass conditions:

- infrastructure error rate near zero;
- prompt/completion/logprob alignment is complete;
- at least one no-solution success;
- at least one mixed-reward group;
- false no-solution does not materially increase;
- M2.3-mini does not materially regress general tasks.

### M3.0B-1: Single-batch gradient smoke

One rollout batch only:

1. collect groups;
2. filter invalid groups;
3. recompute current log-probabilities;
4. verify old/current equality before update;
5. calculate advantages and loss;
6. backpropagate through LoRA parameters;
7. verify finite gradients and non-zero trainable parameter updates;
8. save a disposable checkpoint;
9. reload and run a small evaluation.

No success claim is made at this stage.

### M3.0B-2: Small pilot

Suggested initial budget:

```text
8–16 tasks per update
K = 4
5–20 optimizer updates
1 primary training seed
```

Evaluate periodically on a fixed RL development slice. Stop on action-format collapse, false no-solution growth, or general-task degradation.

### M3.0C: Formal comparison

Run at least:

```text
Canonical Base
M2.2R SFT
M2.3-mini SFT
M3 RL checkpoint
```

Report task-type slices, action validity, environment action success, trajectory length, no-solution confusion, and multiple RL seeds when resources permit.

## 8. Acceptance Criteria

M3.0 may be considered complete only when:

- training and rollout artifacts are reproducible from hashes and seeds;
- policy updates use aligned old/current token log-probabilities;
- no infrastructure failure contributes gradient;
- RL improves a predeclared development metric over M2.3-mini;
- final_test_v2 is evaluated once after model/hyperparameter freeze;
- action validity and false no-solution remain within predeclared tolerances;
- at least two RL seeds are reported for any strong performance claim.

## 9. Stop Conditions

Immediately stop and diagnose when:

- all groups have zero reward variance;
- old/current log-probabilities differ before the first update;
- NaN/Inf appears in loss or gradients;
- Schema Valid collapses;
- the model learns to overproduce no-solution;
- browser/database failures enter reward data;
- the same Frozen Test is used to choose checkpoints or temperature.

## 10. Deferred Work

The following are not part of the first M3.0 implementation:

- process reward model;
- learned value function;
- GAE/PPO critic training;
- visual observations;
- vLLM distributed rollout;
- multi-node training;
- generic external websites;
- long-horizon curriculum beyond the procurement environment.

They can be added only after the outcome-only baseline is stable and measured.
