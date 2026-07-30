# M3.0 Multi-turn Agentic RL Plan

## 1. Objective

M3.0 converts the M2.3-mini SFT browser policy into a small, auditable online policy-improvement system.

Primary question:

> Can outcome-only group-relative policy optimization improve multi-turn procurement success over the Canonical SFT policy without degrading JSON/Action validity, general-task performance, or no-solution precision?

This is a **multi-turn GRPO-style** implementation. It must not be described as a stock one-shot GRPO training run.

## 2. Why a Custom Multi-turn Objective Is Required

The browser Agent does not generate one monolithic completion. At every browser turn it:

```text
current Observation + bounded history
→ rebuild Canonical Prompt v2
→ generate one JSON action
→ execute action
→ receive a new Observation
```

Therefore one trajectory contains multiple independently rendered prompts and multiple action completions. Treating the full trajectory as one completion would lose the true conditioning context and misalign token log-probabilities.

Approved implementation:

```text
one forward per Agent turn
→ one prompt/completion/logprob segment per turn
→ concatenate action-token segments within each trajectory
→ apply one terminal group-relative advantage to all action tokens
→ mean over tokens within each trajectory
→ mean equally over trajectories
```

The project may use Accelerate/TRL utilities for model loading, distributed execution, logging, or checkpointing. It must not force the environment into a one-shot `GRPOTrainer` completion interface unless a later version formally introduces an equivalent stateful environment/masking contract.

## 3. Initial and Reference Policies

Initial policy:

```text
Qwen3.5-4B + M2.3-mini seed_1234 LoRA
```

Selection rule: lowest Canonical Valid Loss. Historical Frozen Test performance is not used for checkpoint selection.

Reference policy:

```text
frozen copy of the same starting M2.3-mini policy
```

Development tasks:

```text
versioned rollout_dev task source
```

`legacy_frozen_test_v1` is excluded from optimization and hyperparameter selection. A new `final_test_v2` is opened only after the model and hyperparameters are frozen.

## 4. Rollout Evidence Contract

Every Agent turn stores:

```text
prompt_hash
prompt_token_ids
completion_token_ids
raw_policy_logprobs
sampling_distribution_logprobs
raw_output
strict_json/schema status
parsed_action
environment_action_result
```

Every trajectory stores:

```text
task_id / task_type
policy and adapter identity
rollout seed
ordered turn records
terminal reward
termination reason
failure origin
Verifier result
```

Every same-task group stores:

```text
K trajectories
valid trajectory count
infrastructure error count
reward sequence
reward mean/std
has_reward_variance
has_learning_signal
update_distribution_compatible
valid_for_grpo_update
```

Infrastructure failure:

```text
rollout_valid = false
failure_origin = infrastructure
reward = null
```

It is excluded before reward normalization and gradient construction.

## 5. Reward

First formal reward:

```text
Verifier success       = 1.0
Valid policy failure   = 0.0
Infrastructure failure = null
```

Do not initially reward:

- strict JSON;
- Schema-valid actions;
- successful clicks;
- reaching an empty-result page;
- choosing no-solution;
- form completion;
- shorter trajectories.

These remain diagnostics. Process or progress rewards require a later explicit ablation against the outcome-only baseline.

## 6. Diagnostic Sampling Versus Training Sampling

### 6.1 Readiness Probe

M2.3 diagnostic probes may use:

```text
K ∈ {4, 8}
temperature ∈ {0.2, 0.4}
top_p = 0.9
```

Their purpose is to test exploration, no-solution success, and reward variance. A mixed-reward diagnostic group has `has_learning_signal=true`.

It does **not** automatically have `valid_for_grpo_update=true`, because top-p truncation changes the actual behavior distribution.

### 6.2 First Formal Update Distribution

Preferred strict pilot:

```text
temperature = 1.0
top_p = 1.0
```

Then the sampled behavior distribution equals the raw model categorical policy and old/current raw policy log-probabilities are directly comparable.

Fallback, only when the unrestricted policy cannot produce usable groups:

```text
fixed temperature T ∈ [0.7, 1.0]
top_p = 1.0
```

In that case, both old and current log-probabilities must be computed under the same temperature-scaled categorical distribution.

`top_p < 1.0` is not permitted in the first optimizer batch unless the implementation explicitly recomputes and differentiates the same truncated behavior distribution. Raw model log-probabilities and post-top-p sampling log-probabilities must never be mixed in one importance ratio.

## 7. Group-relative Advantage

For valid policy rollouts of the same task:

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + epsilon)
```

Requirements:

- at least two valid trajectories;
- finite rewards;
- non-zero population standard deviation;
- no infrastructure records;
- one task and one policy/distribution per group.

Every action completion in trajectory `i` initially receives `A_i`. This is trajectory-level credit assignment broadcast across turns. It is not a process reward and does not identify which earlier action caused success or failure.

## 8. Policy Objective

For generated action token `t` in trajectory `i`:

```text
ratio_it = exp(log π_current(a_it | prompt_it)
               - log π_old(a_it | prompt_it))
```

Use a clipped objective:

```text
min(ratio_it * A_i,
    clip(ratio_it, 1-ε, 1+ε) * A_i)
```

Aggregation:

```text
mean over action tokens within trajectory
→ mean over trajectories
```

This prevents long trajectories from receiving more weight solely because they contain more generated action tokens.

Optional reference stabilization:

```text
loss = clipped_policy_loss
       + beta_kl * KL(current_policy || frozen_reference_policy)
```

Reference KL is a regularizer, not reward shaping.

The implementation in `src/miniwebwork/rl/objective.py` provides:

- zero-variance group rejection;
- trajectory-normalized clipped objective;
- optional non-negative reference KL estimator;
- finite-value and shape invariants;
- clip fraction, approximate KL, mean ratio, and token-count diagnostics.

## 9. Minimal Training Architecture

```text
Current LoRA Policy
    ↓ grouped multi-turn browser rollouts
Typed Rollout Artifact
    ↓ validity and distribution filters
Group-relative Advantages
    ↓ per-turn forward over prompt + action tokens
Trajectory-normalized Clipped Loss
    ↓
LoRA-only optimizer update
    ↓
Checkpoint + fixed development evaluation
```

First pilot configuration:

- base model frozen;
- LoRA parameters trainable;
- no value model;
- no GAE;
- no replay buffer;
- no off-policy reuse across policy versions;
- one collected batch followed by one bounded optimizer update;
- gradient clipping;
- frozen reference policy;
- explicit policy-version binding in artifacts.

## 10. Stage Sequence

### M3.0B-0 — Canonical Rollout Qualification

Pass conditions:

1. infrastructure error rate is near zero;
2. prompt/completion/raw/sampling log-prob evidence has 100% coverage on valid turns;
3. at least one no-solution trajectory succeeds;
4. at least one same-task group has mixed reward;
5. false no-solution does not materially increase;
6. M2.3-mini does not materially regress general tasks.

The current top-p readiness probe can pass the learning-signal gate but not the update-distribution gate.

### M3.0B-1 — Strict On-policy Collection Smoke

1. switch to the approved update distribution, preferably `temperature=1`, `top_p=1`;
2. collect a small same-task grouped batch;
3. require mixed rewards and complete evidence;
4. mark groups `update_distribution_compatible=true` only after old log-probabilities are verified under that exact distribution;
5. save without updating the model.

### M3.0B-2 — Single-batch Gradient Smoke

1. load one compatible rollout batch;
2. recompute current log-probabilities turn by turn;
3. verify old/current equality before the first update within numerical tolerance;
4. compute group advantages;
5. build the trajectory-normalized clipped loss;
6. backpropagate through LoRA parameters only;
7. require finite, non-zero trainable gradients;
8. perform one optimizer step;
9. require parameter delta and finite post-update log-probabilities;
10. save a disposable checkpoint;
11. reload and run a small development evaluation.

This stage proves optimizer correctness, not capability improvement.

### M3.0B-3 — Small Online Pilot

Initial budget:

```text
8–16 development tasks per update
K = 4
5–20 optimizer updates
1 primary training seed
```

Stop immediately on action-format collapse, false no-solution growth, infrastructure contamination, or general-task regression.

### M3.0C — Formal Comparison

Compare at least:

```text
Canonical Base
M2.2R SFT
M2.3-mini SFT
M3 Agentic RL checkpoint
```

Report:

- overall and task-type success;
- Strict JSON and Schema Valid;
- environment action success;
- trajectory length and termination reasons;
- no-solution precision/recall style confusion counts;
- infrastructure error rate;
- multiple RL seeds for any strong improvement claim.

## 11. Acceptance Criteria

M3.0 is complete only when:

- all training artifacts are reproducible from hashes, seeds, and policy versions;
- old/current token log-probabilities use the same declared distribution;
- each browser turn is replayed under its actual prompt;
- no infrastructure failure contributes reward or gradient;
- RL improves a predeclared development metric over M2.3-mini;
- action validity and false no-solution remain within predeclared tolerances;
- `final_test_v2` is evaluated once after model/hyperparameter freeze;
- at least two RL seeds support any strong performance claim.

## 12. Stop Conditions

Stop and diagnose when:

- all groups have zero reward variance;
- a diagnostic group is incorrectly marked update-compatible;
- old/current log-probabilities differ before the first update;
- raw and sampling-distribution log-probabilities are mixed;
- NaN/Inf appears in loss or gradients;
- Schema Valid collapses;
- no-solution is overproduced;
- browser/database/CUDA failure enters reward data;
- Frozen Test influences checkpoint, temperature, or optimizer selection.

## 13. Deferred Work

Not part of the first implementation:

- learned process reward model;
- learned value function;
- GAE/PPO critic training;
- replay-buffer or off-policy training;
- visual observations;
- vLLM distributed rollout;
- multi-node browser farms;
- generic external websites;
- long-horizon curriculum beyond the procurement environment.
