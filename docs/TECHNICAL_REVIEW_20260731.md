# Technical Review — 2026-07-31

## 1. Review Scope

This review covered:

- project positioning and stage claims;
- Agent runtime versus Agentic RL boundary;
- task/public/Oracle isolation;
- Prompt, Observation, Action, and history contracts;
- browser/thread/process/database lifecycle;
- deterministic verification and reward attribution;
- rollout evidence and probability semantics;
- multi-turn GRPO-style optimization;
- Slurm shared-node safety;
- dependency, CI, test, and documentation governance.

## 2. Final Technical Position

MiniWebWork-RL is a focused Agent algorithm internship project with the approved path:

```text
Deterministic Procurement Environment
→ Custom Text-browser Agent Runtime
→ Canonical Base Evaluation
→ Expert and Recovery SFT
→ Closed-loop Rollout Qualification
→ Multi-turn Outcome-only GRPO-style Update
→ Frozen Final Evaluation
```

It is not a general public-Web benchmark, visual browser Agent, Agent-framework demonstration, stock one-shot TRL GRPO example, process-reward project, or paper-level SOTA claim.

## 3. Corrected Technical Paths

### 3.1 The repository already has an Agent runtime

The repository contains a typed browser environment, Observation/Action contracts, Qwen policy runtime, bounded history, lifecycle management, trajectory capture, and deterministic Verifier. The missing layer is online policy optimization.

### 3.2 Outcome-only reward is the first baseline

```text
success = 1
valid policy failure = 0
infrastructure failure = null
```

JSON, clicks, empty-result detection, no-solution declarations, form completion, and trajectory length remain diagnostics rather than rewards.

### 3.3 Value models and replay are not prerequisites

The first GRPO-style implementation is on-policy and group-relative. It does not require a replay buffer, critic/value network, GAE, or PPO value loss.

### 3.4 Expert-state accuracy is not closed-loop capability

Near-perfect next-action validation measures expert-state imitation. Autonomous multi-turn rollout remains the capability test.

### 3.5 Prompt Contract is part of the policy

All active model paths use Canonical Prompt Contract v2. A Prompt change creates a new experiment family and requires Base/SFT re-baselining.

### 3.6 M2.3-mini targeted state coverage

The no-solution gap involved empty-result recognition, terminal submission, multi-constraint combinations, and off-expert recovery. M2.3-mini therefore added targeted expert and recovery states instead of duplicating ideal trajectories.

### 3.7 Diagnostic and optimizer rollouts are distinct

The system separates:

```text
has_learning_signal
update_distribution_compatible
valid_for_grpo_update
```

A diagnostic group can have mixed reward while remaining ineligible for update.

### 3.8 A multi-turn episode is not one completion

Each browser turn has a new Observation and freshly rendered Prompt. The optimizer replays each turn independently, aggregates action-token evidence at trajectory level, broadcasts one terminal advantage, averages within trajectory, then averages trajectories equally.

### 3.9 Complete sampling identity is mandatory

Every schema-v3.3 trajectory and group records:

```text
temperature
top_p
top_k
```

Raw-policy and post-processor sampling log-probabilities remain separate. The first strict distribution is:

```text
temperature = 1.0
top_p = 1.0
top_k = 0
```

A group is update-compatible only when raw and sampling token log-probabilities also agree within the predeclared numerical tolerance. Callers cannot bypass this parameter gate.

### 3.10 Aggregate A/B counts are insufficient

A single `58 vs 43` total cannot establish patch superiority or be dismissed as variance. Formal comparison pairs `(task_id, rollout_index)`, reports discordant outcomes, per-task differences, exact McNemar results, task bootstrap intervals, and multiple master seeds.

## 4. Corrected Engineering Contracts

### Task and Oracle

- one exclusive task source per process;
- default and development datasets never merge;
- duplicate task IDs fail fast;
- Oracle never enters prompts;
- episode must belong to task.

### Browser and Environment

- one async Playwright worker thread;
- browser objects never cross threads;
- startup/shutdown errors propagate;
- exact child process/thread cleanup only;
- no shared-node `pkill -f`;
- Slurm owns GPU visibility;
- temporary lifecycle monkey-patch preserves the original `headless` signature and has a regression test.

### Web and Database

- navigation preserves episode/task context;
- numeric filter errors do not silently disappear;
- one submission per episode;
- positive quantity;
- active-episode-only submission;
- transaction rollback on failure.

### Verifier and Reward

- deterministic constraint and optimality recomputation;
- deterministic tie-breaking;
- structured failure reasons;
- policy failure reward 0;
- infrastructure failure reward null.

### Rollout and Replay

- exact prompt and completion token IDs;
- finite raw-policy and sampling log-probabilities;
- complete probability evidence required for replay;
- same task/policy/distribution within each group;
- warped distributions cannot be marked strict-update compatible;
- infrastructure-invalid records never enter advantages or gradients.

## 5. Current Readiness Assessment

| Capability | Status |
|---|---|
| deterministic environment | pass |
| custom Agent runtime | pass |
| Canonical Prompt/Action contracts | pass |
| expert/recovery SFT | pass |
| Adapter artifact validity | pass |
| historical M2.3 readiness GPU probe | pass |
| no-solution closed-loop capability | confirmed |
| raw/sampling log-prob coverage | confirmed on readiness run |
| schema-v3.3 complete sampling identity | implemented; cluster rerun pending |
| paired A/B analysis | implemented; real artifacts pending |
| typed strict replay batch | implemented and guarded |
| multi-turn GRPO-style objective | implemented and unit-tested in repository |
| strict update-compatible group | not yet collected |
| optimizer/checkpoint smoke | not started |
| feasible rollout-dev slice | pending |
| final_test_v2 | not created |

Formal state:

```text
M2_3_MINI_CANONICAL_PROBE_PASS=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=false
```

## 6. Immediate Execution Order

1. run the CPU quality gate;
2. collect schema-v3.3 A/B diagnostics with explicit `T=0.2, top_p=0.9, top_k=0`;
3. perform paired A/B analysis across additional master seeds;
4. add feasible development tasks and check false no-solution/general regression;
5. select the M3.0 starting policy using the predeclared rule;
6. collect `T=1, top_p=1, top_k=0` strict groups;
7. require probability-match and at least one `valid_for_grpo_update=true` group;
8. run one-batch LoRA gradient/checkpoint/reload smoke;
9. only then begin a 5–10-update online pilot.

## 7. Main Residual Risks

| Risk | Current treatment |
|---|---|
| historical Frozen Test is not pristine | create final_test_v2 before formal comparison |
| trajectory-level credit is coarse | state explicitly; process reward deferred |
| M2.3 may overproduce no-solution | add feasible dev slice and confusion metrics |
| historical readiness artifact omitted top-k | capability evidence only; recollect schema-v3.3 |
| hidden generation processors | explicit top-k plus raw/sampling probability-match gate |
| caller could mislabel update compatibility | non-bypassable strict parameter validation |
| GPU runtime cannot be validated by CPU CI | mandatory Slurm collection and optimizer smoke |
| single RL seed may be unstable | require at least two seeds for strong claims |

## 8. Overall Review

The project now has an auditable path from deterministic environment through SFT, qualified multi-turn rollout, explicit behavior-distribution evidence, paired policy comparison, typed replay batches, and a bounded GRPO-style objective.

The next meaningful result is experimental rather than architectural: collect a schema-v3.3 strict group, prove old/current replay consistency, and execute one LoRA optimizer step with a reloadable checkpoint.
