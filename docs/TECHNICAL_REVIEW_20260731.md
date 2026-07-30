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

MiniWebWork-RL is a focused Agent algorithm internship project with the following approved path:

```text
Deterministic Procurement Environment
→ Custom Text-browser Agent Runtime
→ Canonical Base Evaluation
→ Expert and Recovery SFT
→ Closed-loop Rollout Qualification
→ Multi-turn Outcome-only GRPO-style Update
→ Frozen Final Evaluation
```

It is not:

- a general public-Web benchmark;
- a visual browser Agent;
- a LangChain/LangGraph demonstration;
- a stock one-shot TRL GRPO example;
- a process-reward or value-model project;
- a paper-level SOTA claim.

## 3. Corrected Technical Paths

### 3.1 “No Agent framework” was an incorrect diagnosis

The repository already contains:

- a typed browser environment;
- Observation/Action contracts;
- a Qwen policy runtime;
- bounded history and feedback;
- episode lifecycle and trajectory capture;
- a deterministic Verifier.

This is a minimal custom Agent runtime. The missing layer is online policy optimization.

### 3.2 Step reward shaping is not an RL prerequisite

The first RL baseline uses only:

```text
success = 1
valid policy failure = 0
infrastructure failure = null
```

Rewards for valid JSON, clicks, empty results, no-solution declarations, or shorter trajectories were rejected because they can optimize proxies and cause reward hacking.

### 3.3 Experience replay, value model, and GAE are not required

The first GRPO-style implementation is on-policy and group-relative. It does not require:

- replay buffer;
- critic/value network;
- GAE;
- PPO value loss.

These remain optional later extensions.

### 3.4 Teacher-forced accuracy is not closed-loop capability

Near-perfect Valid next-action accuracy proved that SFT learned expert-state mappings. Frozen E2E was still required to establish autonomous multi-turn behavior.

### 3.5 Prompt Contract is part of the model

The compact/full Prompt mismatch caused the initial M2.2 result to be invalid. All active paths now use Canonical Prompt Contract v2. A prompt change requires a new version and Base/SFT re-baselining.

### 3.6 No-solution weakness was not simply “too little data”

The principal deficiency was state and combination coverage:

- empty-result recognition;
- no-solution terminal flow;
- off-expert recovery states;
- multi-constraint combinations;
- error recovery after policy deviations.

M2.3-mini therefore added targeted expert and recovery states rather than merely duplicating ideal trajectories.

### 3.7 Diagnostic rollout and optimizer rollout are distinct

A readiness probe using temperature/top-p can show exploration and mixed reward. It does not automatically create an optimizer-compatible on-policy batch.

The repository now separates:

```text
has_learning_signal
update_distribution_compatible
valid_for_grpo_update
```

### 3.8 A multi-turn browser episode is not one completion

Every browser turn uses a new Observation and a newly rendered Prompt. The approved optimizer replays each turn independently, concatenates action-token evidence at trajectory level, broadcasts one terminal group advantage, normalizes within trajectory, then averages trajectories equally.

### 3.9 Raw policy and sampling probabilities are different

The rollout artifact stores:

- raw model-policy token log-probabilities;
- post-temperature/top-p sampling-distribution log-probabilities.

They must not be mixed in an importance ratio. The first strict optimizer batch should prefer `temperature=1.0, top_p=1.0`.

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
- Slurm owns GPU visibility.

### Web and Database

- product/supplier/form navigation preserves episode/task context;
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

### Metrics

Separate denominators are used for:

- Strict JSON;
- Schema Valid;
- environment action success;
- task success;
- infrastructure error rate.

## 5. Removed or Superseded Paths

Removed:

- old M2.3 temperature-sweep probe;
- old M2.3 comparison probe;
- browser-agent-v1 SFT builder;
- duplicate Sync Playwright Observation extractor;
- host-specific requirements/Conda dumps;
- unsafe broad process cleanup in supported Slurm jobs.

Historical reports remain evidence of prior stages but are not current execution authority.

## 6. Current Readiness Assessment

| Capability | Status |
|---|---|
| deterministic environment | pass |
| custom Agent runtime | pass |
| Canonical Prompt/Action contracts | pass |
| expert/recovery SFT | pass |
| Adapter artifact validity | pass |
| typed rollout artifact | implemented |
| policy/infrastructure reward separation | implemented |
| raw/sampling log-prob evidence | implemented |
| multi-turn GRPO-style objective | implemented and unit-tested in repository |
| canonical M2.3 GPU A/B rerun | pending |
| strict on-policy update batch | not started |
| optimizer/checkpoint smoke | not started |
| final_test_v2 | not created |

Formal state remains:

```text
READY_FOR_GRPO=false
```

## 7. Immediate Execution Order

1. run CPU quality gate and GitHub CI;
2. run Policy B single-task Slurm smoke;
3. run canonical Policy A/B readiness probes;
4. verify infrastructure errors are near zero;
5. verify token/log-prob coverage is 100%;
6. verify at least one no-solution success and mixed-reward group;
7. collect a separate strict update-distribution batch;
8. run one-batch gradient/checkpoint smoke;
9. only then begin a small online pilot.

## 8. Main Residual Risks

| Risk | Current treatment |
|---|---|
| only a small historical Frozen Test | create final_test_v2 before final comparison |
| trajectory-level credit is coarse | state explicitly; process reward deferred |
| M2.3 could overproduce no-solution | report false/missed no-solution separately |
| top-p diagnostic batch is off-contract for first update | update compatibility remains false |
| GPU runtime cannot be validated by GitHub CPU CI | mandatory Slurm smoke and formal rerun |
| single RL seed may be unstable | require at least two seeds for strong claims |

## 9. Overall Review

The revised design is appropriate for the stated internship objective. Its strength is not scale; it is the completeness and auditability of the path from deterministic environment to supervised policy, grouped multi-turn rollout, correct reward attribution, and a minimal online policy update.

The next meaningful result is not another code refactor. It is the canonical M2.3 A/B GPU rollout and the strict on-policy single-batch update smoke.
