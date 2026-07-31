# Technical Review — 2026-07-31

## 1. Review Scope

This review covered project positioning, Agent runtime and Agentic RL boundaries, task/Oracle isolation, Prompt/Observation/Action contracts, browser and database lifecycle, deterministic verification, rollout probability semantics, multi-turn policy optimization, Slurm safety, test coverage, and documentation governance.

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

### 3.1 Existing Agent runtime

The repository already contains a typed browser environment, Qwen policy runtime, bounded action history, lifecycle management, trajectory capture, and deterministic Verifier. The missing capability is the online policy-improvement loop, not an Agent framework.

### 3.2 First reward and optimization baseline

```text
success = 1
valid policy failure = 0
infrastructure failure = null
```

JSON validity, clicks, empty-result recognition, no-solution declarations, form completion, and trajectory length remain diagnostics. The first optimizer does not require a value model, GAE, replay buffer, PPO critic, or handcrafted process reward.

### 3.3 Multi-turn rather than one-shot optimization

Every browser turn receives a new Observation and freshly rendered Prompt. A trajectory is therefore replayed as ordered per-turn Prompt/action segments. One terminal group-relative advantage is broadcast to all action tokens; token means are computed within each trajectory, and trajectories are averaged equally.

### 3.4 Complete behavior-distribution identity

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

Strict compatibility additionally requires finite, complete raw/sampling token probabilities whose maximum absolute difference remains within the declared numerical tolerance. The replay layer recomputes this condition from records; callers cannot promote a diagnostic artifact with a boolean flag.

### 3.5 Diagnostic versus optimizer evidence

The system distinguishes:

```text
has_learning_signal
update_distribution_compatible
valid_for_grpo_update
```

A mixed-reward readiness group can have learning signal while remaining ineligible for update.

### 3.6 Policy comparison requires paired and feasible evidence

A single `58 vs 43` aggregate cannot establish superiority or be dismissed as sampling variance. Formal comparison pairs `(task_id, rollout_index)`, reports discordant outcomes, exact McNemar results, task-level bootstrap intervals, per-task differences, feasible success, false no-solution, no-solution success, and multiple master seeds.

The frozen `rollout_dev_feasible_v1` slice contains 12 select-product tasks and is a non-training policy-selection gate.

## 4. Corrected Engineering Contracts

### Task, Web, and Verifier

- one exclusive task source per process;
- Public/Oracle one-to-one binding and duplicate-ID rejection;
- Oracle never enters prompts;
- episode/task context preserved through navigation and submission;
- one active-episode submission, positive quantity, transactional persistence;
- deterministic feasibility and objective recomputation;
- policy failure reward 0, infrastructure failure reward null.

### Browser lifecycle

- one async Playwright worker thread;
- browser objects stay in that worker;
- startup and shutdown errors propagate;
- cleanup uses exact owned handles only;
- no shared-node `pkill -f`;
- Slurm owns GPU visibility;
- the temporary lifecycle guard preserves the original `headless` signature and has a regression test.

### Rollout, replay, and optimizer

- exact prompt and completion token IDs;
- finite raw-policy and sampling log-probabilities;
- malformed infrastructure evidence is serialized safely but never becomes replay input;
- same task, policy, and complete distribution within each group;
- concurrent task-source/seed/distribution jobs use isolated output directories;
- replay independently verifies strict probability compatibility;
- the optimizer smoke verifies artifact/adapter hash binding and old/current equality;
- only LoRA parameters may be trainable;
- per-turn streaming backward preserves equal-trajectory weighting while limiting 4B-model memory;
- one update must produce finite non-zero gradients, non-zero Adapter delta, a reloadable checkpoint, and a finite reload forward.

## 5. Current Readiness Assessment

| Capability | Status |
|---|---|
| deterministic environment and Agent runtime | pass |
| Canonical Prompt/Action contracts | pass |
| expert/recovery SFT | pass |
| Adapter artifact validity | pass |
| historical M2.3 readiness GPU probe | pass |
| no-solution closed-loop capability | confirmed |
| schema-v3.3 complete sampling identity | implemented; cluster run pending |
| paired A/B analysis | implemented; real multi-seed artifacts pending |
| feasible policy-selection slice | implemented and frozen; cluster run pending |
| strict replay batch | implemented and non-bypassable |
| multi-turn clipped objective | implemented |
| streaming equal-trajectory loss | implemented |
| one-batch optimizer/checkpoint smoke | implemented; cluster run pending |
| strict update-compatible group | not yet collected |
| final_test_v2 | not created |

Formal state:

```text
M2_3_MINI_CANONICAL_PROBE_PASS=true
SCHEMA_V3_3_ROLLOUT_IMPLEMENTED=true
PAIRED_AB_ANALYSIS_IMPLEMENTED=true
FEASIBLE_ROLLOUT_DEV_IMPLEMENTED=true
M3_0B1_SINGLE_BATCH_SMOKE_IMPLEMENTED=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=false
```

## 6. Immediate Execution Order

1. run the CPU quality gate;
2. collect schema-v3.3 no-solution A/B diagnostics for multiple master seeds;
3. run the feasible A/B gate and inspect false no-solution/general-task success;
4. perform paired analyses and freeze the M3.0 starting policy;
5. collect `T=1, top_p=1, top_k=0` strict groups from the selected policy;
6. require probability-match and at least one `valid_for_grpo_update=true` group;
7. run `m3_0_single_batch_smoke.sbatch` on one strict group;
8. only after the smoke passes, begin a 5–10-update online pilot.

## 7. Residual Risks

| Risk | Current treatment |
|---|---|
| historical Frozen Test is not pristine | create final_test_v2 before formal comparison |
| trajectory-level credit is coarse | state explicitly; process reward deferred |
| M2.3 may overproduce no-solution | feasible non-training gate and confusion metrics |
| historical readiness artifact omitted top-k | capability evidence only; recollect schema-v3.3 |
| hidden generation processors | explicit top-k plus raw/sampling probability-match gate |
| GPU runtime cannot be validated by CPU CI | mandatory Slurm strict collection and optimizer smoke |
| one RL seed may be unstable | at least two seeds for strong claims |

## 8. Overall Review

The repository now contains an auditable implementation from deterministic environment through SFT, qualified multi-turn rollout, explicit behavior-distribution evidence, paired policy comparison, feasible regression gating, typed replay construction, trajectory-normalized objective, and a bounded one-batch LoRA optimizer smoke.

The next result is experimental: select the starting policy, collect a strict mixed-reward group, and demonstrate one finite reloadable optimizer update on the cluster.
