# MiniWebWork-RL Project Journal

> Chronological project record. Current authority is `docs/CURRENT_STATUS.md`; this file preserves major decisions, results, and lessons.

## 0. Project Positioning

Goal: train Qwen3.5-4B into a text-browser procurement Agent and establish a minimal, auditable Agentic RL path.

Constraints:

- local deterministic website rather than uncontrolled public websites;
- GPU work through Slurm;
- GitHub + local/ModelScope model assets;
- no Docker requirement;
- text observations and fixed JSON actions;
- non-LLM terminal verification;
- internship-project quality rather than SOTA benchmark claims.

The core path is:

```text
Environment
→ Agent Runtime
→ Base Evaluation
→ Expert Trajectories
→ LoRA SFT
→ Closed-loop Evaluation
→ Grouped Rollout
→ Outcome-only GRPO
```

## 1. M1.0 — Runtime Baseline

Completed:

- Python 3.11 Conda environment;
- Playwright and headless Chromium;
- local Qwen3.5-4B model loading;
- Slurm browser/model smoke tests.

Key lesson: Slurm owns GPU visibility. Python must not rewrite `CUDA_VISIBLE_DEVICES` after CUDA initialization.

Historical details remain in `M1_0_ENVIRONMENT_REPORT.md` and `M1_0_COMMAND_LOG.md`.

## 2. M1.1 — Deterministic Procurement Environment

Implemented:

- 6 fictional suppliers and 24 fictional products in SQLite;
- FastAPI/Jinja2 website;
- 15 public/private task pairs;
- deterministic constraint and objective recomputation;
- structured Verifier failure reasons;
- Playwright end-to-end tests.

Important corrections:

- `certified_only` uses `None` for “no constraint” rather than `False`;
- task answers are recomputed, not hand-trusted;
- task, episode, and submission IDs must stay bound through the form flow.

## 3. M1.2 — Agent Environment Contract

Implemented:

```text
reset(task_id) -> Observation
step(AgentAction) -> StepResult
close()
```

Observation includes page type, visible text, interactive elements, and previous action result. The action space contains seven JSON actions. A rule-based baseline achieved 2/15.

The environment later moved to a persistent async Playwright worker thread to remove Sync/Async and greenlet cross-thread failures.

## 4. M2.0 — Base Qwen Browser Agent

Implemented:

- Canonical model backend;
- prompt builder;
- strict-first JSON parser with bounded fallback;
- browser loop and trajectory recording.

After metric correction:

- Strict JSON: 174/184;
- effective JSON: 184/184;
- environment action success: 34/184;
- task success: 5/15.

Lesson: parse success, Schema Valid, environment action success, and task success are distinct metrics.

## 5. M2.1F — Expert Data and SFT Dataset

Final frozen scope:

- 96 train tasks;
- 24 valid tasks;
- 120 successful expert trajectories;
- 723 train step samples;
- 174 valid step samples;
- 897 total completion-only SFT samples;
- zero cross-split duplicate IDs;
- zero Frozen Test/Oracle leakage;
- completion masking checks passed.

Task types include exact product, cheapest feasible, highest-rating supplier, and no-feasible-product.

## 6. M2.2 — Initial SFT and Contract Drift

Three Qwen3.5-4B LoRA seeds trained successfully. Initial offline next-action results were only about 40% Exact Match, and all Frozen E2E tasks failed.

The primary cause was not model capacity. Training used a compact prompt/observation contract while E2E used a larger runtime contract. Additional evaluator defects also existed.

This stage is superseded by M2.2R.

## 7. M2.2R — Canonical Contract and Closed-loop Validation

Canonical Prompt Contract v2 unified:

- SFT construction;
- teacher-forced action evaluation;
- Base evaluation;
- Frozen browser E2E.

Teacher-forced Valid:

| Policy | Exact Match | Schema Valid |
|---|---:|---:|
| Base Canonical v2 | 52.3% | 93.7% |
| SFT seed 42 | 99.4% | 100% |
| SFT seed 1234 | 100% | 100% |
| SFT seed 20260726 | 100% | 100% |

Frozen 15-task E2E:

| Policy | Success |
|---|---:|
| Base Canonical v2 | 0/15 |
| SFT seed 42 | 9/15 |
| SFT seed 1234 | 10/15 |
| SFT seed 20260726 | 12/15 |

This proved that SFT formed a usable closed-loop rollout policy. It did not prove final generalization because the test set is small and later became development-visible.

Checkpoint selection remained based on Canonical Valid Loss. `seed_1234` is the formal primary policy.

## 8. M3.0A — Rollout Readiness Audit

The no-solution slice was audited before policy-gradient training.

Results:

- Expert replay passed, confirming the environment/Verifier submission path;
- deterministic SFT evaluation solved only a narrow subset of no-solution tasks;
- stochastic probes for two policies produced 0/16 successes and zero group reward variance.

Decision:

```text
Route B → targeted SFT coverage/recovery patch
```

Reason: outcome-only GRPO cannot learn from a task group whose rewards are all equal.

## 9. M2.3-mini — No-solution and Recovery Patch

Data:

- 28 train and 10 valid new expert tasks;
- 300 off-expert recovery states;
- 900 mixed train samples;
- 249 mixed valid samples;
- patch ratio about 25%;
- no Frozen Test leakage.

Training:

- continued from formal `seed_1234` adapter;
- train loss `0.000975`;
- patch Valid Exact Match 100%;
- patch Valid Schema Valid 100%;
- adapter artifact passed CPU/CUDA forward and generation audits.

The adapter itself was not the source of earlier CUDA crashes. Device-map handling, CUDA visibility timing, browser process stability, and an incorrect token-logprob index path were separately corrected.

## 10. 2026-07-31 Repository Consolidation

The repository was consolidated around one formal rollout path:

```text
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
```

Removed:

- obsolete temperature-sweep probe;
- obsolete comparison probe;
- their Slurm wrappers;
- host-specific environment dumps.

Added or hardened:

- exclusive task-source loading and duplicate-ID rejection;
- episode/task binding in the Verifier;
- connection closure on every Verifier path;
- typed rollout records;
- prompt/completion token retention;
- correctly aligned raw policy token log-probabilities;
- infrastructure `reward=null` contract;
- content hashing and deterministic rollout seeds;
- CPU CI and a project quality-gate script;
- authoritative architecture, governance, and M3.0 documents.

## 11. Current Decision

```text
M2_3_MINI_TRAINING_PASS=true
CANONICAL_ROLLOUT_GPU_RERUN_PENDING=true
READY_FOR_GRPO=false
```

Next:

1. run single-task Policy B smoke;
2. run controlled Policy A/B canonical probes;
3. verify no-solution success and mixed-reward groups;
4. verify token/logprob evidence completeness;
5. only then start M3.0B single-batch GRPO smoke.

## 12. Durable Lessons

1. Prompt/observation contracts are part of the model, not peripheral formatting.
2. Teacher-forced action accuracy and closed-loop task success measure different capabilities.
3. Infrastructure failures must never become policy rewards.
4. Frozen Test results cannot select checkpoints or rollout temperature.
5. Outcome-only GRPO requires within-group reward variance.
6. Successful expert trajectories alone do not cover recovery states.
7. Raw policy log-probabilities must be aligned to exact prompt/completion tokens.
8. A small project benefits from one authoritative path more than many partially overlapping scripts.

*Last updated: 2026-07-31*
