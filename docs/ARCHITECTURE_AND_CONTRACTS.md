# Architecture and Runtime Contracts

## 1. Architecture

```text
┌──────────────────────────────────────────────────────────┐
│ Task and Data Layer                                      │
│ public task / private Oracle / supplier-product database │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│ Browser Environment                                      │
│ FastAPI + SQLite + Playwright                            │
│ reset(task_id) / step(action) / close()                  │
└──────────────────────────┬───────────────────────────────┘
                           │ Observation
┌──────────────────────────▼───────────────────────────────┐
│ Agent Runtime                                             │
│ Canonical Prompt v2 → Qwen3.5-4B → JSON parser → action │
│ bounded history / compact feedback / token evidence      │
└──────────────────────────┬───────────────────────────────┘
                           │ Action
┌──────────────────────────▼───────────────────────────────┐
│ Deterministic Verifier                                   │
│ recompute constraints and optimum from private Oracle    │
└──────────────────────────┬───────────────────────────────┘
                           │ reward ∈ {0,1} or null
┌──────────────────────────▼───────────────────────────────┐
│ Training and Evaluation                                  │
│ Expert SFT / grouped multi-turn rollout / GRPO-style RL  │
└──────────────────────────────────────────────────────────┘
```

The project already contains a custom Agent runtime. It intentionally does not use LangChain, LangGraph, AutoGen, or Qwen-Agent. The missing capability is the online multi-turn policy optimizer, not an “Agent framework.”

## 2. Task Source Contract

A process reads tasks from exactly one source:

```text
no MINIWEBWORK_TASK_DIR
→ data/tasks/tasks_public.jsonl + tasks_oracle.jsonl

explicit task_dir or MINIWEBWORK_TASK_DIR
→ only *_public.jsonl + *_oracle.jsonl in that directory
```

Default and development datasets are never merged. Duplicate `task_id` values fail fast.

Public files contain instructions and non-answer metadata. Oracle files contain constraints, expected decision type, and expected product. Oracle content never enters prompts, observations, policy history, or checkpoint selection.

## 3. Observation Contract

An Observation is a text-only browser snapshot containing:

- `task_id`, `episode_id`, instruction;
- URL/path and typed `page_type`;
- bounded visible text;
- interactive elements with stable `element_id`, role, name, value, options, and disabled state;
- compact previous action result;
- terminal flag.

The model does not receive screenshots, hidden DOM state, SQLite rows, or Oracle fields. Layout recovery and visual grounding are outside the first project scope.

Observation extraction has one canonical async implementation inside the Playwright worker. The obsolete Sync Playwright extractor was removed to prevent divergent element IDs and truncation behavior.

## 4. Action Contract

The model emits one JSON object per turn. Action Schema v1.1 supports:

```text
click / fill / select / check / back / submit / finish
```

A parseable JSON object is not automatically Schema Valid. Action-specific target/value requirements and observed role compatibility are validated independently.

`submit` may target a button. It remains distinct from `click` for trajectory semantics.

Schema-invalid model output:

```text
counts as a model turn
→ does not call env.step
→ is never replaced by synthetic finish
→ terminates after a bounded consecutive failure limit
```

## 5. Prompt and History Contract

All active SFT, teacher-forced evaluation, Base/SFT closed-loop evaluation, and rollout collection use:

```text
PROMPT_VERSION = browser_agent_v2
HISTORY_WINDOW = 5
```

The history stores only:

```text
previous parsed action
parse status
compact deterministic action result
resulting page type
```

It does not recursively embed the next full Observation. The current Observation appears once in the current turn prompt.

Prompt identity contains:

- prompt contract version;
- prompt-builder source SHA-256;
- tokenizer chat-template SHA-256;
- per-turn prompt hash;
- exact prompt token IDs.

Changing this contract creates a new experiment family and requires re-baselining Base and trained policies under the same version.

## 6. Environment and Browser Contract

Playwright runs through `async_playwright` inside one dedicated worker thread. All Browser, Context, Page, navigation, action, and observation operations remain in that thread.

Lifecycle invariant:

```text
thread start
→ event loop ready
→ Playwright/browser start
→ fresh context/page
→ browser actions and observations
→ page/context close
→ browser/Playwright close
→ event loop stop
→ worker join
```

The main loop exposes synchronous `reset / step / close`. Startup, step, service, database, timeout, and cleanup exceptions are infrastructure failures.

The FastAPI child process receives task/database paths through its own subprocess environment. The parent process does not mutate global task/database state per episode.

Shared-node rule: cleanup is by exact process/thread handle only. Broad `pkill -f chromium` or `pkill -f uvicorn` is forbidden.

## 7. Web and Persistence Contract

Every task flow preserves both:

```text
episode_id + task_id
```

across product, supplier, form, and result pages. The Web layer rejects mismatched episode/task pairs.

SQLite enforces:

- valid decision type;
- positive quantity;
- product presence for `select_product`;
- no product for `no_solution`;
- one final submission per episode;
- active-episode-only submission;
- transactional insert and episode status update.

Numeric filter errors return a structured client error instead of being silently ignored.

## 8. Verifier Contract

The Verifier is deterministic and has no LLM dependency. It:

1. resolves the Oracle from the same exclusive task source;
2. verifies that the episode exists and belongs to the requested task;
3. requires one persisted submission;
4. recomputes feasible products from SQLite;
5. checks every explicit constraint;
6. recomputes objective optimality with deterministic tie-breaking;
7. returns structured success and failure reasons.

All database connections close on every path.

Terminal reward:

```text
verified success       → 1.0
valid policy failure   → 0.0
infrastructure failure → null
```

## 9. Rollout Evidence Contract

Every model turn stores:

- exact prompt token IDs;
- exact generated action token IDs;
- raw model-policy token log-probabilities;
- post-generation-processor sampling log-probabilities;
- prompt hash and raw text;
- Strict JSON, fallback, and Schema status;
- parsed action and environment action result;
- termination/truncation flags.

Raw policy and sampling-distribution log-probabilities are separate fields. A readiness probe using temperature/top-p proves exploration and reward variance; it does not automatically prove compatibility with the optimizer distribution.

Every trajectory stores:

- task/policy/adapter/prompt/task-source identity;
- deterministic rollout seed;
- ordered turn sequence;
- terminal verification;
- `rollout_valid`, `failure_origin`, and reward or null.

Every group distinguishes:

```text
has_reward_variance
has_learning_signal
update_distribution_compatible
valid_for_grpo_update
```

A mixed-reward top-p diagnostic group may have learning signal while remaining ineligible for direct policy update.

## 10. Multi-turn Agentic RL Boundary

Each browser turn is a distinct conditional generation:

```text
Observation_t + bounded history_t
→ Prompt_t
→ JSON action tokens_t
```

The full trajectory is not represented as one artificial completion. M3.0 replays each turn under its actual prompt, concatenates action-token log-probability segments within the trajectory, applies one terminal group-relative advantage, averages tokens within each trajectory, then averages trajectories equally.

Core objective implementation:

```text
src/miniwebwork/rl/objective.py
```

The first strict update prefers:

```text
temperature = 1.0
top_p = 1.0
```

so the behavior distribution and raw policy distribution coincide. Any temperature-scaled fallback must recompute old/current log-probabilities under the same scaling. `top_p < 1` is diagnostic-only until the exact truncated distribution is supported in training.

The first implementation does not require:

- a value network;
- generalized advantage estimation;
- replay buffer;
- handcrafted step reward;
- visual model;
- distributed browser farm.

These are later extensions or ablations, not prerequisites.
