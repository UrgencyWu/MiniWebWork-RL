# Architecture and Runtime Contracts

## 1. Architecture

```text
┌──────────────────────────────────────────────────────────┐
│ Task and Data Layer                                      │
│ public task / private oracle / supplier-product database │
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
│ bounded history / action feedback / trajectory evidence  │
└──────────────────────────┬───────────────────────────────┘
                           │ Action
┌──────────────────────────▼───────────────────────────────┐
│ Deterministic Verifier                                   │
│ recompute constraints and optimum from private Oracle    │
└──────────────────────────┬───────────────────────────────┘
                           │ reward ∈ {0,1} or null
┌──────────────────────────▼───────────────────────────────┐
│ Training and Evaluation                                  │
│ Expert SFT / rollout groups / planned outcome-only GRPO  │
└──────────────────────────────────────────────────────────┘
```

The project already has a custom Agent runtime. It intentionally does not use LangChain, LangGraph, AutoGen, or Qwen-Agent. The missing component is not an “Agent framework”; it is the on-policy RL optimizer and training scheduler.

## 2. Task Source Contract

A process must read tasks from exactly one source:

```text
no MINIWEBWORK_TASK_DIR
→ data/tasks/tasks_public.jsonl + tasks_oracle.jsonl

explicit task_dir or MINIWEBWORK_TASK_DIR
→ only *_public.jsonl + *_oracle.jsonl in that directory
```

Default and development datasets are never merged. Duplicate `task_id` values fail fast.

Public files contain instructions and non-answer metadata. Oracle files contain constraints, expected decision type, and expected product. Oracle content must never enter model prompts or rollout policy inputs.

## 3. Observation Contract

An Observation is a text-only snapshot containing:

- `task_id`, `episode_id`, instruction;
- page URL/path and typed `page_type`;
- bounded visible text;
- interactive elements with stable `element_id`, role, label, value, options, and disabled state;
- previous action result;
- terminal flag.

The model does not receive screenshots or hidden DOM state. Layout recovery and visual grounding are out of scope.

## 4. Action Contract

The model emits one strict JSON object per turn. Supported actions are:

```text
click / fill / select / check / back / submit / finish
```

A parseable JSON object is not automatically Schema Valid. Action-specific target/value requirements are validated separately.

Schema-invalid output is a policy output failure. The runner must not replace it with a fabricated `finish` action. After a bounded number of consecutive output failures, the episode terminates as `model_output_failure_limit`.

## 5. Prompt Contract

All SFT, teacher-forced evaluation, Frozen E2E, and rollout collection use:

```text
PROMPT_VERSION = browser_agent_v2
HISTORY_WINDOW = 5
```

The rendered prompt is identified by:

- prompt contract version;
- prompt-builder source SHA-256;
- chat-template SHA-256;
- per-turn prompt hash;
- exact prompt token IDs in rollout artifacts.

Changing the prompt contract requires a new version and re-baselining Base/SFT under the same contract.

## 6. Environment and Browser Contract

Playwright runs in a persistent dedicated thread using `async_playwright`. All Browser, Context, Page, navigation, action, and observation operations remain in that worker thread.

Lifecycle invariant:

```text
start
→ event loop ready
→ Playwright started
→ browser connected
→ context/page created
→ actions
→ page/context/browser/Playwright closed
→ loop stopped
→ worker joined
```

A browser, database, service, CUDA, parser implementation, or cleanup exception is an infrastructure failure. It must not become a policy reward of zero.

## 7. Verifier Contract

The Verifier is deterministic and has no LLM dependency. It:

1. resolves the Oracle from the same task source as the environment;
2. verifies that `episode_id` exists and belongs to `task_id`;
3. requires one persisted submission;
4. recomputes feasible products from SQLite;
5. checks every explicit constraint;
6. recomputes objective optimality;
7. returns a structured result and failure reasons.

Connections are closed on every return path.

Terminal reward:

```text
verified success  → 1.0
valid policy fail → 0.0
infrastructure fail → null
```

## 8. Rollout Evidence Contract

Every model turn stores:

- prompt hash and exact prompt token IDs;
- raw generated text;
- exact generated token IDs;
- old-policy raw token log-probabilities;
- strict JSON and Schema status;
- parsed action;
- environment action result;
- termination/truncation flags.

Raw policy log-probabilities are computed by a correctly aligned teacher-forced forward over `prompt + completion`. Post-top-p generation scores are a separate diagnostic field and are not the canonical GRPO score.

Every trajectory stores:

- task/policy/adapter/prompt/task-source identity;
- deterministic rollout seed;
- full turn sequence;
- terminal verification;
- `rollout_valid` and `failure_origin`;
- reward or null.

## 9. Agentic RL Boundary

M3.0 will add:

- grouped stochastic rollout collection;
- trajectory-level outcome reward;
- group-relative advantage;
- token-level clipped policy objective;
- optional reference-policy KL regularization;
- optimizer, checkpoint, and Frozen evaluation loop.

The first implementation does not require:

- a value network;
- generalized advantage estimation;
- replay buffer;
- step-level handcrafted reward;
- visual model;
- distributed browser farm.

These may be later ablations, not prerequisites.
