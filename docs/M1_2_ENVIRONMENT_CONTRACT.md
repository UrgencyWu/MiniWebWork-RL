# Environment Contract

> Introduced in M1.2 and updated to the current persistent async Playwright implementation.

## Interface

```python
env = ProcurementBrowserEnv(
    max_steps=20,
    run_id="...",
    headless=True,
    keep_db=True,
    task_dir=None,
)

observation = env.reset(task_id)
result = env.step(action)
env.close()
```

| Parameter | Default | Meaning |
|---|---:|---|
| `max_steps` | 20 | maximum environment actions, not model generations |
| `run_id` | generated | runtime/database identity |
| `headless` | true | Chromium mode |
| `keep_db` | true | retain episode DB artifact after close |
| `task_dir` | null | exclusive public/Oracle source; null uses default tasks |

## Playwright Execution Model

The main Agent loop is synchronous. Playwright uses `async_playwright` inside one persistent dedicated worker thread with its own asyncio event loop.

```text
main thread: reset / step / close
              ↓ synchronous call
worker thread: async browser/context/page operation
```

All Playwright objects remain in the worker thread. The browser lifecycle is:

```text
thread start
→ event loop start
→ Playwright/browser start
→ context/page per episode
→ page/context close
→ browser/Playwright close
→ event loop stop
→ worker join
```

Initialization and shutdown errors propagate. They are infrastructure failures, not policy rewards.

## reset(task_id) → Observation

1. Resolve the exclusive task source.
2. Validate `task_id` against the public task file.
3. Clean the previous page/context/service.
4. Create or reuse the isolated runtime SQLite database for the run.
5. Start a local FastAPI service on a free loopback port.
6. Start the Playwright worker/browser and create a fresh context/page.
7. Navigate to the task page and start the episode.
8. Resolve `episode_id` from the redirect.
9. Initialize `TrajectoryRecorder`.
10. Return the initial text Observation.

## step(action) → StepResult

1. Rebuild the current Observation for target validation.
2. Execute the typed action in the Playwright worker.
3. Record the action result in the trajectory.
4. Increment environment `step_index`.
5. Truncate when the environment-action budget is exhausted.
6. Build the next Observation.
7. On a procurement result page, call the deterministic Verifier.
8. On explicit `finish`, verify a persisted submission or return `premature_finish`.
9. Return `StepResult` with reward, terminal flags, and structured info.

A Schema-invalid model output never calls `env.step`; it consumes a model turn in the rollout runner but not an environment step.

## close()

1. Close page and context.
2. terminate the per-environment FastAPI process;
3. close browser and Playwright;
4. stop the worker event loop;
5. join the worker thread;
6. mark the environment closed.

`close()` is idempotent at the Environment level. A failed worker shutdown is surfaced because leaked browser/event-loop state can invalidate subsequent rollouts.

## Termination Semantics

| Condition | `terminated` | `truncated` | policy reward |
|---|---:|---:|---:|
| persisted submission + Verifier success | true | false | 1.0 |
| persisted submission + Verifier rejection | true | false | 0.0 |
| explicit finish without submission | true | false | 0.0 |
| environment step budget reached | true | true | 0.0 |
| browser/service/database exception | invalid rollout | n/a | null |

## Isolation and Security

- fresh browser context/page per environment episode;
- unique episode ID;
- per-run SQLite database;
- loopback-only Web service;
- no Oracle content in observations or prompts;
- no direct Page/SQLite/Verifier access for the policy;
- no arbitrary CSS/XPath/JavaScript action;
- target IDs must be present in the current Observation;
- explicit task source prevents default/development task merging.

## Current Limits

- formal probe is sequential within one process;
- no visual observation;
- no scroll action;
- fixed local website and bounded context;
- no multi-node or browser-farm rollout.
