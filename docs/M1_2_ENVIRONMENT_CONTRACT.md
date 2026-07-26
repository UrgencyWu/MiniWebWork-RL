# M1.2 Environment Contract

## Interface: ProcurementBrowserEnv

### Constructor

```python
env = ProcurementBrowserEnv(max_steps=15, run_id="...", headless=True)
```

| Parameter | Default | Description |
|---|---|---|
| max_steps | 15 | Max environment steps per episode |
| run_id | auto | Unique run identifier |
| headless | True | Chromium headless mode |

### reset(task_id) → Observation

1. Validate task_id against public task set
2. Setup isolated SQLite database (data/runtime/m1_2/{run_id}.db)
3. Start local FastAPI web service on auto-allocated port
4. Start/reuse Playwright Chromium browser with new context
5. Navigate to task page, click start-task-button
6. Extract episode_id from redirect URL
7. Initialize TrajectoryRecorder
8. Build and return initial Observation

### step(action) → StepResult

1. Build current Observation for action validation
2. Execute action on Playwright page via `execute_action()`
3. Record step in trajectory
4. Increment step_index
5. Check max_steps → truncated if exceeded
6. Build new Observation
7. Check termination: `page_type == "procurement_result"` → call verifier
8. Handle `finish` action: check submission exists → verifier or premature_finish
9. Return StepResult(observation, reward, terminated, truncated, info)

### close()

1. Close Playwright page and context (keep browser for reuse across resets)
2. Terminate FastAPI web service
3. Close database connection
4. Idempotent — safe to call multiple times

## Termination Semantics

| Condition | terminated | truncated | reward |
|---|---|---|---|
| Submission created + verifier success | true | false | 1.0 |
| Submission created + verifier failure | true | false | 0.0 |
| max_steps reached | true | true | 0.0 |
| finish without submission | true | false | 0.0 (premature_finish) |
| finish with submission | true | false | verifier result |

## Episode Isolation

Each `reset()`:
- Creates new Browser Context (isolated cookies/storage)
- Uses new episode_id (unique per task)
- Previously created data persists in the shared database
- Web service runs on same port across resets (reused)

## Security Restrictions

- Agent cannot access Playwright page object
- Agent cannot read Oracle files
- Agent cannot query SQLite directly
- Agent cannot call verifier
- Actions limited to predefined schema (no CSS/XPath/JS)
- element_id must exist in current observation

## Current Limitations (M1.2)

- Single environment only (no parallel rollouts)
- All tasks share one database within a run
- No scroll action
- Fixed viewport
