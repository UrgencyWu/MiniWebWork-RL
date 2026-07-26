# M1.2 Command Log

## Phase: Agent Environment Contract & Rule-based Baseline

### Module Creation
```bash
mkdir -p src/miniwebwork/agent_env src/miniwebwork/agents
```

### Key Modules
- `agent_env/schemas.py` — Observation, AgentAction, StepResult, ElementDescriptor
- `agent_env/errors.py` — EnvironmentClosedError, EpisodeFinishedError, InvalidActionError
- `agent_env/observation.py` — Batch JS element extraction, page classification
- `agent_env/actions.py` — validate_action(), execute_action()
- `agent_env/trajectory.py` — TrajectoryRecorder, compute_metrics()
- `agent_env/environment.py` — ProcurementBrowserEnv (reset/step/close)
- `agents/rule_based.py` — RuleBasedProcurementAgent
- `baseline_runner.py` — CLI: --agent rule --tasks all --max-steps 15

### Tests
```bash
pytest tests/test_agent_env.py -q    # 29 tests (schemas, validation, security, trajectory, rule agent)
pytest tests/ -q                       # 90 passed total
```

### Slurm E2E Iterations
```bash
Job 932: TIMEOUT (500s/task)          → optimized observation extraction (batch JS)
Job 933: FAILED (name 'page' error)   → fixed self._page reference
Job 934-937: FAILED (0 steps)         → debugged element extraction, filter form, select-product
Job 938-941: FAILED (0% success)      → debugged episode_id mismatch, URL classification
Job 942: FAILED (all verified, 0%)    → verifier found missing_submission (double episode)
Job 943: FAILED (syntax error)        → sed broke environment.py
Job 944: COMPLETED (2/15, 13.3%)      → baseline established
```

### Key Fixes Applied
1. Observation: removed viewport filter → elements below fold now visible
2. FastAPI: changed query param types from `Optional[float/int]` to `str` → empty form fields no longer crash
3. Episode: extract episode_id from browser URL after redirect, not from fallback DB creation
4. product_detail template: always show select-product button (remove episode_id condition)
5. URL classifier: handle `/procurement/submit` and `/procurement/result/` paths
