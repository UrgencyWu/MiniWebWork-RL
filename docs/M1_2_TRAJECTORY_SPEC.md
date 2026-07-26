# M1.2 Trajectory Spec

## Schema Version: 1.0

### Trajectory Structure

```json
{
  "trajectory_schema_version": "1.0",
  "run_id": "m1_2_slurm_944",
  "task_id": "TASK-001",
  "episode_id": "EP-...",
  "instruction": "...",
  "agent_name": "rule_based",
  "started_at": "2026-07-25T...",
  "ended_at": "2026-07-25T...",
  "max_steps": 15,
  "steps": [...],
  "final_reward": 1.0,
  "success": true,
  "termination_reason": "verified_submission",
  "total_steps": 6,
  "invalid_action_count": 0,
  "verification": {...}
}
```

### Step Record

Each step contains:
- `step_index`: 0-based step counter
- `observation`: full Observation at step start (see Observation Spec)
- `action`: the AgentAction dict that was executed
- `action_result`: ActionResult dict with success/error_code/message
- `reward`: 0.0 for non-terminal, 1.0/0.0 for terminal
- `terminated`: boolean
- `truncated`: boolean
- `elapsed_ms`: action execution time

### Verification

After terminal state, trajectory includes verifier output:
- `success`: verifier result
- `failure_reasons`: list of reason codes
- `selected_product_id` / `expected_product_id`
- `decision_type` / `expected_decision_type`

### Output Formats

- **Per-task JSON**: `{episode_id}.json` — single trajectory
- **JSONL**: `m1_2_baseline_trajectories.jsonl` — all trajectories, one per line

### Privacy Rules

Trajectories MUST NOT contain:
- Cookies or session tokens
- Model weights
- Oracle source files
- Database credentials
- Server internal paths
- Screenshot binary data

### Downstream Usage

Trajectories are designed for:
- **SFT data construction**: extract (observation, action) pairs
- **RL replay**: step-by-step replay of environment states
- **Failure analysis**: identify failure patterns across tasks
- **Comparison**: cross-agent metrics (rule vs model)
