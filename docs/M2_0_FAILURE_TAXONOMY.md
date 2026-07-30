# Failure Taxonomy

> Originated in M2.0 and revised for M2.3/M3.0. The primary distinction is whether a failure is attributable to the policy or to the execution infrastructure.

## 1. Policy Failures

Policy failures are valid learning outcomes and receive terminal reward `0.0`.

### Output and Schema

| Tag | Meaning |
|---|---|
| `empty_generation` | model emitted no text |
| `non_json_output` | output cannot be parsed as JSON |
| `schema_invalid` | JSON does not satisfy the action schema |
| `unknown_action` | unsupported action name |
| `missing_target` | target-required action lacks a target |
| `extra_fields` | action contains unsupported fields |
| `model_output_failure_limit` | bounded consecutive invalid outputs reached |

Schema-invalid output is never replaced by a synthetic `finish` action.

### Grounding and Execution Choice

| Tag | Meaning |
|---|---|
| `invalid_target` | model selected an ID absent from the observation |
| `stale_target` | observed target no longer exists on the page |
| `incompatible_action` | action type is invalid for the selected element |
| `disabled_element` | model selected a disabled control |
| `repeated_action` | same ineffective action repeats |
| `navigation_loop` | policy cycles between states |

A browser action returning a deterministic validation failure is a policy failure. A Playwright exception is infrastructure.

### Planning and Termination

| Tag | Meaning |
|---|---|
| `premature_finish` | policy explicitly finishes without a persisted submission |
| `max_model_turns` | policy fails to reach a terminal state within the turn budget |
| `max_environment_steps` | valid actions consume the environment step budget |
| `ignored_task_constraint` | required filter/constraint is not applied |
| `failed_to_compare_candidates` | policy selects before satisfying the objective |
| `missed_no_solution` | feasible-set empty but policy does not submit no-solution |

### Verifier Rejection

| Tag | Meaning |
|---|---|
| `wrong_product` | selected product differs from the deterministic optimum |
| `objective_not_optimal` | product is feasible but not optimal |
| `false_no_solution` | no-solution submitted while feasible products exist |
| `expected_no_solution` | product submitted when no product is feasible |
| constraint-specific codes | price, memory, delivery, region, certification, rating, stock, or warranty violation |
| `missing_submission` | policy reaches a terminal-like state without a persisted submission, after the environment contract has been verified |

## 2. Infrastructure Failures

Infrastructure failures invalidate a rollout and use:

```text
rollout_valid = false
failure_origin = infrastructure
reward = null
```

They never enter group reward statistics or policy updates.

| Tag | Meaning |
|---|---|
| `model_load_error` | base model or adapter cannot be loaded |
| `model_backend_error` | CUDA/OOM/device assertion or missing token/logprob evidence |
| `environment_start_error` | web service, database, browser, or task setup fails |
| `environment_step_error` | Playwright/database/service exception during `env.step` |
| `environment_cleanup_error` | browser/thread/service lifecycle cannot close cleanly |
| `task_source_error` | public/Oracle source missing, duplicated, or mismatched |
| `artifact_contract_error` | rollout evidence violates its schema/invariants |

## 3. Metric Attribution

The formal denominator rules are:

```text
Task success denominator
= valid policy trajectories

Schema Valid denominator
= all model turns in valid policy trajectories

Environment Action Success denominator
= environment actions attempted in valid policy trajectories

Reward variance
= rewards from valid policy trajectories in the same task group
```

Infrastructure failures are reported separately.

## 4. Historical M2.0 Result

The original Job 951 primary counts were:

| Primary Failure | Count |
|---|---:|
| success | 5 |
| output-format failure | 6 |
| element-grounding failure | 4 |

These values are retained as historical evidence. They were produced before the current policy/infrastructure and action-level metric contracts were frozen, so they must not be directly merged with M2.3/M3.0 statistics.
