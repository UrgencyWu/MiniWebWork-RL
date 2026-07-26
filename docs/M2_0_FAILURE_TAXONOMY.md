# M2.0 Failure Taxonomy

## Four-Layer Classification

### 1. Output Layer (模型输出层)
Model fails to produce valid JSON action.

| Tag | Description |
|---|---|
| empty_generation | Model returned empty string |
| non_json_output | Output is not valid JSON |
| multiple_json_objects | Output contains >1 JSON object |
| malformed_json | JSON parse error |
| schema_invalid | JSON is valid but fails action schema |
| unknown_action | Action field is not one of 7 supported types |
| extra_fields | JSON contains keys beyond {action, target, value, checked} |

### 2. Action Layer (动作层)
Valid action fails to execute on the page.

| Tag | Description |
|---|---|
| invalid_target | element_id not found in page elements |
| stale_target | Element exists in observation but not on page |
| incompatible_action | Action type not compatible with element role |
| disabled_element | Target element is disabled |
| value_too_long | fill value exceeds 500 chars |
| environment_action_error | Playwright execution error |

### 3. Planning Layer (规划层)
Agent makes poor navigation decisions.

| Tag | Description |
|---|---|
| repeated_action | Same action repeated 3+ consecutive turns |
| navigation_loop | Agent cycles between same pages |
| premature_finish | finish called without submission |
| no_submission | Episode ends without reaching procurement form |
| wrong_page | Agent on incorrect page for task stage |
| ignored_task_constraint | Task constraints not applied to filters |
| failed_to_apply_filter | Filter form interaction fails |
| failed_to_compare_candidates | Agent didn't compare multiple products |

### 4. Terminal Layer (终态层)
Submission made but verifier rejects it.

| Tag | Description |
|---|---|
| wrong_product | Selected product doesn't match expected |
| objective_not_optimal | Product is feasible but not optimal |
| false_no_solution | Agent claims no_solution but feasible products exist |
| expected_no_solution | Agent submits product when no feasible exists |
| constraint_failure | Selected product violates one or more constraints |
| max_model_turns | Reached model turn limit without reaching terminal |
| max_environment_steps | Reached env step limit without reaching terminal |
| browser_error | Browser encountered unrecoverable error |
| model_error | Environment/model interaction error |

## Primary Failure Determination

Priority order for assigning `primary_failure`:

1. `output_format_failure` — if non_json_output or schema_invalid
2. `element_grounding_failure` — if invalid_target or stale_target
3. `consecutive_output_failures` — if model_output_failure_limit
4. `no_submission_reached` — if max_model_turns or no submission
5. `premature_finish` — if finish without submission
6. `incorrect_product_selection` — if wrong_product or objective_not_optimal
7. `constraint_violation` — if constraint_failure

## M2.0 Results (Job 951)

| Primary Failure | Count |
|---|---|
| (success) | 5 |
| output_format_failure | 6 |
| element_grounding_failure | 4 |
