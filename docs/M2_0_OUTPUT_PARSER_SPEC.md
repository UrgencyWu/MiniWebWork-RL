# M2.0 Output Parser Spec

## Two-Level Parsing Strategy

### Level 1: Strict JSON Parse

**Rule**: Entire model output must be a single valid JSON object.

```python
raw = output.strip()
payload = json.loads(raw)       # raises JSONDecodeError if invalid
if isinstance(payload, dict):   # must be object, not array/string/number
    result.strict_json_success = True
```

**Rejects**: Markdown fences, surrounding text, multiple objects, JSON5, Python dicts.

### Level 2: Bounded Deterministic Fallback

Only attempted when Level 1 fails. Two strategies in order:

**Fenced Code Block**:
```
```json\n{...}\n```
```
Extract content between ```json and ``` markers, parse as JSON.

**Balanced Brace Extraction**:
Find first `{`, track brace depth, find matching `}`, parse as JSON.

### Prohibited Fixes

The parser NEVER:
- Calls another model to repair output
- Modifies field names
- Auto-guesses target values
- Deletes unknown fields
- Converts natural language to actions
- Replaces invalid actions with rule-based actions

## Schema Validation

After JSON extraction, validate against action schema:

1. `action` field: must be one of 7 supported types
2. Extra fields: `action/target/value/checked` only — unknown fields flagged
3. Target required: for click/fill/select/check/submit
4. Value required: for fill; must be non-empty, ≤ 500 chars
5. Value required: for select
6. Checked required: for check (boolean)

### Schema Errors Recorded

| Error | Trigger |
|---|---|
| unknown_action | Action not in {click,fill,select,check,back,submit,finish} |
| extra_fields | Unknown keys in action JSON |
| missing_target | Target action without target field |
| fill_missing_value | fill without value |
| select_missing_value | select without value |
| check_missing_checked | check without checked boolean |
| value_too_long | Value > 500 characters |

## ParseResult Fields

- `raw_output`: original model output text
- `strict_json_success`: Level 1 passed
- `fallback_used`: Level 2 was attempted
- `fallback_type`: "fenced" or "balanced_brace"
- `fallback_json_success`: Level 2 succeeded
- `parsed_payload`: extracted dict or None
- `schema_valid`: passes action schema check
- `schema_error`: specific schema error
- `errors`: list of all error codes

## Metrics Computed

| Metric | Formula |
|---|---|
| nonempty_generation_rate | % of turns with non-empty output |
| strict_json_success_rate | % of turns passing Level 1 |
| fallback_parse_rate | % of turns requiring Level 2 |
| effective_json_success_rate | % with any valid JSON extraction |
| action_schema_valid_rate | % passing schema validation |
| target_valid_rate | % producing executable AgentAction |

## M2.0 Results

Based on Job 951 (185 total generations):

| Metric | Value |
|---|---|
| Nonempty rate | 94.6% |
| Strict JSON rate | 94.6% |
| Fallback rate | 5.4% |
| Schema valid rate | 100.0% |
