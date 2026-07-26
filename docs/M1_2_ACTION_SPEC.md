# M1.2 Action Spec

## Schema Version: 1.0

### Supported Actions (7)

#### click
```json
{"action": "click", "target": "e5"}
```
- Target role: link, button, submit elements
- Triggers navigation if target is a link

#### fill
```json
{"action": "fill", "target": "e2", "value": "GPU"}
```
- Target role: textbox, searchbox, textarea
- Replaces current value
- Max value length: 500 chars

#### select
```json
{"action": "select", "target": "e4", "value": "GPU"}
```
- Target role: combobox (select elements)
- Value must match an option in the element's options list

#### check
```json
{"action": "check", "target": "e7", "checked": true}
```
- Target role: checkbox
- `checked`: true = check, false = uncheck

#### back
```json
{"action": "back"}
```
- No target required
- Browser history back navigation

#### submit
```json
{"action": "submit", "target": "e10"}
```
- Target: submit button or form button
- Triggers form submission

#### finish
```json
{"action": "finish"}
```
- No target required
- Agent declares task complete
- Environment checks if submission exists → verifier or premature_finish

### Role-Action Compatibility

| Element Role | Allowed Actions |
|---|---|
| link | click |
| button | click |
| textbox | fill |
| searchbox | fill |
| checkbox | check |
| combobox | select |
| spinbutton | fill |
| textarea | fill |

### Validation Pipeline

1. **Action type check**: must be one of 7 supported types
2. **Target existence**: `finish`/`back` skip; others require target in current observation
3. **Disabled check**: disabled elements rejected
4. **Compatibility**: action must match element role
5. **Value requirements**: fill requires non-empty value; value ≤ 500 chars
6. **Extra fields**: unknown fields in action JSON flagged as errors

### Error Codes (12)

| Code | Description |
|---|---|
| invalid_action_type | Unknown action name |
| malformed_action | Action JSON structure invalid |
| invalid_target | Target element_id not in observation |
| stale_target | Element exists in obs but not on page |
| incompatible_action | Action not valid for element role |
| disabled_element | Target element is disabled |
| value_required | fill action missing value |
| value_too_long | Value exceeds 500 character limit |
| navigation_blocked | External URL or unauthorized path |
| browser_error | Playwright execution error |
| environment_closed | Action on closed environment |
| episode_finished | Action after terminal state |

### Security Boundaries

Actions MUST NOT accept:
- `css_selector` — arbitrary CSS selectors
- `xpath` — XPath expressions
- `script` / `javascript` — JavaScript execution
- `url` — arbitrary navigation
- `file` — local file access
- Extra unknown fields beyond `action`, `target`, `value`, `checked`
