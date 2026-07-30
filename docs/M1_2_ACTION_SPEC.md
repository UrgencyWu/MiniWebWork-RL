# Action Contract

> Introduced in M1.2. Current schema version: **1.1**.

## Supported Actions

| Action | Example | Target requirement |
|---|---|---|
| `click` | `{"action":"click","target":"product-link"}` | link or button |
| `fill` | `{"action":"fill","target":"max-price","value":"15000"}` | textbox, searchbox, spinbutton, or textarea |
| `select` | `{"action":"select","target":"category","value":"GPU"}` | combobox; value must be an available option |
| `check` | `{"action":"check","target":"in-stock","checked":true}` | checkbox |
| `back` | `{"action":"back"}` | no target |
| `submit` | `{"action":"submit","target":"submit-procurement"}` | button |
| `finish` | `{"action":"finish"}` | no target |

`submit` and `click` may both target a visible button. They remain separate semantic actions so trajectories can distinguish ordinary interaction from final form submission.

## Role–Action Compatibility

| Element role | Allowed actions |
|---|---|
| link | click |
| button | click, submit |
| textbox | fill |
| searchbox | fill |
| spinbutton | fill |
| textarea | fill |
| checkbox | check |
| combobox | select |

## Validation Order

1. action type belongs to the seven-action vocabulary;
2. target-required actions contain a non-empty target;
3. target ID appears in the current Observation;
4. target is enabled;
5. action is compatible with the observed element role;
6. `fill` has a non-empty value no longer than 500 characters;
7. `select` value is present in the observed option list;
8. unknown JSON fields are rejected by the output parser.

A Schema-valid action can still fail deterministic environment validation, for example because the observed target became stale. Such failures are policy outcomes. A Playwright exception is an infrastructure failure and invalidates the rollout.

## Termination Semantics

### `finish`

`finish` declares that no more browser interaction is needed. The environment then checks persistence:

```text
submission exists
→ deterministic Verifier

submission absent
→ premature_finish, reward 0
```

The runner never fabricates `finish` when model output is invalid.

### `submit`

`submit` clicks a visible form button. A successful click is not itself task success. The resulting persisted submission must still pass the deterministic Verifier.

## Security Boundary

The policy cannot emit:

- arbitrary CSS/XPath selectors;
- JavaScript;
- arbitrary URLs;
- local file paths;
- shell commands;
- hidden Oracle fields.

Only an `element_id` included in the current Observation may be targeted. The environment resolves that ID to the current DOM inside the Playwright worker thread.

## Error Attribution

| Class | Examples | Reward treatment |
|---|---|---|
| policy output | non-JSON, Schema invalid, missing target | 0 if episode fails |
| policy grounding | invalid/stale target, incompatible action | 0 if episode fails |
| policy planning | premature finish, wrong submission, truncation | 0 |
| infrastructure | browser/service/database/CUDA exception | null; excluded from RL |
