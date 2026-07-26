# M1.2 Observation Spec

## Schema Version: 1.0

### Observation Structure

```json
{
  "schema_version": "1.0",
  "task_id": "TASK-001",
  "episode_id": "EP-...",
  "instruction": "...",
  "step_index": 0,
  "url": "http://127.0.0.1:{port}/tasks/TASK-001",
  "path": "/tasks/TASK-001",
  "page_type": "task",
  "title": "Task TASK-001",
  "visible_text": "...",
  "text_truncated": false,
  "elements": [...],
  "last_action_result": null,
  "terminal": false
}
```

### Page Types (10)

| Type | URL Pattern | Description |
|---|---|---|
| home | / | Task listing page |
| task | /tasks/{id} | Single task description |
| products | /products? | Product listing with filters |
| product_detail | /products/{id} | Single product details |
| supplier_detail | /suppliers/{id} | Supplier details + products |
| procurement_form | /procurement/new? | Procurement submission form |
| procurement_result | /procurement/result/{id} | Submission confirmation |
| smoke | /smoke | M1.0 smoke test page |
| error | /health | Health check (not for agent) |
| unknown | * | Unrecognized page |

### Element Extraction

**Source**: Batch JavaScript evaluation — single `page.evaluate()` call queries all interactive DOM elements.

**Selector**: `a, button, input:not([type="hidden"]), select, textarea, [role="button"], [role="link"], [role="textbox"], [role="searchbox"], [role="checkbox"], [role="combobox"], [role="spinbutton"]`

**Filtering**:
- Hidden elements (0×0 rect): excluded
- Disabled elements: included with `disabled: true`

**Roles**: link, button, textbox, searchbox, checkbox, combobox, spinbutton, textarea

### Element Descriptor

```json
{
  "element_id": "search-query",
  "role": "textbox",
  "tag": "input",
  "name": "搜索关键词",
  "text": "",
  "value": "",
  "input_type": "text",
  "testid": "search-query",
  "options": [],
  "disabled": false
}
```

### element_id Rules

- Derived from `data-testid` > `id` > `name` > `{tag}_{index}`
- Unique within a single observation
- NOT guaranteed to persist across page navigations
- Action target must reference an `element_id` from the current observation

### visible_text

- Extracted from `body.inner_text()`
- Maximum: 8,000 characters
- `text_truncated: true` when limit exceeded
- Contains only user-visible text (no scripts, styles, hidden elements)

### Oracle Leak Prevention

Observation MUST NOT contain:
- `expected_product_id`
- `expected_decision_type`
- Oracle constraints JSON
- `tasks_oracle.jsonl` content
- Database contents
- Server file paths
- Verifier expected answers
