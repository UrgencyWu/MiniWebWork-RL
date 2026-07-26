# M2.0 Prompt Spec

## Prompt Version: browser_agent_v1

**File**: `prompts/browser_agent_v1.txt`
**SHA-256**: Computed at runtime, recorded in `artifacts/m2_0/m2_0_prompt_manifest.json`

## System Prompt

Role: web browser procurement agent. Core rules:
- Output ONLY a single JSON object per turn
- No markdown, no explanation, no thinking
- Only use element_id from current observation's elements list
- Exactly one action per turn
- Must navigate to product, select, fill form, and submit to complete procurement
- `finish` does NOT submit procurement — it ends the episode

## User Message Structure

Each turn's user message contains:

1. **Task context**: task_id, instruction
2. **Current page**: url, path, page_type, title, step
3. **Visible text**: page body text (max 8000 chars, truncated if exceeded)
4. **Interactive elements**: compact JSON array with element_id, role, name, testid, disabled, optional value/options
5. **Recent history**: last 5 turns — action, parse status, result, resulting page_type
6. **Last action result**: success/failure feedback from previous step
7. **Final instruction**: "Output exactly one JSON action. Only use element_id from the elements list above."

## Element Serialization

```json
[{
  "element_id": "search-query",
  "role": "textbox",
  "name": "搜索关键词",
  "testid": "search-query",
  "disabled": false
}]
```

Compact format — no CSS classes, no HTML attributes, no bounding boxes.

## Chat Template

- Tokenizer: Qwen2Tokenizer
- Template: `apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)`
- Why `enable_thinking=False`: prevents `<think>...</think>` prefix before JSON output
- Chat template SHA-256: `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`

## Token Limits

| Parameter | Default | Description |
|---|---|---|
| max_visible_text_chars | 8000 | Truncation before prompt construction |
| history_window | 5 | Number of recent turns to include |
| max_new_tokens | 128 | Generation limit |

## Oracle Leak Prevention

Prompt MUST NOT include:
- `expected_product_id` or `expected_decision_type`
- Oracle constraints JSON
- Verifier results
- Database contents
- Server file paths
- Cookie/authentication tokens
- Full HTML source
