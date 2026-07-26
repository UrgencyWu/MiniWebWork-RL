# M2.0 Model Agent Spec

## QwenTransformersBackend

Loads Qwen3.5-4B once, provides `generate(messages)` for chat completions.

### Configuration
| Parameter | Value |
|---|---|
| model_path | /data/share/model/Qwen3.5-4B |
| dtype | bfloat16 |
| device | cuda:0 (Slurm-assigned) |
| max_new_tokens | 128 |
| do_sample | False (greedy) |
| local_files_only | True |
| enable_thinking | False |

### Loading
```python
backend = QwenTransformersBackend(ModelConfig(model_path=...))
backend.load()  # 4.1s, 7.8 GB peak GPU memory
```

### Generation Pipeline
1. Render chat template as text: `tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)`
2. Tokenize: `tokenizer(rendered, return_tensors="pt")`
3. Generate: `model.generate(input_ids, max_new_tokens=128, do_sample=False, num_beams=1)`
4. Decode only new tokens (exclude prompt tokens)

### Generation Config
- `do_sample=False` — deterministic greedy decoding
- `max_new_tokens=128` — sufficient for JSON action output
- `use_cache=True` — KV cache enabled
- `num_beams=1` — no beam search
- No temperature/top_p (conflicts with greedy)

### Model Info
- Architecture: Qwen3_5ForConditionalGeneration
- Chat template SHA-256: `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`
- enable_thinking: supported and disabled

## QwenBrowserAgent

```python
agent = QwenBrowserAgent(backend, prompt_builder, output_parser)
agent.reset(task_id, instruction)
attempt = agent.act(observation)  # → ModelActionAttempt
agent.record_feedback(attempt, action_result, page_type)
```

### ModelActionAttempt
Contains full record of one model turn:
- raw_output, strict_json_success, fallback_used
- parsed_payload, schema_valid
- action (AgentAction or None), errors
- prompt_hash, input_tokens, output_tokens, latency_ms

## Agent Loop

```python
run_model_episode(task_id, env, agent, max_model_turns=20, max_env_steps=15)
```

### Flow
1. env.reset(task_id) → initial Observation
2. agent.reset(task_id, instruction)
3. Loop:
   a. agent.act(observation) → ModelActionAttempt
   b. If schema_valid and action exists → env.step(action) → StepResult
   c. If invalid → record feedback, increment consecutive_failures
   d. If 3 consecutive failures → terminate (model_output_failure_limit)
   e. If terminated/truncated → exit loop
4. Build model trajectory
5. Return result dict

### Model Turn vs Environment Step
- `model_turn_index`: incremented on every agent.act() call
- `environment_step_index`: incremented only when valid action executed via env.step()
- Invalid model outputs: increment model_turn but NOT environment_step

## Why Not Qwen-Agent / vLLM

- **Qwen-Agent**: Full agent framework with tool calling, not designed for granular trajectory recording and action-level metrics
- **vLLM**: Requires transformers<5 (version conflict), adds deployment complexity, not needed for single-thread evaluation
- **Direct Transformers**: Maximum transparency — every token, every prompt, every parse result is auditable
