# M2.3-mini Canonical Readiness Review

Date: 2026-07-31

## Scope

This report freezes the reviewed M2.3-mini canonical rollout readiness result and its interpretation boundary. The raw GPU rollout JSON artifacts remain cluster outputs; this repository records the reported aggregate values, code identity, and next-stage decision.

## Infrastructure fix

Commit:

```text
bd1e5b396fc73d4fb87aa641a45fae2c79975a47
```

Root cause:

```text
PlaywrightThreadManager.start(self, headless: bool = True)
```

was replaced by `_checked_start(self)` in `agent_env/__init__.py`. Calling `start(headless=...)` therefore raised `TypeError` and invalidated every probe as an infrastructure failure.

The guard now preserves and forwards the `headless` keyword. A regression test is included in `tests/test_agent_env_start_guard.py`.

## Reported readiness result

Reported sampling parameters:

```text
temperature = 0.2
top_p = 0.9
top_k = implicit model/Transformers generation default
```

The legacy artifact did not explicitly freeze `top_k`. This does not invalidate the infrastructure, closed-loop capability, or logprob-coverage conclusions, but it means the artifact is not a fully identified behavior distribution and must not be used as an optimizer batch.

Reported gates:

| Gate | Result |
|---|---:|
| complete | true |
| infrastructure errors | 0 |
| raw policy logprob coverage | 1.0 |
| sampling logprob coverage | 1.0 |
| A_M2.2R no-solution successes | 58 |
| B_M2.3-mini no-solution successes | 43 |
| valid_for_grpo_update | false |

`valid_for_grpo_update=false` is expected. A truncated readiness distribution demonstrates closed-loop exploration and mixed reward but is not the strict first optimizer distribution.

## Accepted conclusions

```text
M2_3_MINI_CANONICAL_PROBE_PASS=true
NO_SOLUTION_ROLLOUT_CAPABILITY_CONFIRMED=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=false
```

Both policies can produce valid no-solution successes with complete rollout evidence and no infrastructure failures. The original zero-success/zero-variance bootstrap barrier has been removed.

## Conclusions not supported yet

The single aggregate `58 vs 43` result does not establish that Policy A is superior, that Policy B is inferior, or that the difference is merely sampling variance.

Required analysis:

1. pair trajectories by `(task_id, rollout_index)`;
2. report A-only and B-only successes;
3. inspect per-task success differences and termination modes;
4. repeat on additional master seeds;
5. add feasible tasks to measure false no-solution and general-task regression.

The repository provides `scripts/analyze_probe_ab.py` for paired artifact analysis. New schema-v3.3 collections explicitly identify `temperature`, `top_p`, and `top_k`.

## Next stage

### Diagnostic replication

Repeat A/B with a fully explicit diagnostic distribution:

```text
temperature = 0.2
top_p = 0.9
top_k = 0
```

Use identical task source, K, sampling settings, and multiple master seeds.

### Strict collection

Collect the first update-compatible distribution with:

```text
temperature = 1.0
top_p = 1.0
top_k = 0
```

Strict eligibility additionally requires that raw-policy and sampling-distribution token log-probabilities agree within the declared numerical tolerance. This detects hidden or unexpected generation processors.

A group is eligible only when it also contains non-zero reward variance and complete token/logprob evidence.

### Single-batch optimizer smoke

After at least one strict group is available:

1. reconstruct each real per-turn prompt and completion;
2. verify current and stored old-policy logprobs match before update;
3. compute group-relative trajectory advantages;
4. execute one LoRA-only backward and optimizer step;
5. verify finite non-zero gradients and changed adapter weights;
6. save, reload, and run a small closed-loop check.

No performance improvement claim is made at the single-batch smoke stage.
