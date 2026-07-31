# Technical Review — 2026-07-31

## Approved path

```text
Deterministic Environment → Text-browser Agent Runtime → Canonical SFT
→ Qualified Multi-turn Rollout → Outcome-only GRPO-style Update
→ Frozen Final Evaluation
```

The project is not a general public-Web benchmark, visual browser Agent, one-shot GRPO example, process-reward project or SOTA claim.

## Frozen technical conclusions

- The repository already has an Agent runtime; the missing layer is online policy improvement.
- First reward contract: success 1, valid policy failure 0, infrastructure failure null.
- Each browser turn has a new Prompt, so optimization replays per-turn Prompt/action segments and aggregates terminal advantage at trajectory level.
- Current rollout identity includes `temperature`, `top_p`, `top_k`, raw-policy log-probabilities and sampling-distribution log-probabilities.
- The first strict distribution is `temperature=1.0, top_p=1.0, top_k=0`.
- Replay independently checks token alignment, finite probabilities and raw/sampling agreement; diagnostic artifacts cannot be promoted by a caller flag.
- A/B comparison pairs `(task_id, rollout_index)` and reports no-solution success, feasible success, false no-solution, discordant outcomes and multi-seed uncertainty.
- `rollout_dev_feasible_v2` is the only canonical feasible gate. It contains 12 deterministically generated select-product tasks, is excluded from gradients, and replaces deleted v1 data.

## Engineering contracts

- one exclusive task source per process;
- Oracle never enters prompts;
- episode/task/submission identities are bound;
- infrastructure failures never enter rewards or gradients;
- browser lifecycle is owned by one async worker;
- Slurm owns GPU visibility;
- rollout evidence stores exact Prompt/completion token IDs;
- concurrent runs use isolated output directories;
- only LoRA parameters train in the first smoke;
- streaming backward preserves equal trajectory weighting;
- the optimizer smoke uses `AdamW(weight_decay=0.0)` and must produce finite non-zero gradients, changed Adapter parameters, a reloadable checkpoint and finite reload forward.

## Readiness

| Capability | Status |
|---|---|
| deterministic environment and Agent runtime | pass |
| Canonical SFT | pass |
| historical readiness GPU probe | pass |
| schema-v3.3 rollout | implemented; cluster run pending |
| paired A/B analysis | implemented; multi-seed artifacts pending |
| feasible v2 gate | frozen; cluster run pending |
| strict replay and objective | implemented |
| one-batch optimizer smoke | implemented; GPU run pending |
| strict update-compatible group | not yet collected |
| final_test_v2 | not created |

```text
ROLLOUT_DEV_FEASIBLE_V2_FROZEN=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=false
```

## Execution order

1. pass CPU quality checks;
2. run no-solution A/B diagnostics across multiple seeds;
3. run feasible v2 regression and inspect false no-solution;
4. freeze the starting policy;
5. collect a strict mixed-reward group;
6. run the one-batch LoRA GPU smoke;
7. then begin a 5–10-update pilot.
