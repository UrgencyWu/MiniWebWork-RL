# Ten-Minute Project Demo

This is a demonstration script for a mentor, reviewer, or interviewer.  It
shows the full product loop rather than presenting a training score in
isolation: deterministic task → browser observation → JSON action → verifier
reward → auditable policy update evidence.

## 0. Prepare (one minute)

```bash
conda activate miniwebwork
git rev-parse --short HEAD
bash scripts/run_quality_checks.sh -m "not gpu and not slurm"
```

Point out that the CPU gate validates the database seed, public/Oracle task
contract, Python code and tests.  The Oracle is not present in agent prompts.

## 1. Run the deterministic browser loop (three minutes)

Use two tasks that exercise both decision types:

```bash
python -m miniwebwork.baseline_runner \
  --tasks TASK-001,TASK-004 \
  --max-steps 20 \
  --output-dir artifacts/demo_rule
```

Then inspect the portable evidence:

```bash
cat artifacts/demo_rule/m1_2_baseline_metrics.json
sed -n '1,2p' artifacts/demo_rule/m1_2_baseline_trajectories.jsonl
```

Explain the two terminal outcomes:

- `TASK-001` selects a valid product and is verified against the private
  constraint/optimality oracle;
- `TASK-004` proves that the no-solution action is a real, independently
  verified decision rather than a shortcut reward.

## 2. Show the Agent contract (two minutes)

Open [`ARCHITECTURE_AND_CONTRACTS.md`](ARCHITECTURE_AND_CONTRACTS.md) and trace
one turn:

```text
text browser observation
  → Prompt Contract v2
  → Qwen JSON action
  → schema and environment checks
  → deterministic verifier terminal reward
```

Emphasize that each browser turn has a fresh prompt and that the RL objective
replays each stored prompt/completion pair, rather than treating a multi-turn
episode as one ordinary completion.

## 3. Show training integrity (two minutes)

For a completed strict artifact, print the fields that control whether it can
enter a gradient batch:

```bash
python - <<'PY'
import json
from pathlib import Path

artifact = Path("outputs/m2_3_mini/runs/<STRICT_RUN>/<ARTIFACT>.json")
data = json.loads(artifact.read_text())
print("policy:", data["policy"])
print("git:", data["git_sha"])
print("sampling:", data["temperature"], data["top_p"], data["top_k"])
print("runtime:", data["generation_runtime"])
print("metrics:", data["metrics"])
for group in data["groups"]:
    print(group["task_id"], group["valid_for_grpo_update"],
          group["max_raw_sampling_logprob_abs_diff"])
PY
```

State the non-negotiable rule: a trajectory with an infrastructure failure has
`reward=null` and never contributes a gradient.  A group needs complete token
evidence, mixed rewards and raw/sampling agreement before it is eligible.

## 4. Show the update audit and conclusion (two minutes)

For a completed single-batch smoke, open:

```text
outputs/m3_0b1_smoke/<POLICY>_<JOB_ID>/single_batch_smoke_report.json
```

Walk through these fields:

1. source artifact and adapter hashes;
2. pre-update old/current log-probability agreement;
3. finite, non-zero LoRA gradient and parameter delta;
4. saved adapter hash and successful reload forward.

Finish with the evaluation boundary: M2.2R is the fixed baseline,
`rollout_dev_feasible_v2` is a no-gradient regression gate, and the final
held-out evaluation is run only after policy and hyperparameters are frozen.
Any negative result is reported with the same artifacts and failure taxonomy;
the project is designed to make conclusions trustworthy, not merely positive.
