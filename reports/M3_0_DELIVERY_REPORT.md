# MiniWebWork-RL M3.0 Delivery Report

## Provenance

| Item | Value |
|---|---|
| Training code Git SHA | `aa8cfb9073969deb42ba8c57199910aa7e7aba7a` |
| Source rollout artifact | `/home/wushaohua/data/MiniWebWork-RL/outputs/m2_3_mini/runs/A_rollout_dev_no_solution_v1_t1_p1_k0_s20260731_k8_j1080/single_probe_A_t1_p1_k0_20260731_224003.json` |
| Source adapter SHA-256 | `5f65d6c20eafa5509982833eeac71b189d4360d6d7dd9c927811b5e12b1f70de` |
| Updated adapter SHA-256 | `df66dab40a54abdbcb25d4b87ff195755b842868c608394dfb6e4451eb27cfec` |
| Sampling distribution | `{'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}` |
| Strict group | `M2_3_V0001`; valid=True |

## Training correctness

- Valid trajectories: 8/8; infrastructure errors: 0.
- Raw/sampling max difference: `0.006612047553062439` (tolerance `0.05`).
- Pre-update old/current max difference: `0.0`.
- LoRA tensors with non-zero gradient: `256` / `256`; changed tensors: `256`.
- Maximum adapter parameter delta: `1.000240445137024e-06`.

## Frozen paired evaluation

| Metric | M2.2R baseline | Updated policy |
|---|---:|---:|
| Comparable success | 14.58% | 14.58% |
| Infrastructure errors | 0 | 0 |
| Feasible success | 63.64% | 63.64% |
| False no-solution count | 0 | 0 |

- Paired success delta (updated − baseline): 0.00%.
- Task-bootstrap 95% CI: [-3.12%, 3.12%].
- Exact McNemar p-value: `1.0`.
- Conclusion: **no improvement is supported; preserve this as a negative or neutral result**.

## Failure and infrastructure taxonomy

Baseline termination reasons: `{'model_output_failure_limit': 74, 'verified_submission': 22}`

Updated-policy termination reasons: `{'model_output_failure_limit': 74, 'verified_submission': 22}`

Infrastructure-invalid trajectories are excluded from paired success denominators
and are reported above; they never enter training rewards or gradients.
