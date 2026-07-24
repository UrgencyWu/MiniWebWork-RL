# M1.1 Verifier Spec

## Input
- task_id: Oracle task identifier
- episode_id: Episode to verify
- db_path: Optional SQLite path override

## Output
VerificationResult with:
- success: boolean
- task_id, episode_id, submission_id
- decision_type, selected_product_id
- expected_decision_type, expected_product_id
- constraints_satisfied, objective_satisfied
- failure_reasons: list of reason codes
- details: metadata dictionary
- verifier_version: "1.0.0"

## Re-computation Logic
1. Load oracle constraints from tasks_oracle.jsonl
2. Re-compute feasible products via SQL query with all constraints
3. Re-compute optimal product per objective function
4. Compare submission against re-computed ground truth
5. Check all constraint violations individually

## Failure Reasons
- missing_submission, invalid_episode
- wrong_decision_type, wrong_product
- price_constraint_failed, memory_constraint_failed, delivery_constraint_failed
- supplier_certification_failed, supplier_rating_failed, region_constraint_failed
- out_of_stock, warranty_constraint_failed
- objective_not_optimal, false_no_solution, expected_no_solution

## Determinism
- Same input always produces same output
- No random elements
- No LLM or model dependency

## Why Not LLM Judge
- Deterministic tasks have mathematically verifiable answers
- LLM judge adds latency, cost, and non-determinism
- Verifier serves as both evaluation and RL reward signal
