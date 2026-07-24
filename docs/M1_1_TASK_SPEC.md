# M1.1 Task Spec

## Public / Oracle Separation

- `tasks_public.jsonl`: Agent-visible — task_id, instruction, start_path, task_type
- `tasks_oracle.jsonl`: Verifier-only — task_id, constraints, objective, expected_product_id

## Task Types
- exact_product (3 tasks)
- cheapest_feasible (8 tasks)
- highest_rating_supplier (2 tasks)
- no_feasible_product (2 tasks)

## Constraint Fields
category, keyword, max_price, min_memory_gb, max_delivery_days, min_supplier_rating, certified_only, supplier_region, in_stock_only, min_warranty_months

## Objective Functions
- exact_product: single product matching keyword
- cheapest_feasible: minimum price among feasible products
- highest_rating_supplier: maximum supplier rating, tie-break by minimum price
- no_feasible_product: verified empty feasible set

## Unique Answer Rules
- All 15 tasks have unique correct answers
- no_solution tasks verified by exhaustive constraint computation
- Optimal products verified by re-computation, not manual assignment

## Data Leak Prevention
- Public pages never expose oracle content
- expected_product_id not in HTML responses
- Task pages show only natural language instruction
