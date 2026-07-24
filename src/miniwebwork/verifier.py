"""Deterministic procurement verifier — no LLM dependency."""

from typing import Optional

from .db import get_connection
from .models import (
    FAILURE_EXPECTED_NO_SOLUTION,
    FAILURE_FALSE_NO_SOLUTION,
    FAILURE_INVALID_EPISODE,
    FAILURE_MISSING_SUBMISSION,
    FAILURE_OBJECTIVE_NOT_OPTIMAL,
    FAILURE_OUT_OF_STOCK,
    FAILURE_PRICE_CONSTRAINT_FAILED,
    FAILURE_DELIVERY_CONSTRAINT_FAILED,
    FAILURE_MEMORY_CONSTRAINT_FAILED,
    FAILURE_REGION_CONSTRAINT_FAILED,
    FAILURE_SUPPLIER_CERTIFICATION_FAILED,
    FAILURE_SUPPLIER_RATING_FAILED,
    FAILURE_WARRANTY_CONSTRAINT_FAILED,
    FAILURE_WRONG_DECISION_TYPE,
    FAILURE_WRONG_PRODUCT,
    DECISION_SELECT_PRODUCT,
    DECISION_NO_SOLUTION,
    OBJECTIVE_NO_FEASIBLE_PRODUCT,
    VerificationResult,
)
from .tasks import compute_feasible_products, compute_optimal_product, get_oracle, parse_constraints


def verify_episode(task_id: str, episode_id: str, db_path: Optional[str] = None) -> VerificationResult:
    """Verify a submitted episode against its task Oracle.

    Returns VerificationResult with detailed failure reasons.
    """
    result = VerificationResult(
        success=False,
        task_id=task_id,
        episode_id=episode_id,
    )

    oracle = get_oracle(task_id)
    if oracle is None:
        result.failure_reasons.append(FAILURE_INVALID_EPISODE)
        result.details["error"] = f"Unknown task: {task_id}"
        return result

    result.expected_decision_type = oracle.get("expected_decision_type", "")
    result.expected_product_id = oracle.get("expected_product_id", "")

    conn = get_connection(db_path)

    # Verify episode exists
    episode = conn.execute(
        "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    if episode is None:
        result.failure_reasons.append(FAILURE_INVALID_EPISODE)
        result.details["error"] = f"Episode {episode_id} not found"
        return result

    # Get submission
    submission = conn.execute(
        "SELECT * FROM procurement_submissions WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()

    if submission is None:
        result.failure_reasons.append(FAILURE_MISSING_SUBMISSION)
        result.details["error"] = "No submission found for this episode"
        return result

    result.submission_id = submission["submission_id"]
    result.decision_type = submission["decision_type"]
    result.selected_product_id = submission["product_id"] or ""

    # Check decision type matches expected
    expected_decision = oracle.get("expected_decision_type", "")
    if result.decision_type != expected_decision:
        result.failure_reasons.append(FAILURE_WRONG_DECISION_TYPE)
        result.details["expected"] = expected_decision
        result.details["actual"] = result.decision_type

    # Case: expected no_solution
    if expected_decision == DECISION_NO_SOLUTION:
        if result.decision_type != DECISION_NO_SOLUTION:
            result.failure_reasons.append(FAILURE_EXPECTED_NO_SOLUTION)
            return result

        # Verify that no_solution is truly correct
        feasible = compute_feasible_products(oracle, conn)
        if len(feasible) > 0:
            result.failure_reasons.append(FAILURE_FALSE_NO_SOLUTION)
            result.details["feasible_products"] = [r["product_id"] for r in feasible]
            return result

        # Correctly identified no_solution
        result.success = True
        result.constraints_satisfied = True
        result.objective_satisfied = True
        return result

    # Case: expected select_product
    if result.decision_type == DECISION_NO_SOLUTION:
        # Agent said no_solution but there ARE feasible products
        feasible = compute_feasible_products(oracle, conn)
        if len(feasible) > 0:
            result.failure_reasons.append(FAILURE_FALSE_NO_SOLUTION)
            optimal = compute_optimal_product(oracle, feasible)
            result.details["missed_product"] = optimal["product_id"] if optimal else "unknown"
        return result

    # Both expect and have select_product — verify the selected product
    product_id = result.selected_product_id
    product = conn.execute(
        """SELECT p.*, s.rating as supplier_rating, s.certified, s.region
           FROM products p JOIN suppliers s ON p.supplier_id = s.supplier_id
           WHERE p.product_id = ?""",
        (product_id,),
    ).fetchone()

    if product is None:
        result.failure_reasons.append(FAILURE_WRONG_PRODUCT)
        result.details["error"] = f"Product {product_id} not found"
        return result

    # Check all constraints against the selected product
    constraints = parse_constraints(oracle.get("constraints", {}))
    c = constraints
    all_constraints_ok = True

    if c.category and product["category"] != c.category:
        result.failure_reasons.append(FAILURE_WRONG_PRODUCT)
        all_constraints_ok = False

    if c.keyword:
        kw = c.keyword.lower()
        model = (product["model_number"] or "").lower()
        name = (product["name"] or "").lower()
        desc = (product["description"] or "").lower()
        if kw not in model and kw not in name and kw not in desc:
            result.failure_reasons.append(FAILURE_WRONG_PRODUCT)
            all_constraints_ok = False

    if c.max_price is not None and product["price"] > c.max_price:
        result.failure_reasons.append(FAILURE_PRICE_CONSTRAINT_FAILED)
        all_constraints_ok = False

    if c.min_memory_gb is not None and (product["memory_gb"] or 0) < c.min_memory_gb:
        result.failure_reasons.append(FAILURE_MEMORY_CONSTRAINT_FAILED)
        all_constraints_ok = False

    if c.max_delivery_days is not None and product["delivery_days"] > c.max_delivery_days:
        result.failure_reasons.append(FAILURE_DELIVERY_CONSTRAINT_FAILED)
        all_constraints_ok = False

    if c.min_supplier_rating is not None and product["supplier_rating"] < c.min_supplier_rating:
        result.failure_reasons.append(FAILURE_SUPPLIER_RATING_FAILED)
        all_constraints_ok = False

    if c.certified_only is not None:
        expected_certified = 1 if c.certified_only else 0
        if product["certified"] != expected_certified:
            result.failure_reasons.append(FAILURE_SUPPLIER_CERTIFICATION_FAILED)
            all_constraints_ok = False

    if c.supplier_region and product["region"] != c.supplier_region:
        result.failure_reasons.append(FAILURE_REGION_CONSTRAINT_FAILED)
        all_constraints_ok = False

    if c.in_stock_only and product["stock"] <= 0:
        result.failure_reasons.append(FAILURE_OUT_OF_STOCK)
        all_constraints_ok = False

    if c.min_warranty_months is not None and product["warranty_months"] < c.min_warranty_months:
        result.failure_reasons.append(FAILURE_WARRANTY_CONSTRAINT_FAILED)
        all_constraints_ok = False

    result.constraints_satisfied = all_constraints_ok

    # Check optimality
    feasible = compute_feasible_products(oracle, conn)
    optimal = compute_optimal_product(oracle, feasible)
    result.objective_satisfied = (optimal is not None and optimal["product_id"] == product_id)

    if not result.objective_satisfied and optimal is not None:
        result.failure_reasons.append(FAILURE_OBJECTIVE_NOT_OPTIMAL)
        result.details["optimal_product"] = optimal["product_id"]
        result.details["optimal_price"] = optimal["price"]
        result.details["optimal_rating"] = optimal["rating"]

    result.success = all_constraints_ok and result.objective_satisfied

    return result
