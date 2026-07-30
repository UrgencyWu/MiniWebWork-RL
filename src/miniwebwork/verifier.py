"""Deterministic procurement verifier with no LLM dependency."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Optional

from .db import get_connection
from .models import (
    DECISION_NO_SOLUTION,
    FAILURE_DELIVERY_CONSTRAINT_FAILED,
    FAILURE_EXPECTED_NO_SOLUTION,
    FAILURE_FALSE_NO_SOLUTION,
    FAILURE_INVALID_EPISODE,
    FAILURE_MEMORY_CONSTRAINT_FAILED,
    FAILURE_MISSING_SUBMISSION,
    FAILURE_OBJECTIVE_NOT_OPTIMAL,
    FAILURE_OUT_OF_STOCK,
    FAILURE_PRICE_CONSTRAINT_FAILED,
    FAILURE_REGION_CONSTRAINT_FAILED,
    FAILURE_SUPPLIER_CERTIFICATION_FAILED,
    FAILURE_SUPPLIER_RATING_FAILED,
    FAILURE_WARRANTY_CONSTRAINT_FAILED,
    FAILURE_WRONG_DECISION_TYPE,
    FAILURE_WRONG_PRODUCT,
    VerificationResult,
)
from .tasks import compute_feasible_products, compute_optimal_product, get_oracle, parse_constraints


def verify_episode(
    task_id: str,
    episode_id: str,
    db_path: Optional[str] = None,
    task_dir: str | Path | None = None,
) -> VerificationResult:
    """Verify one persisted submission against the frozen Oracle.

    ``task_dir`` selects the same exclusive task source used by the browser
    environment.  Connections are always closed, including early-return
    failure paths.
    """
    result = VerificationResult(success=False, task_id=task_id, episode_id=episode_id)

    oracle = get_oracle(task_id, task_dir=task_dir)
    if oracle is None:
        result.failure_reasons.append(FAILURE_INVALID_EPISODE)
        result.details["error"] = f"Unknown task: {task_id}"
        return result

    result.expected_decision_type = oracle.get("expected_decision_type", "")
    result.expected_product_id = oracle.get("expected_product_id", "")

    with closing(get_connection(db_path)) as conn:
        episode = conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if episode is None:
            result.failure_reasons.append(FAILURE_INVALID_EPISODE)
            result.details["error"] = f"Episode {episode_id} not found"
            return result

        # Prevent a valid submission from one task being verified against a
        # different task Oracle.
        if "task_id" in episode.keys() and episode["task_id"] != task_id:
            result.failure_reasons.append(FAILURE_INVALID_EPISODE)
            result.details["error"] = (
                f"Episode {episode_id} belongs to {episode['task_id']}, not {task_id}"
            )
            return result

        submission = conn.execute(
            "SELECT * FROM procurement_submissions WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if submission is None:
            result.failure_reasons.append(FAILURE_MISSING_SUBMISSION)
            result.details["error"] = "No submission found for this episode"
            return result

        result.submission_id = submission["submission_id"]
        result.decision_type = submission["decision_type"]
        result.selected_product_id = submission["product_id"] or ""

        expected_decision = oracle.get("expected_decision_type", "")
        if result.decision_type != expected_decision:
            result.failure_reasons.append(FAILURE_WRONG_DECISION_TYPE)
            result.details["expected_decision_type"] = expected_decision
            result.details["actual_decision_type"] = result.decision_type

        if expected_decision == DECISION_NO_SOLUTION:
            if result.decision_type != DECISION_NO_SOLUTION:
                result.failure_reasons.append(FAILURE_EXPECTED_NO_SOLUTION)
                return result

            feasible = compute_feasible_products(oracle, conn)
            if feasible:
                result.failure_reasons.append(FAILURE_FALSE_NO_SOLUTION)
                result.details["feasible_products"] = [row["product_id"] for row in feasible]
                return result

            result.success = True
            result.constraints_satisfied = True
            result.objective_satisfied = True
            return result

        if result.decision_type == DECISION_NO_SOLUTION:
            feasible = compute_feasible_products(oracle, conn)
            if feasible:
                result.failure_reasons.append(FAILURE_FALSE_NO_SOLUTION)
                optimal = compute_optimal_product(oracle, feasible)
                result.details["missed_product"] = optimal["product_id"] if optimal else "unknown"
            return result

        product_id = result.selected_product_id
        product = conn.execute(
            """SELECT p.*, s.rating as supplier_rating, s.certified, s.region
               FROM products p
               JOIN suppliers s ON p.supplier_id = s.supplier_id
               WHERE p.product_id = ?""",
            (product_id,),
        ).fetchone()
        if product is None:
            result.failure_reasons.append(FAILURE_WRONG_PRODUCT)
            result.details["error"] = f"Product {product_id} not found"
            return result

        constraints = parse_constraints(oracle.get("constraints", {}))
        all_constraints_ok = True

        def fail(code: str) -> None:
            nonlocal all_constraints_ok
            if code not in result.failure_reasons:
                result.failure_reasons.append(code)
            all_constraints_ok = False

        if constraints.category and product["category"] != constraints.category:
            fail(FAILURE_WRONG_PRODUCT)
        if constraints.keyword:
            keyword = constraints.keyword.lower()
            fields = (
                (product["model_number"] or "").lower(),
                (product["name"] or "").lower(),
                (product["description"] or "").lower(),
            )
            if not any(keyword in field for field in fields):
                fail(FAILURE_WRONG_PRODUCT)
        if constraints.max_price is not None and product["price"] > constraints.max_price:
            fail(FAILURE_PRICE_CONSTRAINT_FAILED)
        if constraints.min_memory_gb is not None and (product["memory_gb"] or 0) < constraints.min_memory_gb:
            fail(FAILURE_MEMORY_CONSTRAINT_FAILED)
        if constraints.max_delivery_days is not None and product["delivery_days"] > constraints.max_delivery_days:
            fail(FAILURE_DELIVERY_CONSTRAINT_FAILED)
        if constraints.min_supplier_rating is not None and product["supplier_rating"] < constraints.min_supplier_rating:
            fail(FAILURE_SUPPLIER_RATING_FAILED)
        if constraints.certified_only is not None:
            expected_certified = 1 if constraints.certified_only else 0
            if product["certified"] != expected_certified:
                fail(FAILURE_SUPPLIER_CERTIFICATION_FAILED)
        if constraints.supplier_region and product["region"] != constraints.supplier_region:
            fail(FAILURE_REGION_CONSTRAINT_FAILED)
        if constraints.in_stock_only and product["stock"] <= 0:
            fail(FAILURE_OUT_OF_STOCK)
        if constraints.min_warranty_months is not None and product["warranty_months"] < constraints.min_warranty_months:
            fail(FAILURE_WARRANTY_CONSTRAINT_FAILED)

        result.constraints_satisfied = all_constraints_ok
        feasible = compute_feasible_products(oracle, conn)
        optimal = compute_optimal_product(oracle, feasible)
        result.objective_satisfied = bool(
            optimal is not None and optimal["product_id"] == product_id
        )
        if not result.objective_satisfied:
            result.failure_reasons.append(FAILURE_OBJECTIVE_NOT_OPTIMAL)
            if optimal is not None:
                result.details["optimal_product"] = optimal["product_id"]
                result.details["optimal_price"] = optimal["price"]
                result.details["optimal_rating"] = optimal["rating"]

        result.success = all_constraints_ok and result.objective_satisfied
        return result
