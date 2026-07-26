"""Task loading, validation, and constraint computation for MiniWebWork-RL."""

import json
from pathlib import Path
from typing import Optional

from .models import (
    OBJECTIVE_CHEAPEST_FEASIBLE,
    OBJECTIVE_EXACT_PRODUCT,
    OBJECTIVE_HIGHEST_RATING_SUPPLIER,
    OBJECTIVE_NO_FEASIBLE_PRODUCT,
    TaskConstraints,
)

import os

TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tasks"


def _get_public_paths() -> list:
    """Get public task file paths, including M2.1 custom paths if set."""
    paths = [TASKS_DIR / "tasks_public.jsonl"]
    extra = os.environ.get("MINIWEBWORK_TASK_DIR", "")
    if extra:
        ep = Path(extra)
        for f in sorted(ep.glob("*_public.jsonl")):
            if f not in paths:
                paths.append(f)
    return paths


def _get_oracle_paths() -> list:
    paths = [TASKS_DIR / "tasks_oracle.jsonl"]
    extra = os.environ.get("MINIWEBWORK_TASK_DIR", "")
    if extra:
        ep = Path(extra)
        for f in sorted(ep.glob("*_oracle.jsonl")):
            if f not in paths:
                paths.append(f)
    return paths


def load_public_tasks() -> list:
    """Load public tasks from JSONL."""
    tasks = []
    for path in _get_public_paths():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))
    return tasks


def load_oracle_tasks() -> list:
    """Load oracle tasks from JSONL."""
    tasks = []
    for path in _get_oracle_paths():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))
    return tasks


def get_public_task(task_id: str) -> Optional[dict]:
    for t in load_public_tasks():
        if t["task_id"] == task_id:
            return t
    return None


def get_oracle(task_id: str) -> Optional[dict]:
    for t in load_oracle_tasks():
        if t["task_id"] == task_id:
            return t
    return None


def parse_constraints(data: dict) -> TaskConstraints:
    """Parse constraints dict into TaskConstraints dataclass."""
    return TaskConstraints(
        category=data.get("category"),
        keyword=data.get("keyword"),
        max_price=data.get("max_price"),
        min_memory_gb=data.get("min_memory_gb"),
        max_delivery_days=data.get("max_delivery_days"),
        min_supplier_rating=data.get("min_supplier_rating"),
        certified_only=data.get("certified_only"),  # None if not in dict -> no filter
        supplier_region=data.get("supplier_region"),
        in_stock_only=data.get("in_stock_only", False),
        min_warranty_months=data.get("min_warranty_months"),
    )


def compute_feasible_products(oracle: dict, conn) -> list:
    """Recompute the set of feasible products from oracle constraints.

    Returns list of (product_row, supplier_row) tuples.
    """
    constraints = parse_constraints(oracle.get("constraints", {}))
    c = constraints

    query = """
        SELECT p.*, s.supplier_id as s_supplier_id, s.name as supplier_name,
               s.rating, s.region, s.certified, s.delivery_reliability
        FROM products p
        JOIN suppliers s ON p.supplier_id = s.supplier_id
        WHERE 1=1
    """
    params = []

    if c.category:
        query += " AND p.category = ?"
        params.append(c.category)

    if c.keyword:
        query += " AND (p.model_number LIKE ? OR p.name LIKE ? OR p.description LIKE ?)"
        kw = f"%{c.keyword}%"
        params.extend([kw, kw, kw])

    if c.max_price is not None:
        query += " AND p.price <= ?"
        params.append(c.max_price)

    if c.min_memory_gb is not None:
        query += " AND p.memory_gb >= ?"
        params.append(c.min_memory_gb)

    if c.max_delivery_days is not None:
        query += " AND p.delivery_days <= ?"
        params.append(c.max_delivery_days)

    if c.min_supplier_rating is not None:
        query += " AND s.rating >= ?"
        params.append(c.min_supplier_rating)

    if c.certified_only is not None:
        # certified_only=True -> only certified; certified_only=False -> only uncertified
        query += " AND s.certified = ?"
        params.append(1 if c.certified_only else 0)

    if c.supplier_region:
        query += " AND s.region = ?"
        params.append(c.supplier_region)

    if c.in_stock_only:
        query += " AND p.stock > 0"

    if c.min_warranty_months is not None:
        query += " AND p.warranty_months >= ?"
        params.append(c.min_warranty_months)

    rows = conn.execute(query, params).fetchall()
    return rows


def compute_optimal_product(oracle: dict, feasible: list):
    """Given feasible products, select the optimal one per objective."""
    objective = oracle.get("objective", "")
    if not feasible:
        return None

    if objective == OBJECTIVE_EXACT_PRODUCT:
        return feasible[0] if len(feasible) == 1 else None

    if objective == OBJECTIVE_CHEAPEST_FEASIBLE:
        return min(feasible, key=lambda r: r["price"])

    if objective == OBJECTIVE_HIGHEST_RATING_SUPPLIER:
        max_rating = max(r["rating"] for r in feasible)
        top = [r for r in feasible if r["rating"] == max_rating]
        return min(top, key=lambda r: r["price"])

    return None


def validate_tasks(conn) -> dict:
    """Validate all tasks for correctness. Returns dict with validation results."""
    errors = []
    public_tasks = load_public_tasks()
    oracle_tasks = load_oracle_tasks()

    public_ids = {t["task_id"] for t in public_tasks}
    oracle_ids = {t["task_id"] for t in oracle_tasks}

    # Check 1-to-1 mapping
    only_public = public_ids - oracle_ids
    only_oracle = oracle_ids - public_ids
    if only_public:
        errors.append(f"Tasks in public only (no oracle): {only_public}")
    if only_oracle:
        errors.append(f"Tasks in oracle only (no public): {only_oracle}")

    # Check minimum task count
    if len(public_tasks) < 15:
        errors.append(f"Expected >= 15 public tasks, got {len(public_tasks)}")

    # Check each task
    for oracle in oracle_tasks:
        tid = oracle["task_id"]
        obj = oracle.get("objective", "")
        expected_decision = oracle.get("expected_decision_type", "")
        expected_pid = oracle.get("expected_product_id", "")

        if expected_decision == "select_product" and not expected_pid:
            errors.append(f"{tid}: select_product but no expected_product_id")

        # Recompute feasible products
        feasible = compute_feasible_products(oracle, conn)

        if obj == OBJECTIVE_NO_FEASIBLE_PRODUCT:
            if len(feasible) > 0:
                pids = [r["product_id"] for r in feasible]
                errors.append(f"{tid}: expected no_solution but found feasible: {pids}")
            if expected_decision != "no_solution":
                errors.append(f"{tid}: expected no_solution decision type mismatch")
        else:
            if len(feasible) == 0:
                errors.append(f"{tid}: expected feasible products but got none")
                continue

            optimal = compute_optimal_product(oracle, feasible)
            if optimal is None:
                errors.append(f"{tid}: could not determine optimal product")
                continue

            if optimal["product_id"] != expected_pid:
                errors.append(
                    f"{tid}: optimal product mismatch. "
                    f"Expected {expected_pid}, computed {optimal['product_id']} "
                    f"(price={optimal['price']}, rating={optimal['rating']})"
                )

            # Check that no other product is "more optimal"
            for alt in feasible:
                if alt["product_id"] == optimal["product_id"]:
                    continue
                if obj == OBJECTIVE_CHEAPEST_FEASIBLE and alt["price"] < optimal["price"]:
                    errors.append(f"{tid}: {alt['product_id']} cheaper than expected {optimal['product_id']}")
                if obj == OBJECTIVE_HIGHEST_RATING_SUPPLIER:
                    if alt["rating"] > optimal["rating"]:
                        errors.append(f"{tid}: {alt['product_id']} rating higher than expected {optimal['product_id']}")
                    elif alt["rating"] == optimal["rating"] and alt["price"] < optimal["price"]:
                        errors.append(f"{tid}: {alt['product_id']} same rating but cheaper than {optimal['product_id']}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "public_count": len(public_tasks),
        "oracle_count": len(oracle_tasks),
    }
