"""Task loading, source isolation, validation, and constraint computation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from .models import (
    OBJECTIVE_CHEAPEST_FEASIBLE,
    OBJECTIVE_EXACT_PRODUCT,
    OBJECTIVE_HIGHEST_RATING_SUPPLIER,
    OBJECTIVE_NO_FEASIBLE_PRODUCT,
    TaskConstraints,
)

TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tasks"
TASK_DIR_ENV = "MINIWEBWORK_TASK_DIR"


class TaskDataError(ValueError):
    """Raised when a task source violates the dataset contract."""


def _resolve_task_dir(task_dir: str | Path | None = None) -> Path | None:
    """Resolve the exclusive task source for the current process.

    When an explicit directory or ``MINIWEBWORK_TASK_DIR`` is provided, only
    files from that directory are loaded.  The default 15-task dataset is not
    merged in.  This prevents train/valid/rollout tasks from being silently
    shadowed by equal IDs in the historical default dataset.
    """
    if task_dir is not None:
        return Path(task_dir).expanduser().resolve()
    configured = os.environ.get(TASK_DIR_ENV, "").strip()
    return Path(configured).expanduser().resolve() if configured else None


def _task_paths(kind: str, task_dir: str | Path | None = None) -> list[Path]:
    if kind not in {"public", "oracle"}:
        raise ValueError(f"Unsupported task kind: {kind}")

    source_dir = _resolve_task_dir(task_dir)
    if source_dir is None:
        return [TASKS_DIR / f"tasks_{kind}.jsonl"]
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {source_dir}")

    paths = sorted(source_dir.glob(f"*_{kind}.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No *_{kind}.jsonl files found in {source_dir}")
    return paths


def _load_jsonl(paths: Iterable[Path], kind: str) -> list[dict]:
    tasks: list[dict] = []
    seen: dict[str, Path] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TaskDataError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if not isinstance(task, dict):
                    raise TaskDataError(f"Expected object in {path}:{line_number}")
                task_id = task.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    raise TaskDataError(f"Missing task_id in {path}:{line_number}")
                if task_id in seen:
                    raise TaskDataError(
                        f"Duplicate {kind} task_id {task_id!r} in {seen[task_id]} and {path}"
                    )
                seen[task_id] = path
                tasks.append(task)
    return tasks


def load_public_tasks(task_dir: str | Path | None = None) -> list[dict]:
    """Load public tasks from exactly one task source."""
    return _load_jsonl(_task_paths("public", task_dir), "public")


def load_oracle_tasks(task_dir: str | Path | None = None) -> list[dict]:
    """Load private Oracle tasks from exactly one task source."""
    return _load_jsonl(_task_paths("oracle", task_dir), "oracle")


def get_public_task(task_id: str, task_dir: str | Path | None = None) -> Optional[dict]:
    return next((task for task in load_public_tasks(task_dir) if task["task_id"] == task_id), None)


def get_oracle(task_id: str, task_dir: str | Path | None = None) -> Optional[dict]:
    return next((task for task in load_oracle_tasks(task_dir) if task["task_id"] == task_id), None)


def parse_constraints(data: dict) -> TaskConstraints:
    """Parse an Oracle constraint object into the typed runtime contract."""
    return TaskConstraints(
        category=data.get("category"),
        keyword=data.get("keyword"),
        max_price=data.get("max_price"),
        min_memory_gb=data.get("min_memory_gb"),
        max_delivery_days=data.get("max_delivery_days"),
        min_supplier_rating=data.get("min_supplier_rating"),
        certified_only=data.get("certified_only"),
        supplier_region=data.get("supplier_region"),
        in_stock_only=data.get("in_stock_only", False),
        min_warranty_months=data.get("min_warranty_months"),
    )


def compute_feasible_products(oracle: dict, conn) -> list:
    """Recompute all products satisfying the Oracle constraints."""
    c = parse_constraints(oracle.get("constraints", {}))
    query = """
        SELECT p.*, s.supplier_id as s_supplier_id, s.name as supplier_name,
               s.rating, s.region, s.certified, s.delivery_reliability
        FROM products p
        JOIN suppliers s ON p.supplier_id = s.supplier_id
        WHERE 1=1
    """
    params: list = []

    if c.category:
        query += " AND p.category = ?"
        params.append(c.category)
    if c.keyword:
        query += " AND (p.model_number LIKE ? OR p.name LIKE ? OR p.description LIKE ?)"
        keyword = f"%{c.keyword}%"
        params.extend([keyword, keyword, keyword])
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

    return conn.execute(query, params).fetchall()


def compute_optimal_product(oracle: dict, feasible: list):
    """Select the deterministic optimum for the requested objective."""
    objective = oracle.get("objective", "")
    if not feasible:
        return None
    if objective == OBJECTIVE_EXACT_PRODUCT:
        return feasible[0] if len(feasible) == 1 else None
    if objective == OBJECTIVE_CHEAPEST_FEASIBLE:
        return min(feasible, key=lambda row: (row["price"], row["product_id"]))
    if objective == OBJECTIVE_HIGHEST_RATING_SUPPLIER:
        max_rating = max(row["rating"] for row in feasible)
        top = [row for row in feasible if row["rating"] == max_rating]
        return min(top, key=lambda row: (row["price"], row["product_id"]))
    return None


def validate_tasks(conn, task_dir: str | Path | None = None) -> dict:
    """Validate source pairing, feasibility, and frozen expected answers."""
    errors: list[str] = []
    public_tasks = load_public_tasks(task_dir)
    oracle_tasks = load_oracle_tasks(task_dir)

    public_ids = {task["task_id"] for task in public_tasks}
    oracle_ids = {task["task_id"] for task in oracle_tasks}
    only_public = public_ids - oracle_ids
    only_oracle = oracle_ids - public_ids
    if only_public:
        errors.append(f"Tasks in public only (no oracle): {sorted(only_public)}")
    if only_oracle:
        errors.append(f"Tasks in oracle only (no public): {sorted(only_oracle)}")
    if not public_tasks:
        errors.append("Task source is empty")

    for oracle in oracle_tasks:
        task_id = oracle["task_id"]
        objective = oracle.get("objective", "")
        expected_decision = oracle.get("expected_decision_type", "")
        expected_product_id = oracle.get("expected_product_id", "")

        if expected_decision == "select_product" and not expected_product_id:
            errors.append(f"{task_id}: select_product but no expected_product_id")

        feasible = compute_feasible_products(oracle, conn)
        if objective == OBJECTIVE_NO_FEASIBLE_PRODUCT:
            if feasible:
                errors.append(
                    f"{task_id}: expected no_solution but found feasible: "
                    f"{[row['product_id'] for row in feasible]}"
                )
            if expected_decision != "no_solution":
                errors.append(f"{task_id}: expected no_solution decision type mismatch")
            continue

        if not feasible:
            errors.append(f"{task_id}: expected feasible products but got none")
            continue

        optimal = compute_optimal_product(oracle, feasible)
        if optimal is None:
            errors.append(f"{task_id}: could not determine a unique deterministic optimum")
            continue
        if optimal["product_id"] != expected_product_id:
            errors.append(
                f"{task_id}: optimal product mismatch; expected {expected_product_id}, "
                f"computed {optimal['product_id']}"
            )

    return {
        "valid": not errors,
        "errors": errors,
        "public_count": len(public_tasks),
        "oracle_count": len(oracle_tasks),
        "task_source": str(_resolve_task_dir(task_dir) or TASKS_DIR),
    }
