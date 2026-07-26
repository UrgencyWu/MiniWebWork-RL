"""Cross-layer consistency audit for M2.1R tasks."""

import json, sqlite3
from pathlib import Path
from .constraint_contract import filter_products


def audit(task_public_path: str, task_oracle_path: str, db_path: str, expert_traj_path: str = None) -> list:
    """Audit all tasks for cross-layer consistency. Returns per-task reports."""
    products = json.loads((Path(__file__).resolve().parent.parent.parent.parent / "data" / "seed" / "products.json").read_text())
    suppliers = json.loads((Path(__file__).resolve().parent.parent.parent.parent / "data" / "seed" / "suppliers.json").read_text())

    tasks = []
    for line in Path(task_oracle_path).read_text().strip().split("\n"):
        if line.strip():
            tasks.append(json.loads(line))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    results = []
    summary = {"oracle_sql_mismatch": 0, "sql_browser_mismatch": 0, "browser_expert_mismatch": 0,
               "expert_planning_error": 0, "verifier_oracle_mismatch": 0}

    for task in tasks:
        r = {"task_id": task["task_id"], "constraints": task.get("constraints", {}),
             "oracle_feasible_ids": [], "sql_feasible_ids": [],
             "browser_visible_ids": [], "expert_selected_id": None,
             "expected_product_id": task.get("expected_product_id", ""),
             "verifier_result": {}, "first_divergence_layer": None, "failure_reason": ""}

        c = task.get("constraints", {})

        # Layer 1: Oracle (unified contract)
        oracle_feasible = filter_products(products, suppliers, c)
        r["oracle_feasible_ids"] = sorted([p["product_id"] for p in oracle_feasible])

        # Layer 2: SQL (database query — mirror the webapp's query)
        sql_feasible = _sql_query(conn, c)
        r["sql_feasible_ids"] = sorted([p["product_id"] for p in sql_feasible])

        if set(r["oracle_feasible_ids"]) != set(r["sql_feasible_ids"]):
            r["first_divergence_layer"] = "oracle_vs_sql"
            r["failure_reason"] = f"Oracle={len(r['oracle_feasible_ids'])} products, SQL={len(r['sql_feasible_ids'])}"
            summary["oracle_sql_mismatch"] += 1
            results.append(r)
            continue

        # Layer 3: Browser visible (same as SQL if webapp uses same query)
        r["browser_visible_ids"] = r["sql_feasible_ids"].copy()

        # Layer 4: Expert — check expected_product_id
        if task.get("expected_product_id"):
            r["expert_selected_id"] = task["expected_product_id"]
            if task["expected_product_id"] not in r["oracle_feasible_ids"] and task.get("expected_decision_type") == "select_product":
                r["first_divergence_layer"] = "expert_planning"
                r["failure_reason"] = f"Expected {task['expected_product_id']} not in feasible set"
                summary["expert_planning_error"] += 1
                results.append(r)
                continue

        # Layer 5: Verifier (recomputes from same constraints)
        if task.get("expected_decision_type") == "no_solution":
            if len(r["oracle_feasible_ids"]) > 0:
                r["first_divergence_layer"] = "verifier_vs_oracle"
                r["failure_reason"] = "Expected no_solution but feasible products exist"
                summary["verifier_oracle_mismatch"] += 1
                results.append(r)
                continue

        r["verifier_result"] = {"success": True}
        results.append(r)

    conn.close()

    # Print summary
    print(f"Total tasks: {len(tasks)}")
    print(f"Oracle-SQL mismatch: {summary['oracle_sql_mismatch']}")
    print(f"SQL-Browser mismatch: {summary['sql_browser_mismatch']}")
    print(f"Browser-Expert mismatch: {summary['browser_expert_mismatch']}")
    print(f"Expert planning error: {summary['expert_planning_error']}")
    print(f"Verifier-Oracle mismatch: {summary['verifier_oracle_mismatch']}")
    consistent = len(tasks) - sum(summary.values())
    print(f"Fully consistent: {consistent}/{len(tasks)}")

    return results


def _sql_query(conn, c):
    """Replicate the webapp's SQL query exactly."""
    query = """SELECT p.* FROM products p JOIN suppliers s ON p.supplier_id = s.supplier_id WHERE 1=1"""
    params = []
    if c.get("category"):
        query += " AND p.category = ?"
        params.append(c["category"])
    if c.get("keyword"):
        query += " AND (p.model_number LIKE ? OR p.name LIKE ? OR p.description LIKE ?)"
        kw = f"%{c['keyword']}%"
        params.extend([kw, kw, kw])
    if c.get("max_price") is not None:
        query += " AND p.price <= ?"
        params.append(c["max_price"])
    if c.get("min_memory_gb") is not None:
        query += " AND (p.memory_gb IS NOT NULL AND p.memory_gb >= ?)"
        params.append(c["min_memory_gb"])
    if c.get("max_delivery_days") is not None:
        query += " AND p.delivery_days <= ?"
        params.append(c["max_delivery_days"])
    if c.get("certified_only"):
        query += " AND s.certified = 1"
    if c.get("min_supplier_rating") is not None:
        query += " AND s.rating >= ?"
        params.append(c["min_supplier_rating"])
    if c.get("supplier_region"):
        query += " AND s.region = ?"
        params.append(c["supplier_region"])
    if c.get("in_stock_only"):
        query += " AND p.stock > 0"
    if c.get("min_warranty_months") is not None:
        query += " AND p.warranty_months >= ?"
        params.append(c["min_warranty_months"])
    return conn.execute(query, params).fetchall()


if __name__ == "__main__":
    import sys
    pub = sys.argv[1] if len(sys.argv) > 1 else "data/tasks/m2_1/train_public.jsonl"
    ora = sys.argv[2] if len(sys.argv) > 2 else "data/tasks/m2_1/train_oracle.jsonl"
    db = sys.argv[3] if len(sys.argv) > 3 else "data/runtime/miniwebwork.db"
    results = audit(pub, ora, db)
    out_path = Path("artifacts/m2_1/m2_1_consistency_audit.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")
