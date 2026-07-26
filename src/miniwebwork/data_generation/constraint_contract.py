"""Unified constraint contract v1.0 — single source of truth for all layers.

Task generator, SQL builder, verifier, and expert agent MUST use these definitions.
"""

CONSTRAINT_DEFINITIONS = {
    "max_price":          {"field": "price",             "op": "<=", "type": "float"},
    "min_memory_gb":      {"field": "memory_gb",         "op": ">=", "type": "int",   "nullable": True},
    "max_delivery_days":  {"field": "delivery_days",     "op": "<=", "type": "int"},
    "min_supplier_rating": {"field": "rating",            "op": ">=", "type": "float", "table": "suppliers"},
    "certified_only":     {"field": "certified",          "op": "==", "type": "bool",  "table": "suppliers", "value": 1},
    "supplier_region":    {"field": "region",             "op": "==", "type": "str",   "table": "suppliers"},
    "in_stock_only":      {"field": "stock",              "op": ">",  "type": "int",   "value": 0},
    "min_warranty_months": {"field": "warranty_months",   "op": ">=", "type": "int"},
}


def filter_products(products: list, suppliers: list, constraints: dict) -> list:
    """Compute feasible products from constraints. THE canonical implementation."""
    sup_by_id = {s["supplier_id"]: s for s in suppliers}
    result = []
    for p in products:
        s = sup_by_id.get(p["supplier_id"], {})
        if not _check(p, s, constraints):
            continue
        result.append(p)
    return result


def _check(product: dict, supplier: dict, constraints: dict) -> bool:
    c = constraints
    if c.get("category") and product.get("category") != c["category"]:
        return False
    if c.get("keyword"):
        kw = c["keyword"].lower()
        model = (product.get("model_number") or "").lower()
        name = (product.get("name") or "").lower()
        if kw not in model and kw not in name:
            return False
    if c.get("max_price") is not None and product.get("price", 0) > c["max_price"]:
        return False
    if c.get("min_memory_gb") is not None and (product.get("memory_gb") or 0) < c["min_memory_gb"]:
        return False
    if c.get("max_delivery_days") is not None and product.get("delivery_days", 999) > c["max_delivery_days"]:
        return False
    if c.get("certified_only") and not supplier.get("certified"):
        return False
    if c.get("min_supplier_rating") is not None and supplier.get("rating", 0) < c["min_supplier_rating"]:
        return False
    if c.get("supplier_region") and supplier.get("region") != c["supplier_region"]:
        return False
    if c.get("in_stock_only") and product.get("stock", 0) <= 0:
        return False
    if c.get("min_warranty_months") is not None and product.get("warranty_months", 0) < c["min_warranty_months"]:
        return False
    return True


def compute_unique_answer(products: list, suppliers: list, constraints: dict, objective: str) -> dict:
    """Compute the unique expected answer for a task. Returns None on ambiguity."""
    feasible = filter_products(products, suppliers, constraints)

    if objective == "no_feasible_product":
        if len(feasible) == 0:
            return {"expected_decision_type": "no_solution", "expected_product_id": "",
                    "feasible_count": 0}
        return None  # Should be no_solution but feasible products exist

    if len(feasible) == 0:
        return None  # Expected to find products but none exist

    if objective == "exact_product":
        if len(feasible) == 1:
            return {"expected_decision_type": "select_product",
                    "expected_product_id": feasible[0]["product_id"],
                    "feasible_count": 1}
        return None  # Ambiguous

    if objective == "cheapest_feasible":
        best = min(feasible, key=lambda p: p["price"])
        same_price = [p for p in feasible if p["price"] == best["price"]]
        if len(same_price) > 1:
            return None  # Tie — not unique
        return {"expected_decision_type": "select_product",
                "expected_product_id": best["product_id"],
                "feasible_count": len(feasible)}

    if objective == "highest_rating_supplier":
        sup_ratings = {p["supplier_id"]: next((s["rating"] for s in suppliers
                        if s["supplier_id"] == p["supplier_id"]), 0) for p in feasible}
        max_r = max(sup_ratings.values())
        top = [p for p in feasible if sup_ratings[p["supplier_id"]] == max_r]
        if len(top) > 1:
            top = sorted(top, key=lambda p: p["price"])
        return {"expected_decision_type": "select_product",
                "expected_product_id": top[0]["product_id"],
                "feasible_count": len(feasible)}

    return None
