"""Generate M2.1 train/valid procurement tasks with programmatically computed Oracles."""

import hashlib, json, random, sys
from pathlib import Path

SEED = 20260726
TASKS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "tasks" / "m2_1"

TEMPLATES = {
    "exact_product": [
        "请查找型号为 {model} 的产品，确认其价格和供应商。",
        "需要采购型号 {model} 的设备，请找到该产品并提交采购。",
        "查找产品型号 {model}，核实库存后下单采购。",
        "型号 {model} 的物料需要补充库存，请查找并提交采购申请。",
        "请定位型号为 {model} 的产品，确认是否有库存。",
        "公司需要采购型号 {model} 的设备，请在系统中查找并提交。",
    ],
    "cheapest_feasible": [
        "需要采购{c}。请筛选出价格最低的可行商品并提交采购。",
        "请帮我找到最便宜的{c}，并完成采购流程。",
        "采购部门要求采购{c}。选择价格最低的选项。",
        "需采购{c}，预算有限，请选出价格最低且满足条件的商品。",
        "请筛选{c}，从中选择价格最低的商品进行采购。",
        "我们急需{c}。请在符合条件的商品中选择最便宜的一款。",
    ],
    "highest_rating_supplier": [
        "采购{c}，优先选择供应商评分最高的。评分相同时选价格低的。",
        "需要采购{c}。请在符合条件的商品中选择供应商评分最高的。",
        "为{m}寻找{c}，供应商信誉很重要。请选评分最高的。",
        "需要{c}。公司政策要求优先使用高评分供应商。选择评分最高的。",
        "采购{c}，供应商评分是关键考虑因素。请选评分最高的。",
    ],
    "no_feasible_product": [
        "需要{c}。请确认是否存在符合条件的商品。",
        "请帮我查找{c}，如果不存在请声明无可行商品。",
        "采购部门要求{c}。请判断是否存在满足所有条件的商品。",
        "请问是否存在{c}？如果没有请声明无解决方案。",
        "请确认{c}是否有库存。如果没有可行选项请申报。",
    ],
}


def _build_constraint_desc(ct: dict) -> str:
    parts = []
    if ct.get("category"):
        parts.append(f"类别为{ct['category']}")
    if ct.get("min_memory_gb"):
        parts.append(f"显存至少{ct['min_memory_gb']}GB")
    if ct.get("max_price"):
        parts.append(f"价格不超过{ct['max_price']}元")
    if ct.get("max_delivery_days"):
        parts.append(f"交付时间不超过{ct['max_delivery_days']}天")
    if ct.get("certified_only"):
        parts.append("仅限认证供应商")
    if ct.get("min_supplier_rating"):
        parts.append(f"供应商评分不低于{ct['min_supplier_rating']}")
    if ct.get("supplier_region"):
        parts.append(f"供应商位于{ct['supplier_region']}地区")
    if ct.get("in_stock_only"):
        parts.append("必须有库存")
    if ct.get("min_warranty_months"):
        parts.append(f"保修期至少{ct['min_warranty_months']}个月")
    return "，".join(parts) if parts else "满足条件的商品"


def generate():
    random.seed(SEED)
    suppliers = json.loads((Path(__file__).resolve().parent.parent.parent.parent / "data" / "seed" / "suppliers.json").read_text())
    products = json.loads((Path(__file__).resolve().parent.parent.parent.parent / "data" / "seed" / "products.json").read_text())

    all_tasks = {"train": [], "valid": []}
    task_idx = {"train": 0, "valid": 0}

    # Generate per type
    for split, counts in [("train", {"exact_product": 24, "cheapest_feasible": 24, "highest_rating_supplier": 24, "no_feasible_product": 24}),
                           ("valid", {"exact_product": 6, "cheapest_feasible": 6, "highest_rating_supplier": 6, "no_feasible_product": 6})]:
        for task_type, count in counts.items():
            for _ in range(count):
                task_idx[split] += 1
                tid = f"M2_1_{split[:1].upper()}{task_idx[split]:04d}"

                oracle = _generate_task(task_type, products, suppliers)
                while oracle is None:
                    oracle = _generate_task(task_type, products, suppliers)

                c = oracle["constraints"]
                desc = _build_constraint_desc(c)

                if task_type == "exact_product":
                    model = c.get("keyword", "UNKNOWN")
                    tmpl = random.choice(TEMPLATES["exact_product"])
                    instruction = tmpl.format(model=model)
                elif task_type == "no_feasible_product":
                    tmpl = random.choice(TEMPLATES["no_feasible_product"])
                    instruction = tmpl.format(c=desc)
                elif task_type == "highest_rating_supplier":
                    tmpl = random.choice(TEMPLATES["highest_rating_supplier"])
                    instruction = tmpl.format(c=desc, m="项目组")
                else:
                    tmpl = random.choice(TEMPLATES["cheapest_feasible"])
                    instruction = tmpl.format(c=desc)

                oracle["task_id"] = tid
                oracle["task_type"] = task_type

                public = {"task_id": tid, "instruction": instruction, "start_path": "/products", "task_type": task_type}
                all_tasks[split].append({"public": public, "oracle": oracle})

    # Write files
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid"]:
        with open(TASKS_DIR / f"{split}_public.jsonl", "w") as f:
            for t in all_tasks[split]:
                f.write(json.dumps(t["public"], ensure_ascii=False) + "\n")
        with open(TASKS_DIR / f"{split}_oracle.jsonl", "w") as f:
            for t in all_tasks[split]:
                f.write(json.dumps(t["oracle"], ensure_ascii=False) + "\n")

    # Manifest
    manifest = {
        "schema_version": "1.0", "seed": SEED,
        "train_task_count": len(all_tasks["train"]),
        "valid_task_count": len(all_tasks["valid"]),
        "train_public_sha256": _sha256(TASKS_DIR / "train_public.jsonl"),
        "train_oracle_sha256": _sha256(TASKS_DIR / "train_oracle.jsonl"),
        "valid_public_sha256": _sha256(TASKS_DIR / "valid_public.jsonl"),
        "valid_oracle_sha256": _sha256(TASKS_DIR / "valid_oracle.jsonl"),
    }
    with open(TASKS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated: {len(all_tasks['train'])} train + {len(all_tasks['valid'])} valid tasks")
    return manifest


def _generate_task(task_type, products, suppliers):
    """Generate one task with deterministic Oracle."""
    if task_type == "exact_product":
        gpu_prods = [p for p in products if p["category"] == "GPU" and p.get("model_number")]
        if not gpu_prods:
            return None
        prod = random.choice(gpu_prods)
        return {
            "constraints": {"category": "GPU", "keyword": prod["model_number"]},
            "objective": "exact_product",
            "expected_decision_type": "select_product",
            "expected_product_id": prod["product_id"],
            "explanation": f"型号 {prod['model_number']} 唯一对应 {prod['product_id']}"
        }

    # Build random constraint set
    c = {}
    if random.random() < 0.5:
        cats = ["GPU", "服务器", "存储", "网络"]
        c["category"] = random.choice(cats)
    if random.random() < 0.4:
        c["max_price"] = random.choice([15000, 20000, 25000, 30000, 50000, 70000, 100000])
    if random.random() < 0.4:
        c["min_memory_gb"] = random.choice([12, 16, 24, 32, 48, 64])
    if random.random() < 0.3:
        c["max_delivery_days"] = random.choice([7, 14, 21, 30])
    if random.random() < 0.3:
        c["certified_only"] = True
    if random.random() < 0.2:
        c["min_supplier_rating"] = random.choice([3.5, 4.0, 4.5])
    if random.random() < 0.25:
        c["supplier_region"] = random.choice(["华北", "华南", "华东", "西北"])
    if random.random() < 0.3:
        c["in_stock_only"] = True
    if random.random() < 0.2:
        c["min_warranty_months"] = random.choice([12, 24, 36])

    # Compute feasible products
    feasible = _filter_products(products, suppliers, c)

    if task_type == "no_feasible_product":
        if not feasible and c:  # Need at least some constraints and truly 0 results
            return {
                "constraints": c, "objective": "no_feasible_product",
                "expected_decision_type": "no_solution", "expected_product_id": "",
                "explanation": "经穷举验证，无商品满足所有约束条件"
            }
        return None

    if not feasible:
        return None

    if task_type == "cheapest_feasible":
        best = min(feasible, key=lambda p: p["price"])
        # Check uniqueness
        same_price = [p for p in feasible if p["price"] == best["price"]]
        if len(same_price) > 1:
            return None
        return {
            "constraints": c, "objective": "cheapest_feasible",
            "expected_decision_type": "select_product", "expected_product_id": best["product_id"],
            "explanation": f"满足条件的商品中{best['product_id']}价格最低({best['price']}元)"
        }

    if task_type == "highest_rating_supplier":
        sup_ratings = {p["supplier_id"]: next(s["rating"] for s in suppliers if s["supplier_id"] == p["supplier_id"])
                       for p in feasible}
        max_rating = max(sup_ratings.values())
        top = [p for p in feasible if sup_ratings[p["supplier_id"]] == max_rating]
        if len(top) > 1:
            top = sorted(top, key=lambda p: p["price"])
        best = top[0]
        return {
            "constraints": c, "objective": "highest_rating_supplier",
            "expected_decision_type": "select_product", "expected_product_id": best["product_id"],
            "explanation": f"评分最高的供应商({max_rating})中{best['product_id']}价格最低"
        }

    return None


def _filter_products(products, suppliers, c):
    sup_map = {s["supplier_id"]: s for s in suppliers}
    result = []
    for p in products:
        s = sup_map.get(p["supplier_id"], {})
        if c.get("category") and p["category"] != c["category"]:
            continue
        if c.get("keyword") and c["keyword"] not in str(p.get("model_number", "")):
            continue
        if c.get("max_price") and p["price"] > c["max_price"]:
            continue
        if c.get("min_memory_gb") and (p.get("memory_gb") or 0) < c["min_memory_gb"]:
            continue
        if c.get("max_delivery_days") and p["delivery_days"] > c["max_delivery_days"]:
            continue
        if c.get("certified_only") and not s.get("certified"):
            continue
        if c.get("min_supplier_rating") and s.get("rating", 0) < c["min_supplier_rating"]:
            continue
        if c.get("supplier_region") and s.get("region") != c["supplier_region"]:
            continue
        if c.get("in_stock_only") and p["stock"] <= 0:
            continue
        if c.get("min_warranty_months") and p["warranty_months"] < c["min_warranty_months"]:
            continue
        result.append(p)
    return result


def _sha256(path):
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return ""


if __name__ == "__main__":
    generate()
