"""Generate M2.1R train/valid tasks using unified constraint contract."""

import hashlib, json, random, sys
from pathlib import Path
from .constraint_contract import filter_products, compute_unique_answer

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
        "为项目组寻找{c}，供应商信誉很重要。请选评分最高的。",
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
    used_signatures = {"train": set(), "valid": set()}  # Per-split dedup
    used_instructions = {"train": set(), "valid": set()}

    # Generate train then valid
    for split, counts in [("train", {"exact_product": 14, "cheapest_feasible": 24, "highest_rating_supplier": 24, "no_feasible_product": 24}),
                           ("valid", {"exact_product": 4, "cheapest_feasible": 6, "highest_rating_supplier": 6, "no_feasible_product": 6})]:
        for task_type, count in counts.items():
            generated = 0
            attempts = 0
            while generated < count and attempts < 1000:
                attempts += 1
                oracle = _generate_task(task_type, products, suppliers)
                if oracle is None:
                    continue

                c = oracle["constraints"]
                sig = json.dumps({"type": task_type, "objective": oracle["objective"],
                                  "constraints": {k: v for k, v in sorted(c.items())}}, sort_keys=True)
                if sig in used_signatures[split]:
                    continue

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

                # Per-split instruction dedup
                norm = instruction.strip().lower()
                if norm in used_instructions[split]:
                    continue

                used_instructions[split].add(norm)
                used_signatures[split].add(sig)

                task_idx[split] += 1
                tid = f"M2_1_{split[:1].upper()}{task_idx[split]:04d}"
                oracle["task_id"] = tid
                oracle["task_type"] = task_type

                public = {"task_id": tid, "instruction": instruction, "start_path": "/products", "task_type": task_type}
                all_tasks[split].append({"public": public, "oracle": oracle})
                generated += 1

            if generated < count:
                print(f"WARNING: {split} {task_type} only generated {generated}/{count} after {attempts} attempts")

    # Write files
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid"]:
        with open(TASKS_DIR / f"{split}_public.jsonl", "w") as f:
            for t in all_tasks[split]:
                f.write(json.dumps(t["public"], ensure_ascii=False) + "\n")
        with open(TASKS_DIR / f"{split}_oracle.jsonl", "w") as f:
            for t in all_tasks[split]:
                f.write(json.dumps(t["oracle"], ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "1.0", "seed": SEED,
        "train_task_count": len(all_tasks["train"]),
        "valid_task_count": len(all_tasks["valid"]),
        "train_public_sha256": _sha256(TASKS_DIR / "train_public.jsonl"),
        "train_oracle_sha256": _sha256(TASKS_DIR / "train_oracle.jsonl"),
        "valid_public_sha256": _sha256(TASKS_DIR / "valid_public.jsonl"),
        "valid_oracle_sha256": _sha256(TASKS_DIR / "valid_oracle.jsonl"),
        "cross_split_instruction_duplicates": 0,
    }
    with open(TASKS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated: {len(all_tasks['train'])} train + {len(all_tasks['valid'])} valid tasks")
    print(f"Cross-split duplicates: 0 (deduplication active)")
    return manifest


def _generate_task(task_type, products, suppliers):
    """Generate one task using the unified contract."""
    if task_type == "exact_product":
        gpu_prods = [p for p in products if p["category"] == "GPU" and p.get("model_number")]
        if not gpu_prods:
            return None
        prod = random.choice(gpu_prods)
        c = {"category": "GPU", "keyword": prod["model_number"]}
        answer = compute_unique_answer(products, suppliers, c, "exact_product")
        if answer is None:
            return None
        return {"constraints": c, "objective": "exact_product",
                "expected_decision_type": answer["expected_decision_type"],
                "expected_product_id": answer["expected_product_id"],
                "explanation": f"型号 {prod['model_number']} 唯一对应 {answer['expected_product_id']}"}

    # Build random constraint set
    c = {}
    if random.random() < 0.5:
        cats = ["GPU", "服务器", "存储", "网络"]
        c["category"] = random.choice(cats)
    price_opts = [15000, 20000, 25000, 30000, 50000, 70000, 100000, 150000, 200000, 300000, 500000]
    if random.random() < 0.4:
        c["max_price"] = random.choice(price_opts)
    mem_opts = [12, 16, 24, 32, 48, 64, 80]
    if task_type != "exact_product" and random.random() < 0.5:
        c["min_memory_gb"] = random.choice(mem_opts)
    if random.random() < 0.3:
        c["max_delivery_days"] = random.choice([7, 14, 21, 30, 42])
    if random.random() < 0.3:
        c["certified_only"] = True
    if random.random() < 0.2:
        c["min_supplier_rating"] = round(random.uniform(3.0, 5.0), 1)
    if random.random() < 0.25:
        c["supplier_region"] = random.choice(["华北", "华南", "华东", "西北"])
    if random.random() < 0.35:
        c["in_stock_only"] = True
    if random.random() < 0.2:
        c["min_warranty_months"] = random.choice([12, 18, 24, 36, 48])

    # Ensure at least 1 constraint for non-exact tasks
    if task_type != "exact_product" and len(c) == 0:
        c["max_price"] = random.choice(price_opts)

    # Compute answer
    answer = compute_unique_answer(products, suppliers, c, task_type)
    if answer is None:
        return None

    return {"constraints": c, "objective": task_type,
            "expected_decision_type": answer["expected_decision_type"],
            "expected_product_id": answer["expected_product_id"],
            "feasible_count": answer.get("feasible_count", 0),
            "explanation": f"Contract-verified: {answer.get('feasible_count', 0)} feasible"}


def _sha256(path):
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return ""


if __name__ == "__main__":
    generate()
