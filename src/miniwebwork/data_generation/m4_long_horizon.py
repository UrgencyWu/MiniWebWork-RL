"""Deterministic, split-isolated tasks for the focused long-horizon study."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from ..m4_long_horizon_protocol import audit_horizon_dataset, classify_horizon
from .constraint_contract import compute_unique_answer, filter_products

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tasks" / "m4_long_horizon_v2"
DEFAULT_SEED_DIR = PROJECT_ROOT / "data" / "seed_m4_long_horizon_v2"
DATASET_ID = "m4_long_horizon_v2"
SCHEMA_VERSION = "2.0"
SPLIT_WORLD_COUNTS = {"train": 60, "dev": 18, "test": 30}
TASK_TYPES = (
    "exact_product",
    "cheapest_feasible",
    "highest_reliability_supplier",
    "no_feasible_product",
)
SPLIT_ROLES = {
    "train": "optimizer_source",
    "dev": "model_selection_only",
    "test": "frozen_final_evaluation",
}
SPLIT_PURPOSES = {
    "train": {"offline_training", "online_training", "development_evaluation"},
    "dev": {"development_evaluation", "model_selection"},
    "test": {"final_evaluation"},
}
SPLIT_MANIFEST_FILENAME = "m4_long_horizon_split_manifest.json"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
FEASIBLE_SUPPLIER_ROLES = ("A", "B", "C")
SUPPLIER_VISIT_ROLES = ("B", "A", "C")
WINNER_ASSIGNMENT_VERSION = "split_balanced_sha256_permutation_v1"
WINNER_ASSIGNMENT_SALT = "m4-long-horizon-v2-public-id-debias-20260808"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _signature(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _jsonl_text(values: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for value in values
    )


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _world_id(index: int) -> str:
    return f"W{index:03d}"


def _world_category(world_id: str) -> str:
    index = int(world_id[1:])
    return ("GPU", "服务器", "存储", "网络")[(index - 1) % 4]


def _supplier_id(world_id: str, role: str) -> str:
    return f"M4-LH-SUP-{world_id}-{role}"


def _product_id(world_id: str, role: str) -> str:
    return f"M4-LH-PRD-{world_id}-{role}"


def _world_split_offset(world_id: str) -> tuple[str, int]:
    world_index = int(world_id[1:])
    first_index = 1
    for split, count in SPLIT_WORLD_COUNTS.items():
        if first_index <= world_index < first_index + count:
            return split, world_index - first_index
        first_index += count
    raise ValueError(f"world outside frozen split roster: {world_id}")


@lru_cache(maxsize=None)
def _balanced_winner_roles(split: str) -> tuple[str, ...]:
    """Return an exactly balanced, non-periodic role roster for one split."""

    if split not in SPLIT_WORLD_COUNTS:
        raise ValueError(f"unknown split: {split}")
    count = SPLIT_WORLD_COUNTS[split]
    if count % len(FEASIBLE_SUPPLIER_ROLES):
        raise ValueError(f"split cannot balance supplier roles exactly: {split}")
    tokens = [
        (role, occurrence)
        for role in FEASIBLE_SUPPLIER_ROLES
        for occurrence in range(count // len(FEASIBLE_SUPPLIER_ROLES))
    ]
    tokens.sort(
        key=lambda token: hashlib.sha256(
            (
                f"{WINNER_ASSIGNMENT_SALT}|winner|{split}|"
                f"{token[0]}|{token[1]}"
            ).encode("utf-8")
        ).digest()
    )
    return tuple(role for role, _ in tokens)


def supplier_reliability_by_role(world_id: str) -> dict[str, float]:
    """Assign a split-balanced optimum with no simple world-ID periodic rule."""

    split, offset = _world_split_offset(world_id)
    winner = _balanced_winner_roles(split)[offset]
    remaining = sorted(
        (role for role in FEASIBLE_SUPPLIER_ROLES if role != winner),
        key=lambda role: hashlib.sha256(
            (
                f"{WINNER_ASSIGNMENT_SALT}|secondary|{world_id}|{role}"
            ).encode("utf-8")
        ).digest(),
    )
    return {winner: 0.99, remaining[0]: 0.93, remaining[1]: 0.88}


def build_long_horizon_seed() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build four products and four private-to-world suppliers per world."""

    suppliers: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    regions = ("华北", "华东", "华南", "西北")
    total_worlds = sum(SPLIT_WORLD_COUNTS.values())
    for index in range(1, total_worlds + 1):
        world_id = _world_id(index)
        category = _world_category(world_id)
        region = regions[(index - 1) % len(regions)]
        reliability = supplier_reliability_by_role(world_id)
        supplier_specs = {
            "A": ("Northbridge Supply", 4.60, 1, reliability["A"]),
            "B": ("Lattice Supply", 4.20, 1, reliability["B"]),
            "C": ("Harbor Supply", 4.95, 1, reliability["C"]),
            "D": ("Clearance Exchange", 3.50, 0, 0.65),
        }
        for role, (label, rating, certified, reliability) in supplier_specs.items():
            suppliers.append(
                {
                    "supplier_id": _supplier_id(world_id, role),
                    "name": f"{world_id} {label}",
                    "rating": rating,
                    "region": region,
                    "certified": certified,
                    "delivery_reliability": reliability,
                    "description": (
                        f"Supplier {role} dedicated to isolated long-horizon world {world_id}."
                    ),
                }
            )
        base_price = 10_000 + 37 * index
        common = {
            "category": category,
            "description": (
                f"Dedicated product for focused long-horizon procurement world {world_id}."
            ),
        }
        products.extend(
            [
                {
                    **common,
                    "product_id": _product_id(world_id, "A"),
                    "supplier_id": _supplier_id(world_id, "A"),
                    "name": f"Atlas {world_id} A Node",
                    "price": float(base_price + 1_400),
                    "memory_gb": 64,
                    "delivery_days": 5,
                    "stock": 9,
                    "warranty_months": 36,
                    "model_number": f"M4-LH-{world_id}-EXACT",
                },
                {
                    **common,
                    "product_id": _product_id(world_id, "B"),
                    "supplier_id": _supplier_id(world_id, "B"),
                    "name": f"Atlas {world_id} B Node",
                    "price": float(base_price),
                    "memory_gb": 48,
                    "delivery_days": 7,
                    "stock": 15,
                    "warranty_months": 24,
                    "model_number": f"M4-LH-{world_id}-VALUE",
                },
                {
                    **common,
                    "product_id": _product_id(world_id, "C"),
                    "supplier_id": _supplier_id(world_id, "C"),
                    "name": f"Atlas {world_id} C Node",
                    "price": float(base_price + 3_800),
                    "memory_gb": 96,
                    "delivery_days": 12,
                    "stock": 7,
                    "warranty_months": 48,
                    "model_number": f"M4-LH-{world_id}-EXTENDED",
                },
                {
                    **common,
                    "product_id": _product_id(world_id, "D"),
                    "supplier_id": _supplier_id(world_id, "D"),
                    "name": f"Atlas {world_id} Clearance Node",
                    "price": float(base_price + 2_100),
                    "memory_gb": 24,
                    "delivery_days": 20,
                    "stock": 0,
                    "warranty_months": 12,
                    "model_number": f"M4-LH-{world_id}-DISTRACTOR",
                },
            ]
        )
    return suppliers, products


def _task_spec(split: str, world_id: str, task_type: str) -> dict[str, Any]:
    category = _world_category(world_id)
    task_id = f"M4-LH-{split.upper()}-{world_id}-{task_type.upper()}"
    workflow_requirements: dict[str, Any] = {}
    if task_type == "exact_product":
        constraints = {"category": category, "keyword": f"M4-LH-{world_id}-EXACT"}
        instruction = (
            f"For procurement world {world_id}, locate model M4-LH-{world_id}-EXACT "
            f"in the {category} catalogue and submit that exact device."
        )
    elif task_type == "cheapest_feasible":
        constraints = {
            "category": category,
            "keyword": f"M4-LH-{world_id}",
            "min_memory_gb": 32,
            "max_delivery_days": 8,
            "certified_only": True,
            "in_stock_only": True,
            "min_warranty_months": 24,
        }
        instruction = (
            f"For procurement world {world_id}, search the {category} catalogue for "
            f"M4-LH-{world_id}. Among in-stock certified devices with at least 32GB "
            "memory, delivery within 8 days, and at least 24 months warranty, submit "
            "the lowest-price option."
        )
    elif task_type == "highest_reliability_supplier":
        constraints = {
            "category": category,
            "keyword": f"M4-LH-{world_id}",
            "min_memory_gb": 32,
            "max_delivery_days": 15,
            "certified_only": True,
            "in_stock_only": True,
            "min_warranty_months": 24,
        }
        workflow_requirements = {
            "required_supplier_detail_ids": [
                _supplier_id(world_id, role)
                for role in SUPPLIER_VISIT_ROLES
            ]
        }
        instruction = (
            f"For procurement world {world_id}, search the {category} catalogue for "
            f"M4-LH-{world_id}. Keep in-stock certified devices with at least 32GB "
            "memory, delivery within 15 days, and at least 24 months warranty. Open "
            "the supplier detail page for every feasible result, compare the delivery "
            "reliability shown there, return to the catalogue after each inspection, "
            "and submit the product from the most reliable supplier."
        )
    elif task_type == "no_feasible_product":
        constraints = {
            "category": category,
            "keyword": f"M4-LH-{world_id}",
            "min_memory_gb": 256,
            "max_delivery_days": 7,
            "certified_only": True,
            "in_stock_only": True,
        }
        instruction = (
            f"For procurement world {world_id}, search the {category} catalogue for "
            f"M4-LH-{world_id}. Find an in-stock certified device with at least 256GB "
            "memory and delivery within 7 days. If none satisfies every requirement, "
            "submit no_solution."
        )
    else:
        raise ValueError(f"unsupported long-horizon task type: {task_type}")
    return {
        "task_id": task_id,
        "split": split,
        "world_id": world_id,
        "task_type": task_type,
        "task_family": task_type,
        "instruction": instruction,
        "start_path": "/products",
        "constraints": constraints,
        "objective": task_type,
        "workflow_requirements": workflow_requirements,
    }


def build_long_horizon_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    world_index = 1
    for split, world_count in SPLIT_WORLD_COUNTS.items():
        for _ in range(world_count):
            world_id = _world_id(world_index)
            specs.extend(_task_spec(split, world_id, task_type) for task_type in TASK_TYPES)
            world_index += 1
    return specs


def _action(action: str, target_testid: str = "", **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"action": action}
    if target_testid:
        result["target_testid"] = target_testid
    result.update(values)
    return result


def build_reference_trace(spec: dict[str, Any], answer: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical minimal scripted path under the task contract."""

    constraints = spec["constraints"]
    trace: list[dict[str, Any]] = []
    field_actions = (
        ("keyword", "search-query", "fill"),
        ("category", "filter-category", "select"),
        ("max_price", "filter-max-price", "fill"),
        ("min_memory_gb", "filter-min-memory", "fill"),
        ("max_delivery_days", "filter-max-delivery", "fill"),
        ("min_supplier_rating", "filter-min-rating", "fill"),
        ("min_warranty_months", "filter-min-warranty", "fill"),
        ("supplier_region", "filter-region", "select"),
    )
    for key, testid, action_type in field_actions:
        if constraints.get(key) is not None:
            trace.append(_action(action_type, testid, value=str(constraints[key])))
    if constraints.get("certified_only") is not None:
        trace.append(
            _action(
                "select",
                "filter-certified",
                value="1" if constraints["certified_only"] else "0",
            )
        )
    if constraints.get("in_stock_only"):
        trace.append(_action("check", "filter-in-stock", checked=True))
    trace.append(_action("click", "apply-filters"))
    for supplier_id in spec.get("workflow_requirements", {}).get(
        "required_supplier_detail_ids", []
    ):
        trace.append(_action("click", f"supplier-link-{supplier_id}"))
        trace.append(_action("back"))
    if answer["expected_decision_type"] == "no_solution":
        trace.append(_action("click", "declare-no-solution"))
    else:
        trace.append(
            _action("click", f"product-link-{answer['expected_product_id']}")
        )
        trace.append(_action("click", "select-product"))
    trace.append(
        _action(
            "fill",
            "procurement-justification",
            value="根据任务要求完成采购。",
        )
    )
    trace.append(_action("click", "submit-procurement"))
    return trace


def _public_record(spec: dict[str, Any], horizon: int, stratum: str) -> dict[str, Any]:
    return {
        "task_id": spec["task_id"],
        "instruction": spec["instruction"],
        "start_path": spec["start_path"],
        "task_type": spec["task_type"],
        "task_family": spec["task_family"],
        "horizon_stratum": stratum,
        "oracle_min_env_actions": horizon,
    }


def _oracle_record(
    spec: dict[str, Any],
    answer: dict[str, Any],
    trace: list[dict[str, Any]],
    stratum: str,
) -> dict[str, Any]:
    return {
        "task_id": spec["task_id"],
        "task_type": spec["task_type"],
        "task_family": spec["task_family"],
        "world_id": spec["world_id"],
        "constraints": spec["constraints"],
        "objective": spec["objective"],
        "workflow_requirements": spec["workflow_requirements"],
        "expected_decision_type": answer["expected_decision_type"],
        "expected_product_id": answer["expected_product_id"],
        "feasible_count": int(answer["feasible_count"]),
        "oracle_min_env_actions": len(trace),
        "horizon_stratum": stratum,
        "reference_trace": trace,
        "oracle_trace_sha256": _signature(trace),
        "explanation": "Deterministic focused-study world and browser workflow contract.",
    }


def _seed_manifest(
    suppliers: list[dict[str, Any]], products: list[dict[str, Any]]
) -> dict[str, Any]:
    supplier_text = _json_text(suppliers)
    product_text = _json_text(products)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed_version": DATASET_ID,
        "supplier_count": len(suppliers),
        "product_count": len(products),
        "supplier_scope": "one supplier set per world; no cross-split supplier ids",
        "files": {
            "suppliers.json": {
                "sha256": _sha256_bytes(supplier_text.encode("utf-8"))
            },
            "products.json": {
                "sha256": _sha256_bytes(product_text.encode("utf-8"))
            },
        },
    }


def _world_records(
    world_id: str,
    suppliers: list[dict[str, Any]],
    products: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    world_suppliers = [
        supplier
        for supplier in suppliers
        if supplier["supplier_id"].startswith(f"M4-LH-SUP-{world_id}-")
    ]
    world_products = [
        product
        for product in products
        if product["product_id"].startswith(f"M4-LH-PRD-{world_id}-")
    ]
    if len(world_suppliers) != 4 or len(world_products) != 4:
        raise ValueError(f"incomplete isolated world: {world_id}")
    return world_suppliers, world_products


def _correct_supplier_visit_position(
    spec: dict[str, Any],
    answer: dict[str, Any],
    world_products: list[dict[str, Any]],
) -> int:
    """Locate the hidden correct choice in the public inspection order."""

    expected_product_id = answer.get("expected_product_id")
    selected = next(
        (
            product
            for product in world_products
            if product["product_id"] == expected_product_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"expected product not found: {expected_product_id}")
    required = spec.get("workflow_requirements", {}).get(
        "required_supplier_detail_ids", []
    )
    try:
        return required.index(selected["supplier_id"]) + 1
    except ValueError as exc:
        raise ValueError(
            f"correct supplier missing from inspection order: {spec['task_id']}"
        ) from exc


def build_long_horizon_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    seed_dir: Path = DEFAULT_SEED_DIR,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    seed_dir = Path(seed_dir).expanduser().resolve()
    suppliers, products = build_long_horizon_seed()
    seed_manifest = _seed_manifest(suppliers, products)
    _atomic_write(seed_dir / "suppliers.json", _json_text(suppliers))
    _atomic_write(seed_dir / "products.json", _json_text(products))
    _atomic_write(seed_dir / "manifest.json", _json_text(seed_manifest))

    split_records = {
        split: {"public": [], "oracle": [], "audit": []}
        for split in SPLIT_WORLD_COUNTS
    }
    decision_position_counts = {
        split: Counter() for split in SPLIT_WORLD_COUNTS
    }
    winner_role_counts = {split: Counter() for split in SPLIT_WORLD_COUNTS}
    modulo_agreement_counts = {split: 0 for split in SPLIT_WORLD_COUNTS}
    all_specs: list[dict[str, Any]] = []
    for spec in build_long_horizon_specs():
        world_suppliers, world_products = _world_records(
            spec["world_id"], suppliers, products
        )
        answer = compute_unique_answer(
            world_products,
            world_suppliers,
            spec["constraints"],
            spec["objective"],
        )
        if answer is None:
            raise ValueError(f"task is not uniquely solvable: {spec['task_id']}")
        correct_supplier_visit_position = None
        if spec["workflow_requirements"]:
            feasible = sorted(
                filter_products(world_products, world_suppliers, spec["constraints"]),
                key=lambda product: (product["price"], product["product_id"]),
            )
            expected_suppliers = [product["supplier_id"] for product in feasible]
            if expected_suppliers != spec["workflow_requirements"][
                "required_supplier_detail_ids"
            ]:
                raise ValueError(f"workflow supplier order drift: {spec['task_id']}")
            correct_supplier_visit_position = _correct_supplier_visit_position(
                spec, answer, world_products
            )
            decision_position_counts[spec["split"]][
                correct_supplier_visit_position
            ] += 1
            winner_role = answer["expected_product_id"].rsplit("-", 1)[-1]
            winner_role_counts[spec["split"]][winner_role] += 1
            simple_periodic_role = FEASIBLE_SUPPLIER_ROLES[
                (int(spec["world_id"][1:]) - 1) % len(FEASIBLE_SUPPLIER_ROLES)
            ]
            if winner_role == simple_periodic_role:
                modulo_agreement_counts[spec["split"]] += 1
        trace = build_reference_trace(spec, answer)
        stratum = classify_horizon(len(trace))
        public = _public_record(spec, len(trace), stratum)
        oracle = _oracle_record(spec, answer, trace, stratum)
        product_ids = sorted(product["product_id"] for product in world_products)
        supplier_ids = sorted(supplier["supplier_id"] for supplier in world_suppliers)
        audit = {
            **spec,
            **answer,
            "oracle_min_env_actions": len(trace),
            "horizon_stratum": stratum,
            "oracle_trace_sha256": oracle["oracle_trace_sha256"],
            "world_signature": _signature(spec["world_id"]),
            "product_signature": _signature(product_ids),
            "supplier_signature": _signature(supplier_ids),
            "constraint_signature": _signature(
                {
                    "objective": spec["objective"],
                    "constraints": spec["constraints"],
                    "workflow": spec["workflow_requirements"],
                }
            ),
            "answer_signature": _signature(
                {
                    "world_id": spec["world_id"],
                    "decision": answer["expected_decision_type"],
                    "product_id": answer["expected_product_id"],
                }
            ),
        }
        if correct_supplier_visit_position is not None:
            audit["correct_supplier_visit_position"] = correct_supplier_visit_position
        records = split_records[spec["split"]]
        records["public"].append(public)
        records["oracle"].append(oracle)
        records["audit"].append(audit)
        all_specs.append(audit)

    horizon_audit = audit_horizon_dataset(
        {split: records["audit"] for split, records in split_records.items()}
    )
    split_files: dict[str, Any] = {}
    for split, records in split_records.items():
        split_dir = output_dir / split
        public_text = _jsonl_text(records["public"])
        oracle_text = _jsonl_text(records["oracle"])
        public_path = split_dir / f"{split}_public.jsonl"
        oracle_path = split_dir / f"{split}_oracle.jsonl"
        split_manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "split": split,
            "role": SPLIT_ROLES[split],
            "may_update_model": split == "train",
            "allowed_purposes": sorted(SPLIT_PURPOSES[split]),
            "world_count": SPLIT_WORLD_COUNTS[split],
            "task_count": len(records["public"]),
            "task_type_counts": dict(
                sorted(Counter(row["task_type"] for row in records["public"]).items())
            ),
            "horizon_counts": horizon_audit["splits"][split]["horizon_counts"],
            "medium_long_fraction": horizon_audit["splits"][split][
                "medium_long_fraction"
            ],
            "public_sha256": _sha256_bytes(public_text.encode("utf-8")),
            "oracle_sha256": _sha256_bytes(oracle_text.encode("utf-8")),
            "seed_products_sha256": seed_manifest["files"]["products.json"]["sha256"],
            "seed_suppliers_sha256": seed_manifest["files"]["suppliers.json"]["sha256"],
        }
        _atomic_write(public_path, public_text)
        _atomic_write(oracle_path, oracle_text)
        _atomic_write(split_dir / SPLIT_MANIFEST_FILENAME, _json_text(split_manifest))
        split_files[split] = {
            "public_sha256": split_manifest["public_sha256"],
            "oracle_sha256": split_manifest["oracle_sha256"],
            "split_manifest_sha256": _signature(split_manifest),
        }

    spec_text = _jsonl_text(all_specs)
    dataset_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "world_count": sum(SPLIT_WORLD_COUNTS.values()),
        "task_count": len(all_specs),
        "split_world_counts": SPLIT_WORLD_COUNTS,
        "split_task_counts": {
            split: count * len(TASK_TYPES)
            for split, count in SPLIT_WORLD_COUNTS.items()
        },
        "task_types": list(TASK_TYPES),
        "horizon_audit": horizon_audit,
        "spec_sha256": _sha256_bytes(spec_text.encode("utf-8")),
        "seed_manifest_sha256": _signature(seed_manifest),
        "split_files": split_files,
        "isolation_contract": {
            "worlds_disjoint_across_splits": True,
            "product_ids_disjoint_across_splits": True,
            "supplier_ids_disjoint_across_splits": True,
            "constraint_signatures_disjoint_across_splits": True,
            "answer_signatures_disjoint_across_splits": True,
            "test_role": "frozen_final_evaluation",
        },
        "shortcut_audit": {
            "contract": "correct long-task supplier must be balanced across visit positions 1,2,3",
            "neutral_feasible_supplier_roles": list(FEASIBLE_SUPPLIER_ROLES),
            "supplier_visit_roles": list(SUPPLIER_VISIT_ROLES),
            "winner_assignment_version": WINNER_ASSIGNMENT_VERSION,
            "winner_role_counts": {
                split: {
                    role: winner_role_counts[split][role]
                    for role in FEASIBLE_SUPPLIER_ROLES
                }
                for split in SPLIT_WORLD_COUNTS
            },
            "correct_supplier_visit_position_counts": {
                split: {
                    str(position): decision_position_counts[split][position]
                    for position in (1, 2, 3)
                }
                for split in SPLIT_WORLD_COUNTS
            },
            "balanced": all(
                decision_position_counts[split][position]
                == SPLIT_WORLD_COUNTS[split] // 3
                for split in SPLIT_WORLD_COUNTS
                for position in (1, 2, 3)
            ),
            "world_index_modulo3_agreement_counts": modulo_agreement_counts,
            "world_index_modulo3_agreement_fraction": {
                split: modulo_agreement_counts[split] / SPLIT_WORLD_COUNTS[split]
                for split in SPLIT_WORLD_COUNTS
            },
            "maximum_allowed_modulo3_agreement_fraction": 0.5,
            "non_periodic": all(
                modulo_agreement_counts[split] / SPLIT_WORLD_COUNTS[split] <= 0.5
                for split in SPLIT_WORLD_COUNTS
            ),
        },
    }
    _atomic_write(output_dir / "spec.jsonl", spec_text)
    _atomic_write(
        output_dir / DATASET_MANIFEST_FILENAME,
        _json_text(dataset_manifest),
    )
    return dataset_manifest


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_long_horizon_split_purpose(task_dir: Path, purpose: str) -> dict[str, Any]:
    manifest_path = Path(task_dir).expanduser().resolve() / SPLIT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"long-horizon split manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError(f"unexpected long-horizon dataset id: {manifest.get('dataset_id')!r}")
    allowed = set(manifest.get("allowed_purposes", []))
    if purpose not in allowed:
        raise PermissionError(
            f"split {manifest.get('split')!r} cannot be used for {purpose!r}; "
            f"allowed={sorted(allowed)}"
        )
    return manifest


def validate_long_horizon_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    seed_dir: Path = DEFAULT_SEED_DIR,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    seed_dir = Path(seed_dir).expanduser().resolve()
    errors: list[str] = []
    try:
        manifest = _read_json(output_dir / DATASET_MANIFEST_FILENAME)
        seed_manifest = _read_json(seed_dir / "manifest.json")
        products = _read_json(seed_dir / "products.json")
        suppliers = _read_json(seed_dir / "suppliers.json")
        specs = _read_jsonl(output_dir / "spec.jsonl")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [str(exc)]}
    if manifest.get("dataset_id") != DATASET_ID:
        errors.append("dataset id mismatch")
    if manifest.get("split_task_counts") != {
        split: count * len(TASK_TYPES)
        for split, count in SPLIT_WORLD_COUNTS.items()
    }:
        errors.append("split task counts mismatch")
    if len(suppliers) != 4 * sum(SPLIT_WORLD_COUNTS.values()):
        errors.append("supplier count mismatch")
    if len(products) != 4 * sum(SPLIT_WORLD_COUNTS.values()):
        errors.append("product count mismatch")
    for filename in ("suppliers.json", "products.json"):
        path = seed_dir / filename
        expected = seed_manifest.get("files", {}).get(filename, {}).get("sha256")
        actual = _sha256_bytes(path.read_bytes()) if path.is_file() else ""
        if expected != actual:
            errors.append(f"seed hash mismatch: {filename}")
    if manifest.get("spec_sha256") != _sha256_bytes(
        (output_dir / "spec.jsonl").read_bytes()
    ):
        errors.append("spec hash mismatch")

    rows_by_split = {split: [] for split in SPLIT_WORLD_COUNTS}
    decision_position_counts = {
        split: Counter() for split in SPLIT_WORLD_COUNTS
    }
    winner_role_counts = {split: Counter() for split in SPLIT_WORLD_COUNTS}
    modulo_agreement_counts = {split: 0 for split in SPLIT_WORLD_COUNTS}
    for spec in specs:
        split = spec.get("split")
        if split not in rows_by_split:
            errors.append(f"unknown split in spec: {split}")
            continue
        rows_by_split[split].append(spec)
        world_suppliers, world_products = _world_records(
            spec["world_id"], suppliers, products
        )
        answer = compute_unique_answer(
            world_products,
            world_suppliers,
            spec["constraints"],
            spec["objective"],
        )
        if answer is None or any(
            answer.get(key) != spec.get(key)
            for key in (
                "expected_decision_type",
                "expected_product_id",
                "feasible_count",
            )
        ):
            errors.append(f"answer does not recompute: {spec.get('task_id')}")
        if answer is not None and spec.get("workflow_requirements"):
            try:
                position = _correct_supplier_visit_position(
                    spec, answer, world_products
                )
                decision_position_counts[split][position] += 1
                winner_role = answer["expected_product_id"].rsplit("-", 1)[-1]
                winner_role_counts[split][winner_role] += 1
                simple_periodic_role = FEASIBLE_SUPPLIER_ROLES[
                    (int(spec["world_id"][1:]) - 1)
                    % len(FEASIBLE_SUPPLIER_ROLES)
                ]
                if winner_role == simple_periodic_role:
                    modulo_agreement_counts[split] += 1
                if spec.get("correct_supplier_visit_position") != position:
                    errors.append(
                        f"correct supplier position drift: {spec.get('task_id')}"
                    )
            except ValueError as exc:
                errors.append(str(exc))
        trace = build_reference_trace(spec, answer or {}) if answer else []
        if len(trace) != spec.get("oracle_min_env_actions"):
            errors.append(f"reference horizon drift: {spec.get('task_id')}")
        if _signature(trace) != spec.get("oracle_trace_sha256"):
            errors.append(f"reference trace hash drift: {spec.get('task_id')}")
    try:
        horizon_audit = audit_horizon_dataset(rows_by_split)
    except ValueError as exc:
        errors.append(str(exc))
        horizon_audit = None

    expected_position_counts = {
        split: {str(position): SPLIT_WORLD_COUNTS[split] // 3 for position in (1, 2, 3)}
        for split in SPLIT_WORLD_COUNTS
    }
    actual_position_counts = {
        split: {
            str(position): decision_position_counts[split][position]
            for position in (1, 2, 3)
        }
        for split in SPLIT_WORLD_COUNTS
    }
    if actual_position_counts != expected_position_counts:
        errors.append("correct supplier visit positions are not balanced")
    shortcut_audit = manifest.get("shortcut_audit", {})
    if shortcut_audit.get("correct_supplier_visit_position_counts") != actual_position_counts:
        errors.append("shortcut audit position counts mismatch")
    if shortcut_audit.get("balanced") is not True:
        errors.append("shortcut audit is not balanced")
    expected_role_counts = {
        split: {
            role: SPLIT_WORLD_COUNTS[split] // len(FEASIBLE_SUPPLIER_ROLES)
            for role in FEASIBLE_SUPPLIER_ROLES
        }
        for split in SPLIT_WORLD_COUNTS
    }
    actual_role_counts = {
        split: {
            role: winner_role_counts[split][role]
            for role in FEASIBLE_SUPPLIER_ROLES
        }
        for split in SPLIT_WORLD_COUNTS
    }
    if actual_role_counts != expected_role_counts:
        errors.append("winner roles are not balanced")
    if shortcut_audit.get("winner_role_counts") != actual_role_counts:
        errors.append("shortcut audit winner role counts mismatch")
    if shortcut_audit.get("winner_assignment_version") != WINNER_ASSIGNMENT_VERSION:
        errors.append("winner assignment version mismatch")
    if shortcut_audit.get("world_index_modulo3_agreement_counts") != modulo_agreement_counts:
        errors.append("shortcut audit modulo agreement counts mismatch")
    if shortcut_audit.get("non_periodic") is not True or any(
        modulo_agreement_counts[split] / SPLIT_WORLD_COUNTS[split] > 0.5
        for split in SPLIT_WORLD_COUNTS
    ):
        errors.append("winner assignment retains a simple world-index modulo shortcut")

    for split, expected_worlds in SPLIT_WORLD_COUNTS.items():
        split_dir = output_dir / split
        try:
            split_manifest = assert_long_horizon_split_purpose(
                split_dir,
                "online_training"
                if split == "train"
                else "model_selection"
                if split == "dev"
                else "final_evaluation",
            )
            public_path = split_dir / f"{split}_public.jsonl"
            oracle_path = split_dir / f"{split}_oracle.jsonl"
            public = _read_jsonl(public_path)
            oracle = _read_jsonl(oracle_path)
        except (FileNotFoundError, ValueError, PermissionError, json.JSONDecodeError) as exc:
            errors.append(f"{split}: {exc}")
            continue
        expected_count = expected_worlds * len(TASK_TYPES)
        if len(public) != expected_count or len(oracle) != expected_count:
            errors.append(f"{split}: task count mismatch")
        if {row.get("task_id") for row in public} != {
            row.get("task_id") for row in oracle
        }:
            errors.append(f"{split}: public/oracle pairing mismatch")
        if split_manifest.get("public_sha256") != _sha256_bytes(public_path.read_bytes()):
            errors.append(f"{split}: public hash mismatch")
        if split_manifest.get("oracle_sha256") != _sha256_bytes(oracle_path.read_bytes()):
            errors.append(f"{split}: oracle hash mismatch")
        if horizon_audit and split_manifest.get("horizon_counts") != horizon_audit[
            "splits"
        ][split]["horizon_counts"]:
            errors.append(f"{split}: horizon count mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "dataset_id": manifest.get("dataset_id"),
        "task_count": manifest.get("task_count"),
        "split_task_counts": manifest.get("split_task_counts"),
        "horizon_audit": horizon_audit,
    }
