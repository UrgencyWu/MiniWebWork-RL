"""Deterministic M4 RLVR world and split builder.

M4 is deliberately built from 108 disjoint procurement worlds instead of
paraphrasing the historical task set.  A world owns four products and yields
one task for each supported terminal-verification objective.  Worlds, products,
constraint signatures, instructions and selected-product answers are all
disjoint across train/dev/final-test.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .constraint_contract import compute_unique_answer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tasks" / "m4_rlvr_v1"
DEFAULT_SEED_DIR = PROJECT_ROOT / "data" / "seed_m4_rlvr_v1"
DATASET_ID = "m4_rlvr_v1"
SCHEMA_VERSION = "1.0"
SPLIT_WORLD_COUNTS = {"train": 60, "dev": 18, "test": 30}
TASK_TYPES = (
    "exact_product",
    "cheapest_feasible",
    "highest_rating_supplier",
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
SPLIT_MANIFEST_FILENAME = "m4_split_manifest.json"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
    """Choose only categories exposed by the real browser filter UI."""
    index = int(world_id[1:])
    return ("GPU", "服务器", "存储", "网络")[(index - 1) % 4]


def build_m4_seed() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the fixed M4 supplier/product catalogue without writing files."""
    suppliers = [
        {
            "supplier_id": "M4-SUP-01",
            "name": "M4 Apex Verified Systems",
            "rating": 4.95,
            "region": "North",
            "certified": 1,
            "delivery_reliability": 0.98,
            "description": "High-reliability certified supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-02",
            "name": "M4 Standard Compute",
            "rating": 4.60,
            "region": "East",
            "certified": 1,
            "delivery_reliability": 0.92,
            "description": "Certified general-purpose supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-03",
            "name": "M4 Value Fabric",
            "rating": 4.10,
            "region": "South",
            "certified": 1,
            "delivery_reliability": 0.88,
            "description": "Certified value supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-04",
            "name": "M4 Clearance Exchange",
            "rating": 3.70,
            "region": "West",
            "certified": 0,
            "delivery_reliability": 0.73,
            "description": "Uncertified distractor supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-05",
            "name": "M4 Meridian Systems",
            "rating": 4.72,
            "region": "Central",
            "certified": 1,
            "delivery_reliability": 0.94,
            "description": "Certified regional supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-06",
            "name": "M4 Budget Infrastructure",
            "rating": 4.18,
            "region": "North",
            "certified": 1,
            "delivery_reliability": 0.86,
            "description": "Certified value supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-07",
            "name": "M4 Harbor Compute",
            "rating": 4.05,
            "region": "East",
            "certified": 1,
            "delivery_reliability": 0.84,
            "description": "Certified value supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-08",
            "name": "M4 Southern Components",
            "rating": 3.98,
            "region": "South",
            "certified": 1,
            "delivery_reliability": 0.82,
            "description": "Certified value supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-09",
            "name": "M4 Frontier Logistics",
            "rating": 3.88,
            "region": "West",
            "certified": 1,
            "delivery_reliability": 0.78,
            "description": "Certified value supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-10",
            "name": "M4 Outlet One",
            "rating": 3.65,
            "region": "Central",
            "certified": 0,
            "delivery_reliability": 0.72,
            "description": "Uncertified distractor supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-11",
            "name": "M4 Outlet Two",
            "rating": 3.55,
            "region": "North",
            "certified": 0,
            "delivery_reliability": 0.69,
            "description": "Uncertified distractor supplier for M4 benchmark worlds.",
        },
        {
            "supplier_id": "M4-SUP-12",
            "name": "M4 Outlet Three",
            "rating": 3.45,
            "region": "South",
            "certified": 0,
            "delivery_reliability": 0.66,
            "description": "Uncertified distractor supplier for M4 benchmark worlds.",
        },
    ]
    products: list[dict[str, Any]] = []
    for index in range(1, sum(SPLIT_WORLD_COUNTS.values()) + 1):
        world_id = _world_id(index)
        category = _world_category(world_id)
        base_price = 10_000 + 37 * index
        reference_supplier = f"M4-SUP-{2 + (index - 1) % 4:02d}"
        value_supplier = f"M4-SUP-{6 + (index - 1) % 4:02d}"
        distractor_supplier = f"M4-SUP-{10 + (index - 1) % 3:02d}"
        product_common = {
            "category": category,
            "description": (
                f"Dedicated product for isolated M4 procurement world {world_id}. "
                "Inspect product and supplier details before submitting a decision."
            ),
        }
        products.extend(
            [
                {
                    **product_common,
                    "product_id": f"M4-PRD-{world_id}-E",
                    "supplier_id": reference_supplier,
                    "name": f"Atlas {world_id} Reference Node",
                    "price": float(base_price + 1_400),
                    "memory_gb": 64,
                    "delivery_days": 5,
                    "stock": 9,
                    "warranty_months": 36,
                    "model_number": f"M4-{world_id}-EXACT",
                },
                {
                    **product_common,
                    "product_id": f"M4-PRD-{world_id}-C",
                    "supplier_id": value_supplier,
                    "name": f"Atlas {world_id} Value Node",
                    "price": float(base_price),
                    "memory_gb": 48,
                    "delivery_days": 7,
                    "stock": 15,
                    "warranty_months": 24,
                    "model_number": f"M4-{world_id}-CHEAP",
                },
                {
                    **product_common,
                    "product_id": f"M4-PRD-{world_id}-R",
                    "supplier_id": "M4-SUP-01",
                    "name": f"Atlas {world_id} Resilient Node",
                    "price": float(base_price + 3_800),
                    "memory_gb": 96,
                    "delivery_days": 12,
                    "stock": 7,
                    "warranty_months": 48,
                    "model_number": f"M4-{world_id}-RATED",
                },
                {
                    **product_common,
                    "product_id": f"M4-PRD-{world_id}-D",
                    "supplier_id": distractor_supplier,
                    "name": f"Atlas {world_id} Clearance Node",
                    "price": float(base_price + 2_100),
                    "memory_gb": 24,
                    "delivery_days": 20,
                    "stock": 0,
                    "warranty_months": 12,
                    "model_number": f"M4-{world_id}-DISTRACTOR",
                },
            ]
        )
    return suppliers, products


def _task_spec(split: str, world_id: str, task_type: str) -> dict[str, Any]:
    category = _world_category(world_id)
    task_id = f"M4-{split.upper()}-{world_id}-{task_type.upper()}"
    if task_type == "exact_product":
        constraints = {"category": category, "keyword": f"M4-{world_id}-EXACT"}
        instruction = (
            f"For M4 procurement world {world_id}, locate the device with model "
            f"identifier M4-{world_id}-EXACT in the {category} catalogue. Submit that exact device."
        )
    elif task_type == "cheapest_feasible":
        constraints = {
            "category": category,
            "keyword": f"M4-{world_id}",
            "min_memory_gb": 32,
            "max_delivery_days": 8,
            "certified_only": True,
            "in_stock_only": True,
            "min_warranty_months": 24,
        }
        instruction = (
            f"For M4 procurement world {world_id}, search the {category} catalogue for "
            f"keyword M4-{world_id}. Choose an "
            "in-stock device from a certified supplier with at least 32GB memory, delivery "
            "within 8 days, and at least 24 months warranty. Among feasible devices, submit "
            "the lowest-price option."
        )
    elif task_type == "highest_rating_supplier":
        constraints = {
            "category": category,
            "keyword": f"M4-{world_id}",
            "min_memory_gb": 32,
            "max_delivery_days": 15,
            "certified_only": True,
            "in_stock_only": True,
            "min_warranty_months": 24,
        }
        instruction = (
            f"For M4 procurement world {world_id}, search the {category} catalogue for "
            f"keyword M4-{world_id}. Choose an "
            "in-stock certified device with at least 32GB memory, delivery within 15 days, "
            "and at least 24 months warranty. Among feasible options, submit the one whose "
            "supplier has the highest rating."
        )
    elif task_type == "no_feasible_product":
        constraints = {
            "category": category,
            "keyword": f"M4-{world_id}",
            "min_memory_gb": 256,
            "max_delivery_days": 7,
            "certified_only": True,
            "in_stock_only": True,
        }
        instruction = (
            f"For M4 procurement world {world_id}, search the {category} catalogue for "
            f"keyword M4-{world_id}. Find an "
            "in-stock certified device with at least 256GB memory and delivery within 7 days. "
            "If none satisfies every requirement, submit no_solution rather than guessing."
        )
    else:
        raise ValueError(f"Unsupported M4 task type: {task_type}")
    return {
        "task_id": task_id,
        "split": split,
        "world_id": world_id,
        "task_type": task_type,
        "instruction": instruction,
        "start_path": "/products",
        "constraints": constraints,
        "objective": task_type,
    }


def build_m4_specs() -> list[dict[str, Any]]:
    """Return all M4 split/world/task specifications in canonical order."""
    specs: list[dict[str, Any]] = []
    world_index = 1
    for split, world_count in SPLIT_WORLD_COUNTS.items():
        for _ in range(world_count):
            world_id = _world_id(world_index)
            specs.extend(_task_spec(split, world_id, task_type) for task_type in TASK_TYPES)
            world_index += 1
    return specs


def _public_record(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": spec["task_id"],
        "instruction": spec["instruction"],
        "start_path": spec["start_path"],
        "task_type": spec["task_type"],
    }


def _oracle_record(spec: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": spec["task_id"],
        "task_type": spec["task_type"],
        "world_id": spec["world_id"],
        "constraints": spec["constraints"],
        "objective": spec["objective"],
        "expected_decision_type": answer["expected_decision_type"],
        "expected_product_id": answer["expected_product_id"],
        "feasible_count": int(answer["feasible_count"]),
        "explanation": "Deterministic M4 world contract; recompute against seed_m4_rlvr_v1.",
    }


def _seed_manifest(suppliers: list[dict[str, Any]], products: list[dict[str, Any]]) -> dict[str, Any]:
    suppliers_text = _json_text(suppliers)
    products_text = _json_text(products)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed_version": DATASET_ID,
        "description": "Versioned procurement worlds for the M4 RLVR algorithm study.",
        "supplier_count": len(suppliers),
        "product_count": len(products),
        "files": {
            "suppliers.json": {"sha256": _sha256_bytes(suppliers_text.encode("utf-8"))},
            "products.json": {"sha256": _sha256_bytes(products_text.encode("utf-8"))},
        },
    }


def build_m4_rlvr_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    seed_dir: Path = DEFAULT_SEED_DIR,
) -> dict[str, Any]:
    """Build the checked M4 task splits and their versioned world catalogue."""
    output_dir = output_dir.expanduser().resolve()
    seed_dir = seed_dir.expanduser().resolve()
    suppliers, products = build_m4_seed()
    suppliers_text = _json_text(suppliers)
    products_text = _json_text(products)
    seed_manifest = _seed_manifest(suppliers, products)
    _atomic_write(seed_dir / "suppliers.json", suppliers_text)
    _atomic_write(seed_dir / "products.json", products_text)
    _atomic_write(seed_dir / "manifest.json", _json_text(seed_manifest))

    spec_records: list[dict[str, Any]] = []
    split_records: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {"public": [], "oracle": []} for split in SPLIT_WORLD_COUNTS
    }
    for spec in build_m4_specs():
        answer = compute_unique_answer(
            products, suppliers, spec["constraints"], spec["objective"]
        )
        if answer is None:
            raise ValueError(f"M4 spec is not uniquely solvable: {spec['task_id']}")
        oracle = _oracle_record(spec, answer)
        spec_records.append({**spec, **answer})
        split_records[spec["split"]]["public"].append(_public_record(spec))
        split_records[spec["split"]]["oracle"].append(oracle)

    split_manifest_hashes: dict[str, dict[str, str]] = {}
    for split, records in split_records.items():
        split_dir = output_dir / split
        public_text = _jsonl_text(records["public"])
        oracle_text = _jsonl_text(records["oracle"])
        public_filename = f"{split}_public.jsonl"
        oracle_filename = f"{split}_oracle.jsonl"
        public_hash = _sha256_bytes(public_text.encode("utf-8"))
        oracle_hash = _sha256_bytes(oracle_text.encode("utf-8"))
        split_manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "split": split,
            "role": SPLIT_ROLES[split],
            "may_update_model": split == "train",
            "allowed_purposes": sorted(SPLIT_PURPOSES[split]),
            "world_count": SPLIT_WORLD_COUNTS[split],
            "task_count": len(records["public"]),
            "task_type_counts": dict(sorted(Counter(x["task_type"] for x in records["public"]).items())),
            "public_sha256": public_hash,
            "oracle_sha256": oracle_hash,
            "seed_products_sha256": seed_manifest["files"]["products.json"]["sha256"],
            "seed_suppliers_sha256": seed_manifest["files"]["suppliers.json"]["sha256"],
        }
        _atomic_write(split_dir / public_filename, public_text)
        _atomic_write(split_dir / oracle_filename, oracle_text)
        _atomic_write(split_dir / SPLIT_MANIFEST_FILENAME, _json_text(split_manifest))
        split_manifest_hashes[split] = {
            "public_sha256": public_hash,
            "oracle_sha256": oracle_hash,
            "split_manifest_sha256": _sha256_bytes(_json_text(split_manifest).encode("utf-8")),
        }

    spec_text = _jsonl_text(spec_records)
    dataset_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "world_count": sum(SPLIT_WORLD_COUNTS.values()),
        "task_count": len(spec_records),
        "split_world_counts": SPLIT_WORLD_COUNTS,
        "split_task_counts": {
            split: SPLIT_WORLD_COUNTS[split] * len(TASK_TYPES)
            for split in SPLIT_WORLD_COUNTS
        },
        "split_roles": SPLIT_ROLES,
        "task_types": list(TASK_TYPES),
        "task_type_counts_per_split": {
            split: {task_type: SPLIT_WORLD_COUNTS[split] for task_type in TASK_TYPES}
            for split in SPLIT_WORLD_COUNTS
        },
        "spec_sha256": _sha256_bytes(spec_text.encode("utf-8")),
        "seed_manifest_sha256": _sha256_bytes(_json_text(seed_manifest).encode("utf-8")),
        "split_files": split_manifest_hashes,
        "isolation_contract": {
            "worlds_disjoint_across_splits": True,
            "constraint_signatures_disjoint_across_splits": True,
            "selected_product_answers_disjoint_across_splits": True,
            "public_instructions_disjoint_across_splits": True,
            "test_role": "frozen_final_evaluation",
        },
    }
    _atomic_write(output_dir / "spec.jsonl", spec_text)
    _atomic_write(output_dir / DATASET_MANIFEST_FILENAME, _json_text(dataset_manifest))
    return dataset_manifest


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_m4_split_purpose(task_dir: Path, purpose: str) -> dict[str, Any]:
    """Reject accidental optimizer/test leakage before an experiment begins."""
    manifest_path = Path(task_dir).expanduser().resolve() / SPLIT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"M4 split manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError(f"Unexpected M4 dataset id: {manifest.get('dataset_id')!r}")
    allowed = set(manifest.get("allowed_purposes", []))
    if purpose not in allowed:
        raise PermissionError(
            f"M4 split {manifest.get('split')!r} ({manifest.get('role')!r}) "
            f"cannot be used for {purpose!r}; allowed={sorted(allowed)}"
        )
    return manifest


def validate_m4_rlvr_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    seed_dir: Path = DEFAULT_SEED_DIR,
) -> dict[str, Any]:
    """Verify counts, deterministic answers, hashes and cross-split isolation."""
    output_dir = output_dir.expanduser().resolve()
    seed_dir = seed_dir.expanduser().resolve()
    errors: list[str] = []
    try:
        manifest = _read_json(output_dir / DATASET_MANIFEST_FILENAME)
        seed_manifest = _read_json(seed_dir / "manifest.json")
        products = _read_json(seed_dir / "products.json")
        suppliers = _read_json(seed_dir / "suppliers.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [str(exc)]}

    if manifest.get("split_world_counts") != SPLIT_WORLD_COUNTS:
        errors.append("split_world_counts does not match the M4 contract")
    if manifest.get("split_task_counts") != {
        split: count * len(TASK_TYPES) for split, count in SPLIT_WORLD_COUNTS.items()
    }:
        errors.append("split_task_counts does not match the M4 contract")
    for filename in ("suppliers.json", "products.json"):
        path = seed_dir / filename
        expected_hash = seed_manifest.get("files", {}).get(filename, {}).get("sha256")
        actual_hash = _sha256_bytes(path.read_bytes()) if path.is_file() else ""
        if expected_hash != actual_hash:
            errors.append(f"seed hash mismatch: {filename}")

    all_worlds: dict[str, set[str]] = {}
    all_signatures: dict[str, set[str]] = {}
    all_answers: dict[str, set[str]] = {}
    all_instructions: dict[str, set[str]] = {}
    all_task_ids: set[str] = set()
    for split, world_count in SPLIT_WORLD_COUNTS.items():
        split_dir = output_dir / split
        try:
            split_manifest = assert_m4_split_purpose(
                split_dir,
                "online_training" if split == "train" else "model_selection" if split == "dev" else "final_evaluation",
            )
            public_path = split_dir / f"{split}_public.jsonl"
            oracle_path = split_dir / f"{split}_oracle.jsonl"
            public = _read_jsonl(public_path)
            oracle = _read_jsonl(oracle_path)
        except (FileNotFoundError, ValueError, PermissionError, json.JSONDecodeError) as exc:
            errors.append(f"{split}: {exc}")
            continue
        if len(public) != world_count * len(TASK_TYPES) or len(oracle) != len(public):
            errors.append(f"{split}: wrong task count")
        if split_manifest.get("task_count") != len(public):
            errors.append(f"{split}: split manifest task count mismatch")
        if split_manifest.get("public_sha256") != _sha256_bytes(public_path.read_bytes()):
            errors.append(f"{split}: public hash mismatch")
        if split_manifest.get("oracle_sha256") != _sha256_bytes(oracle_path.read_bytes()):
            errors.append(f"{split}: oracle hash mismatch")
        public_by_id = {item.get("task_id"): item for item in public}
        oracle_by_id = {item.get("task_id"): item for item in oracle}
        if len(public_by_id) != len(public) or set(public_by_id) != set(oracle_by_id):
            errors.append(f"{split}: public/oracle task pairing is invalid")
        task_types = Counter(item.get("task_type") for item in public)
        if task_types != Counter({task_type: world_count for task_type in TASK_TYPES}):
            errors.append(f"{split}: task types are not balanced")
        worlds = {str(item.get("world_id")) for item in oracle}
        if len(worlds) != world_count:
            errors.append(f"{split}: wrong unique world count")
        signatures: set[str] = set()
        answers: set[str] = set()
        instructions: set[str] = set()
        for task_id, expected in oracle_by_id.items():
            if task_id in all_task_ids:
                errors.append(f"duplicate task id: {task_id}")
            all_task_ids.add(str(task_id))
            answer = compute_unique_answer(
                products,
                suppliers,
                expected.get("constraints", {}),
                expected.get("objective", ""),
            )
            if answer is None or any(
                answer.get(key) != expected.get(key)
                for key in ("expected_decision_type", "expected_product_id", "feasible_count")
            ):
                errors.append(f"{task_id}: oracle does not recompute from M4 seed")
            signature = json.dumps(
                {"objective": expected.get("objective"), "constraints": expected.get("constraints")},
                sort_keys=True,
                separators=(",", ":"),
            )
            signatures.add(signature)
            product_id = expected.get("expected_product_id")
            if product_id:
                answers.add(str(product_id))
            instruction = public_by_id.get(task_id, {}).get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                errors.append(f"{task_id}: missing public instruction")
            else:
                instructions.add(instruction.strip().casefold())
        if len(signatures) != len(oracle):
            errors.append(f"{split}: duplicate constraint signatures")
        if len(answers) != world_count * 3:
            errors.append(f"{split}: selected answers are not unique")
        if len(instructions) != len(public):
            errors.append(f"{split}: duplicate instructions")
        all_worlds[split] = worlds
        all_signatures[split] = signatures
        all_answers[split] = answers
        all_instructions[split] = instructions

    split_names = tuple(SPLIT_WORLD_COUNTS)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            if all_worlds.get(first, set()) & all_worlds.get(second, set()):
                errors.append(f"world leakage between {first} and {second}")
            if all_signatures.get(first, set()) & all_signatures.get(second, set()):
                errors.append(f"constraint leakage between {first} and {second}")
            if all_answers.get(first, set()) & all_answers.get(second, set()):
                errors.append(f"answer leakage between {first} and {second}")
            if all_instructions.get(first, set()) & all_instructions.get(second, set()):
                errors.append(f"instruction leakage between {first} and {second}")

    spec_path = output_dir / "spec.jsonl"
    if not spec_path.is_file() or manifest.get("spec_sha256") != _sha256_bytes(spec_path.read_bytes()):
        errors.append("spec hash mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "dataset_id": manifest.get("dataset_id"),
        "world_count": manifest.get("world_count"),
        "task_count": manifest.get("task_count"),
        "split_task_counts": manifest.get("split_task_counts"),
    }
