"""Deterministic builder for the non-training feasible rollout-dev gate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .constraint_contract import compute_unique_answer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC_PATH = (
    PROJECT_ROOT / "data" / "tasks" / "rollout_dev_feasible_v2" / "spec.jsonl"
)
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "data" / "seed" / "products.json"
DEFAULT_SUPPLIERS_PATH = PROJECT_ROOT / "data" / "seed" / "suppliers.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        values.append(value)
    if not values:
        raise ValueError(f"No records found in {path}")
    return values


def _jsonl_text(values: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        for value in values
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_feasible_rollout_dev(
    output_dir: Path,
    *,
    spec_path: Path = DEFAULT_SPEC_PATH,
    products_path: Path = DEFAULT_PRODUCTS_PATH,
    suppliers_path: Path = DEFAULT_SUPPLIERS_PATH,
) -> dict[str, Any]:
    """Generate Public/Oracle files and a content manifest from a frozen spec.

    The generated dataset is a policy-selection and regression gate. It must
    never be used as an optimizer task source.
    """
    output_dir = output_dir.expanduser().resolve()
    spec_path = spec_path.expanduser().resolve()
    products_path = products_path.expanduser().resolve()
    suppliers_path = suppliers_path.expanduser().resolve()
    for path in (spec_path, products_path, suppliers_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    specs = _read_jsonl(spec_path)
    products = json.loads(products_path.read_text(encoding="utf-8"))
    suppliers = json.loads(suppliers_path.read_text(encoding="utf-8"))
    if not isinstance(products, list) or not isinstance(suppliers, list):
        raise ValueError("Seed products and suppliers must be JSON arrays")

    seen_ids: set[str] = set()
    seen_instructions: set[str] = set()
    public_records: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []

    for spec in sorted(specs, key=lambda item: str(item.get("task_id", ""))):
        task_id = spec.get("task_id")
        instruction = spec.get("instruction")
        task_type = spec.get("task_type")
        constraints = spec.get("constraints")
        objective = spec.get("objective")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Every feasible spec requires a non-empty task_id")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate feasible spec task_id: {task_id}")
        seen_ids.add(task_id)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Task {task_id} requires a non-empty instruction")
        normalized_instruction = instruction.strip().casefold()
        if normalized_instruction in seen_instructions:
            raise ValueError(f"Duplicate feasible instruction: {instruction}")
        seen_instructions.add(normalized_instruction)
        if not isinstance(task_type, str) or not task_type:
            raise ValueError(f"Task {task_id} requires task_type")
        if not isinstance(constraints, dict):
            raise ValueError(f"Task {task_id} requires constraints")
        if not isinstance(objective, str) or not objective:
            raise ValueError(f"Task {task_id} requires objective")

        answer = compute_unique_answer(
            products,
            suppliers,
            constraints,
            objective,
        )
        if answer is None:
            raise ValueError(
                f"Task {task_id} is not a deterministic feasible task under the current seeds"
            )
        if answer.get("expected_decision_type") != "select_product":
            raise ValueError(f"Task {task_id} unexpectedly resolves to no_solution")
        expected_product_id = answer.get("expected_product_id")
        if not isinstance(expected_product_id, str) or not expected_product_id:
            raise ValueError(f"Task {task_id} has no unique expected product")

        public_records.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "start_path": spec.get("start_path", "/products"),
                "task_type": task_type,
            }
        )
        oracle_records.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "constraints": constraints,
                "objective": objective,
                "expected_decision_type": "select_product",
                "expected_product_id": expected_product_id,
                "feasible_count": int(answer["feasible_count"]),
                "explanation": (
                    f"Deterministically generated from feasible spec; "
                    f"expected product {expected_product_id}."
                ),
            }
        )

    public_text = _jsonl_text(public_records)
    oracle_text = _jsonl_text(oracle_records)
    public_path = output_dir / "valid_public.jsonl"
    oracle_path = output_dir / "valid_oracle.jsonl"
    _atomic_write(public_path, public_text)
    _atomic_write(oracle_path, oracle_text)

    manifest = {
        "schema_version": "2.0",
        "dataset_id": "rollout_dev_feasible_v2",
        "role": "policy_selection_and_regression_gate",
        "may_update_model": False,
        "valid_task_count": len(public_records),
        "task_type_counts": dict(sorted(Counter(
            record["task_type"] for record in public_records
        ).items())),
        "expected_decision_type_counts": {"select_product": len(public_records)},
        "spec_sha256": _sha256(spec_path),
        "products_sha256": _sha256(products_path),
        "suppliers_sha256": _sha256(suppliers_path),
        "valid_public_sha256": hashlib.sha256(public_text.encode("utf-8")).hexdigest(),
        "valid_oracle_sha256": hashlib.sha256(oracle_text.encode("utf-8")).hexdigest(),
    }
    _atomic_write(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest
