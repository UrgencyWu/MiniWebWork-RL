import hashlib
import json
from pathlib import Path

from miniwebwork.data_generation.constraint_contract import compute_unique_answer


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "tasks" / "rollout_dev_feasible_v1"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_feasible_manifest_hashes_and_roles():
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    public_path = DATASET_DIR / "valid_public.jsonl"
    oracle_path = DATASET_DIR / "valid_oracle.jsonl"

    assert manifest["dataset_id"] == "rollout_dev_feasible_v1"
    assert manifest["role"] == "policy_selection_and_regression_gate"
    assert manifest["may_update_model"] is False
    assert manifest["valid_task_count"] == 12
    assert manifest["valid_public_sha256"] == _sha256(public_path)
    assert manifest["valid_oracle_sha256"] == _sha256(oracle_path)
    assert manifest["expected_decision_type_counts"] == {
        "select_product": 12,
        "no_solution": 0,
    }


def test_feasible_oracles_recompute_from_seed_data():
    public = _jsonl(DATASET_DIR / "valid_public.jsonl")
    oracle = _jsonl(DATASET_DIR / "valid_oracle.jsonl")
    products = json.loads((ROOT / "data" / "seed" / "products.json").read_text(encoding="utf-8"))
    suppliers = json.loads((ROOT / "data" / "seed" / "suppliers.json").read_text(encoding="utf-8"))

    public_by_id = {task["task_id"]: task for task in public}
    oracle_by_id = {task["task_id"]: task for task in oracle}
    assert len(public_by_id) == len(public) == 12
    assert len(oracle_by_id) == len(oracle) == 12
    assert set(public_by_id) == set(oracle_by_id)

    for task_id, expected in oracle_by_id.items():
        answer = compute_unique_answer(
            products,
            suppliers,
            expected["constraints"],
            expected["objective"],
        )
        assert answer is not None, task_id
        assert answer["expected_decision_type"] == "select_product"
        assert answer["expected_product_id"] == expected["expected_product_id"]
        assert answer["feasible_count"] == expected["feasible_count"]
        assert public_by_id[task_id]["task_type"] == expected["task_type"]


def test_feasible_public_ids_and_instructions_do_not_overlap_other_public_sources():
    current = _jsonl(DATASET_DIR / "valid_public.jsonl")
    current_ids = {task["task_id"] for task in current}
    current_instructions = {task["instruction"].strip().casefold() for task in current}
    assert len(current_ids) == len(current)
    assert len(current_instructions) == len(current)

    other_ids: set[str] = set()
    other_instructions: set[str] = set()
    for path in (ROOT / "data" / "tasks").rglob("*public.jsonl"):
        if path == DATASET_DIR / "valid_public.jsonl":
            continue
        for task in _jsonl(path):
            other_ids.add(task["task_id"])
            instruction = task.get("instruction")
            if isinstance(instruction, str):
                other_instructions.add(instruction.strip().casefold())

    assert current_ids.isdisjoint(other_ids)
    assert current_instructions.isdisjoint(other_instructions)
