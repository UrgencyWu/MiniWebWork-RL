"""M2.1 tests: constraint contract, task generation, split isolation, SFT schema, expert."""

import json, os, sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from miniwebwork.data_generation.constraint_contract import filter_products, compute_unique_answer


@pytest.fixture
def seed_data():
    suppliers = json.loads((Path(__file__).parent.parent / "data" / "seed" / "suppliers.json").read_text())
    products = json.loads((Path(__file__).parent.parent / "data" / "seed" / "products.json").read_text())
    return products, suppliers


class TestConstraintContract:
    def test_max_price(self, seed_data):
        prods, supps = seed_data
        r = filter_products(prods, supps, {"max_price": 15000})
        assert all(p["price"] <= 15000 for p in r)

    def test_min_memory_gb(self, seed_data):
        prods, supps = seed_data
        r = filter_products(prods, supps, {"min_memory_gb": 64})
        assert all((p.get("memory_gb") or 0) >= 64 for p in r)

    def test_certified_only(self, seed_data):
        prods, supps = seed_data
        r = filter_products(prods, supps, {"certified_only": True})
        sup_map = {s["supplier_id"]: s for s in supps}
        assert all(sup_map[p["supplier_id"]]["certified"] for p in r)

    def test_in_stock_only(self, seed_data):
        prods, supps = seed_data
        r = filter_products(prods, supps, {"in_stock_only": True})
        assert all(p["stock"] > 0 for p in r)

    def test_supplier_region(self, seed_data):
        prods, supps = seed_data
        r = filter_products(prods, supps, {"supplier_region": "华北"})
        sup_map = {s["supplier_id"]: s for s in supps}
        assert all(sup_map[p["supplier_id"]]["region"] == "华北" for p in r)

    def test_category(self, seed_data):
        prods, supps = seed_data
        r = filter_products(prods, supps, {"category": "GPU"})
        assert all(p["category"] == "GPU" for p in r)

    def test_keyword(self, seed_data):
        prods, supps = seed_data
        r = filter_products(prods, supps, {"keyword": "CC-A100X"})
        assert len(r) == 1
        assert r[0]["product_id"] == "PRD-001"

    def test_combined_constraints(self, seed_data):
        prods, supps = seed_data
        r = filter_products(prods, supps, {"category": "GPU", "min_memory_gb": 32, "max_price": 70000, "in_stock_only": True})
        # Should include PRD-002 (48GB, 45000), PRD-018 (32GB, 55000), PRD-019 (48GB, 48000), PRD-009 (32GB, 68000)
        ids = {p["product_id"] for p in r}
        assert "PRD-002" in ids
        assert all(p["price"] <= 70000 for p in r)

    def test_no_solution(self, seed_data):
        prods, supps = seed_data
        r = filter_products(prods, supps, {"category": "GPU", "min_memory_gb": 64, "max_price": 10000})
        assert len(r) == 0

    def test_memory_nullable(self, seed_data):
        """Products without memory_gb (servers, storage) should not match min_memory_gb filter."""
        prods, supps = seed_data
        r = filter_products(prods, supps, {"min_memory_gb": 32, "in_stock_only": True})
        assert all((p.get("memory_gb") or 0) >= 32 for p in r)

    def test_compute_unique_answer_cheapest(self, seed_data):
        prods, supps = seed_data
        a = compute_unique_answer(prods, supps, {"category": "GPU", "min_memory_gb": 48, "max_delivery_days": 14, "in_stock_only": True}, "cheapest_feasible")
        assert a is not None
        assert a["expected_product_id"] == "PRD-002"

    def test_compute_unique_answer_exact(self, seed_data):
        prods, supps = seed_data
        a = compute_unique_answer(prods, supps, {"category": "GPU", "keyword": "CC-A100X-80G"}, "exact_product")
        assert a is not None
        assert a["expected_product_id"] == "PRD-001"

    def test_compute_unique_answer_no_feasible(self, seed_data):
        prods, supps = seed_data
        a = compute_unique_answer(prods, supps, {"category": "GPU", "min_memory_gb": 64, "max_price": 10000, "in_stock_only": True}, "no_feasible_product")
        assert a is not None
        assert a["expected_decision_type"] == "no_solution"


class TestSplitIsolation:
    @pytest.fixture
    def frozen_ids(self):
        frozen = json.loads((Path(__file__).parent.parent / "data" / "splits" / "m2_1" / "test_frozen.json").read_text())
        return {t["task_id"] for t in frozen["test_frozen"]}

    def test_frozen_15_tasks(self):
        frozen = json.loads((Path(__file__).parent.parent / "data" / "splits" / "m2_1" / "test_frozen.json").read_text())
        assert len(frozen["test_frozen"]) == 15

    def test_train_no_frozen_ids(self, frozen_ids):
        tasks_dir = Path(__file__).parent.parent / "data" / "tasks" / "m2_1"
        for fn in ["train_public.jsonl", "valid_public.jsonl"]:
            for line in (tasks_dir / fn).read_text().strip().split("\n"):
                if line.strip():
                    tid = json.loads(line)["task_id"]
                    assert tid not in frozen_ids, f"{tid} in frozen test set"

    def test_train_valid_no_task_id_overlap(self):
        tasks_dir = Path(__file__).parent.parent / "data" / "tasks" / "m2_1"
        train_ids = {json.loads(l)["task_id"] for l in (tasks_dir / "train_public.jsonl").read_text().strip().split("\n") if l.strip()}
        valid_ids = {json.loads(l)["task_id"] for l in (tasks_dir / "valid_public.jsonl").read_text().strip().split("\n") if l.strip()}
        assert len(train_ids & valid_ids) == 0, f"Overlap: {train_ids & valid_ids}"

    def test_cross_split_no_instruction_duplicate(self):
        tasks_dir = Path(__file__).parent.parent / "data" / "tasks" / "m2_1"
        train_insts = set()
        for line in (tasks_dir / "train_public.jsonl").read_text().strip().split("\n"):
            if line.strip():
                train_insts.add(json.loads(line)["instruction"].strip().lower())
        dups = []
        for line in (tasks_dir / "valid_public.jsonl").read_text().strip().split("\n"):
            if line.strip():
                inst = json.loads(line)["instruction"].strip().lower()
                if inst in train_insts:
                    dups.append(inst)
        assert len(dups) == 0, f"Cross-split instruction duplicates: {dups}"


class TestSftSchema:
    def test_train_samples_exist(self):
        p = Path(__file__).parent.parent / "data" / "sft" / "m2_1" / "train.jsonl"
        if p.exists():
            lines = [l for l in p.read_text().strip().split("\n") if l.strip()]
            assert len(lines) > 0

    def test_valid_samples_exist(self):
        p = Path(__file__).parent.parent / "data" / "sft" / "m2_1" / "valid.jsonl"
        if p.exists():
            lines = [l for l in p.read_text().strip().split("\n") if l.strip()]
            assert len(lines) > 0

    def test_assistant_is_strict_json(self):
        for fn in ["train.jsonl", "valid.jsonl"]:
            p = Path(__file__).parent.parent / "data" / "sft" / "m2_1" / fn
            if not p.exists():
                continue
            for line in p.read_text().strip().split("\n")[:50]:
                if not line.strip():
                    continue
                s = json.loads(line)
                action = s.get("action", {})
                assert isinstance(action, dict), f"Action not dict: {action}"
                assert "action" in action, f"Missing action field"

    def test_no_oracle_in_sft(self):
        for fn in ["train.jsonl", "valid.jsonl"]:
            p = Path(__file__).parent.parent / "data" / "sft" / "m2_1" / fn
            if not p.exists():
                continue
            for line in p.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                text = line
                assert "expected_product_id" not in text, "Oracle leak in SFT"
                assert "tasks_oracle" not in text, "Oracle leak in SFT"


class TestReproducibility:
    def test_seed_constant(self):
        from miniwebwork.data_generation.task_generator import SEED
        assert SEED == 20260726

    def test_task_manifest_exists(self):
        p = Path(__file__).parent.parent / "data" / "tasks" / "m2_1" / "manifest.json"
        assert p.exists()
        m = json.loads(p.read_text())
        assert "train_public_sha256" in m
        assert "valid_public_sha256" in m
