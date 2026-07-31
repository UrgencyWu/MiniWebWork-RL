"""Seed data loader and validator for MiniWebWork-RL."""

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_SEED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "seed"
"""The historical M1.1 seed catalogue kept for backwards compatibility."""

SEED_DIR_ENV = "MINIWEBWORK_SEED_DIR"


def get_seed_dir(seed_dir: str | Path | None = None) -> Path:
    """Resolve one explicit, versioned procurement-world catalogue.

    A task source and a product/supplier catalogue are separate provenance
    objects.  Earlier MiniWebWork experiments isolated the former only, which
    made it possible to evaluate a new task on the same products seen during
    training.  M4 passes ``seed_dir`` explicitly (or uses the environment
    override) so an episode always has auditable task *and* world lineage.
    """
    if seed_dir is not None:
        return Path(seed_dir).expanduser().resolve()
    configured = os.environ.get(SEED_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_SEED_DIR


def _load_json(filename: str, seed_dir: str | Path | None = None) -> list:
    path = get_seed_dir(seed_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_suppliers(seed_dir: str | Path | None = None) -> list:
    return _load_json("suppliers.json", seed_dir)


def load_products(seed_dir: str | Path | None = None) -> list:
    return _load_json("products.json", seed_dir)


def seed_database(conn: sqlite3.Connection, seed_dir: str | Path | None = None):
    """Insert seed data into database."""
    suppliers = load_suppliers(seed_dir)
    products = load_products(seed_dir)

    # Insert suppliers
    for s in suppliers:
        conn.execute(
            """INSERT OR REPLACE INTO suppliers
               (supplier_id, name, rating, region, certified, delivery_reliability, description)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                s["supplier_id"],
                s["name"],
                s["rating"],
                s["region"],
                int(s["certified"]),
                s["delivery_reliability"],
                s.get("description", ""),
            ),
        )

    # Insert products
    for p in products:
        conn.execute(
            """INSERT OR REPLACE INTO products
               (product_id, supplier_id, name, category, price, memory_gb,
                delivery_days, stock, warranty_months, model_number, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                p["product_id"],
                p["supplier_id"],
                p["name"],
                p["category"],
                p["price"],
                p.get("memory_gb"),
                p["delivery_days"],
                p["stock"],
                p["warranty_months"],
                p.get("model_number", ""),
                p.get("description", ""),
            ),
        )

    conn.commit()


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def update_manifest(seed_dir: str | Path | None = None):
    """Update manifest.json with computed hashes and counts."""
    resolved_seed_dir = get_seed_dir(seed_dir)
    suppliers = load_suppliers(resolved_seed_dir)
    products = load_products(resolved_seed_dir)

    manifest = {
        "schema_version": "1.0.0",
        "seed_version": "1.0.0",
        "description": "MiniWebWork-RL M1.1 seed data manifest",
        "supplier_count": len(suppliers),
        "product_count": len(products),
        "files": {
            "suppliers.json": {
                "sha256": compute_file_sha256(resolved_seed_dir / "suppliers.json"),
            },
            "products.json": {
                "sha256": compute_file_sha256(resolved_seed_dir / "products.json"),
            },
        },
    }

    manifest_path = resolved_seed_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def validate_seed(seed_dir: str | Path | None = None) -> dict:
    """Validate seed data integrity. Returns dict with validation results."""
    errors = []
    resolved_seed_dir = get_seed_dir(seed_dir)
    suppliers = load_suppliers(resolved_seed_dir)
    products = load_products(resolved_seed_dir)
    manifest_data = {}
    manifest_path = resolved_seed_dir / "manifest.json"

    # Check supplier count
    if len(suppliers) < 6:
        errors.append(f"Expected >= 6 suppliers, got {len(suppliers)}")

    # Check product count
    if len(products) < 24:
        errors.append(f"Expected >= 24 products, got {len(products)}")

    # Check supplier ID uniqueness
    supplier_ids = [s["supplier_id"] for s in suppliers]
    if len(supplier_ids) != len(set(supplier_ids)):
        errors.append("Duplicate supplier IDs found")

    # Check product ID uniqueness
    product_ids = [p["product_id"] for p in products]
    if len(product_ids) != len(set(product_ids)):
        errors.append("Duplicate product IDs found")

    # Check foreign keys
    for p in products:
        if p["supplier_id"] not in supplier_ids:
            errors.append(f"Product {p['product_id']} references unknown supplier {p['supplier_id']}")

    # Check field ranges
    for s in suppliers:
        if not (0 <= s["rating"] <= 5):
            errors.append(f"Supplier {s['supplier_id']}: rating {s['rating']} out of range")
        if s["certified"] not in (0, 1):
            errors.append(f"Supplier {s['supplier_id']}: certified must be 0 or 1")
        if not (0 <= s["delivery_reliability"] <= 1):
            errors.append(f"Supplier {s['supplier_id']}: delivery_reliability out of range")

    for p in products:
        if p["price"] <= 0:
            errors.append(f"Product {p['product_id']}: price must be > 0")
        if p["delivery_days"] < 0:
            errors.append(f"Product {p['product_id']}: delivery_days must be >= 0")
        if p["stock"] < 0:
            errors.append(f"Product {p['product_id']}: stock must be >= 0")
        if p["warranty_months"] < 0:
            errors.append(f"Product {p['product_id']}: warranty_months must be >= 0")

    # Check some products have stock=0
    has_zero_stock = any(p["stock"] == 0 for p in products)
    if not has_zero_stock:
        errors.append("No products with stock=0 found (required by spec)")

    # Verify manifest hashes
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        for fname in ["suppliers.json", "products.json"]:
            expected_hash = manifest_data.get("files", {}).get(fname, {}).get("sha256")
            if expected_hash:
                actual_hash = compute_file_sha256(resolved_seed_dir / fname)
                if actual_hash != expected_hash:
                    errors.append(f"{fname}: hash mismatch (manifest={expected_hash[:16]}..., actual={actual_hash[:16]}...)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "supplier_count": len(suppliers),
        "product_count": len(products),
        "seed_dir": str(resolved_seed_dir),
        "manifest": manifest_data,
    }
