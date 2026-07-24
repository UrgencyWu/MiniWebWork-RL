"""Seed data loader and validator for MiniWebWork-RL."""

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

SEED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "seed"


def _load_json(filename: str) -> list:
    path = SEED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_suppliers() -> list:
    return _load_json("suppliers.json")


def load_products() -> list:
    return _load_json("products.json")


def seed_database(conn: sqlite3.Connection):
    """Insert seed data into database."""
    suppliers = load_suppliers()
    products = load_products()

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


def update_manifest():
    """Update manifest.json with computed hashes and counts."""
    suppliers = load_suppliers()
    products = load_products()

    manifest = {
        "schema_version": "1.0.0",
        "seed_version": "1.0.0",
        "description": "MiniWebWork-RL M1.1 seed data manifest",
        "supplier_count": len(suppliers),
        "product_count": len(products),
        "files": {
            "suppliers.json": {
                "sha256": compute_file_sha256(SEED_DIR / "suppliers.json"),
            },
            "products.json": {
                "sha256": compute_file_sha256(SEED_DIR / "products.json"),
            },
        },
    }

    manifest_path = SEED_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def validate_seed() -> dict:
    """Validate seed data integrity. Returns dict with validation results."""
    errors = []
    suppliers = load_suppliers()
    products = load_products()
    manifest_data = {}
    manifest_path = SEED_DIR / "manifest.json"

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
                actual_hash = compute_file_sha256(SEED_DIR / fname)
                if actual_hash != expected_hash:
                    errors.append(f"{fname}: hash mismatch (manifest={expected_hash[:16]}..., actual={actual_hash[:16]}...)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "supplier_count": len(suppliers),
        "product_count": len(products),
        "manifest": manifest_data,
    }
