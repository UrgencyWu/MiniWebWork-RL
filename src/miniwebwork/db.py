"""SQLite persistence for deterministic procurement episodes."""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "runtime" / "miniwebwork.db"
)


def get_db_path() -> str:
    return os.environ.get("MINIWEBWORK_DB_PATH", DEFAULT_DB_PATH)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys, timeout, and Row access."""
    path = db_path or get_db_path()
    connection = sqlite3.connect(path, timeout=30.0)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.row_factory = sqlite3.Row
    return connection


def init_schema(connection: sqlite3.Connection) -> None:
    """Create the deterministic runtime schema and integrity indexes."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS suppliers (
            supplier_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            rating REAL NOT NULL CHECK(rating >= 0 AND rating <= 5),
            region TEXT NOT NULL,
            certified INTEGER NOT NULL CHECK(certified IN (0, 1)),
            delivery_reliability REAL NOT NULL
                CHECK(delivery_reliability >= 0 AND delivery_reliability <= 1),
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            supplier_id TEXT NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL CHECK(price > 0),
            memory_gb INTEGER,
            delivery_days INTEGER NOT NULL CHECK(delivery_days >= 0),
            stock INTEGER NOT NULL CHECK(stock >= 0),
            warranty_months INTEGER NOT NULL CHECK(warranty_months >= 0),
            model_number TEXT,
            description TEXT,
            FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
        );

        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK(status IN ('active', 'submitted', 'verified', 'failed')),
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS procurement_submissions (
            submission_id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            decision_type TEXT NOT NULL
                CHECK(decision_type IN ('select_product', 'no_solution')),
            product_id TEXT,
            quantity INTEGER NOT NULL DEFAULT 1 CHECK(quantity > 0),
            justification TEXT,
            submitted_at TEXT NOT NULL,
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id),
            CHECK (
                (decision_type = 'select_product' AND product_id IS NOT NULL) OR
                (decision_type = 'no_solution' AND product_id IS NULL)
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS
            idx_procurement_submissions_episode
            ON procurement_submissions(episode_id);
        CREATE INDEX IF NOT EXISTS idx_products_supplier
            ON products(supplier_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_task
            ON episodes(task_id);
        """
    )
    connection.commit()


def create_episode(connection: sqlite3.Connection, task_id: str) -> str:
    """Create one active episode for a non-empty public task ID."""
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    episode_id = f"EP-{uuid.uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc).isoformat()
    connection.execute(
        "INSERT INTO episodes (episode_id, task_id, status, created_at) "
        "VALUES (?, ?, ?, ?)",
        (episode_id, task_id, "active", now),
    )
    connection.commit()
    return episode_id


def create_submission(
    connection: sqlite3.Connection,
    episode_id: str,
    task_id: str,
    decision_type: str,
    product_id: Optional[str] = None,
    quantity: int = 1,
    justification: str = "",
) -> str:
    """Persist the single final decision for an active task episode."""
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        raise ValueError("quantity must be a positive integer")
    if not isinstance(justification, str):
        raise ValueError("justification must be a string")

    episode = connection.execute(
        "SELECT * FROM episodes WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()
    if episode is None:
        raise ValueError(f"Episode {episode_id} does not exist")
    if episode["task_id"] != task_id:
        raise ValueError(
            f"Episode task {episode['task_id']} does not match submission task {task_id}"
        )
    if episode["status"] != "active":
        raise ValueError(
            f"Episode {episode_id} is not active (status: {episode['status']})"
        )

    existing = connection.execute(
        "SELECT submission_id FROM procurement_submissions WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()
    if existing is not None:
        raise ValueError(f"Episode {episode_id} already has a submission")

    if decision_type not in {"select_product", "no_solution"}:
        raise ValueError(f"Invalid decision_type: {decision_type}")
    if decision_type == "select_product":
        if not product_id:
            raise ValueError("select_product requires product_id")
        product = connection.execute(
            "SELECT product_id FROM products WHERE product_id = ?",
            (product_id,),
        ).fetchone()
        if product is None:
            raise ValueError(f"Product {product_id} does not exist")
    elif product_id:
        raise ValueError("no_solution must not have product_id")

    submission_id = f"SUB-{uuid.uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc).isoformat()
    try:
        connection.execute(
            """INSERT INTO procurement_submissions
               (submission_id, episode_id, task_id, decision_type, product_id,
                quantity, justification, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                submission_id,
                episode_id,
                task_id,
                decision_type,
                product_id,
                quantity,
                justification,
                now,
            ),
        )
        connection.execute(
            "UPDATE episodes SET status = ?, completed_at = ? WHERE episode_id = ?",
            ("submitted", now, episode_id),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return submission_id


def reset_db(connection: sqlite3.Connection) -> None:
    """Drop and recreate all runtime tables. Seeding is a separate operation."""
    connection.executescript(
        """
        DROP TABLE IF EXISTS procurement_submissions;
        DROP TABLE IF EXISTS episodes;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS suppliers;
        """
    )
    init_schema(connection)
    connection.commit()


def get_row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    """Return row counts for the fixed runtime tables."""
    counts: dict[str, int] = {}
    for table in ("suppliers", "products", "episodes", "procurement_submissions"):
        row = connection.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
        counts[table] = int(row["cnt"])
    return counts
