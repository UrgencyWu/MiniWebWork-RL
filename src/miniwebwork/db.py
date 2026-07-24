"""SQLite database layer for MiniWebWork-RL procurement environment."""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Default database path
DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "runtime" / "miniwebwork.db"
)


def get_db_path() -> str:
    return os.environ.get("MINIWEBWORK_DB_PATH", DEFAULT_DB_PATH)


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Get a database connection with foreign keys enabled."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection):
    """Create all tables if they do not exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS suppliers (
            supplier_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            rating REAL NOT NULL CHECK(rating >= 0 AND rating <= 5),
            region TEXT NOT NULL,
            certified INTEGER NOT NULL CHECK(certified IN (0, 1)),
            delivery_reliability REAL NOT NULL CHECK(delivery_reliability >= 0 AND delivery_reliability <= 1),
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
            status TEXT NOT NULL CHECK(status IN ('active', 'submitted', 'verified', 'failed')),
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS procurement_submissions (
            submission_id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            decision_type TEXT NOT NULL CHECK(decision_type IN ('select_product', 'no_solution')),
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
    """)


def create_episode(conn: sqlite3.Connection, task_id: str) -> str:
    """Create a new episode and return its ID."""
    episode_id = f"EP-{uuid.uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO episodes (episode_id, task_id, status, created_at) VALUES (?, ?, ?, ?)",
        (episode_id, task_id, "active", now),
    )
    conn.commit()
    return episode_id


def create_submission(
    conn: sqlite3.Connection,
    episode_id: str,
    task_id: str,
    decision_type: str,
    product_id: str = None,
    quantity: int = 1,
    justification: str = "",
) -> str:
    """Create a procurement submission. Returns submission_id."""
    # Validate episode exists and matches task
    episode = conn.execute(
        "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    if episode is None:
        raise ValueError(f"Episode {episode_id} does not exist")
    if episode["task_id"] != task_id:
        raise ValueError(
            f"Episode task {episode['task_id']} does not match submission task {task_id}"
        )
    if episode["status"] != "active":
        raise ValueError(f"Episode {episode_id} is not active (status: {episode['status']})")

    # Check for duplicate submission
    existing = conn.execute(
        "SELECT submission_id FROM procurement_submissions WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()
    if existing:
        raise ValueError(f"Episode {episode_id} already has a submission")

    # Validate decision_type and product_id
    if decision_type not in ("select_product", "no_solution"):
        raise ValueError(f"Invalid decision_type: {decision_type}")

    if decision_type == "select_product":
        if not product_id:
            raise ValueError("select_product requires product_id")
        prod = conn.execute(
            "SELECT product_id FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
        if prod is None:
            raise ValueError(f"Product {product_id} does not exist")
    elif decision_type == "no_solution":
        if product_id:
            raise ValueError("no_solution must not have product_id")

    submission_id = f"SUB-{uuid.uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT INTO procurement_submissions
           (submission_id, episode_id, task_id, decision_type, product_id, quantity, justification, submitted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (submission_id, episode_id, task_id, decision_type, product_id, quantity, justification, now),
    )

    # Mark episode as submitted
    new_status = "submitted"
    conn.execute(
        "UPDATE episodes SET status = ?, completed_at = ? WHERE episode_id = ?",
        (new_status, now, episode_id),
    )
    conn.commit()
    return submission_id


def reset_db(conn: sqlite3.Connection):
    """Drop and recreate all tables, then re-seed."""
    conn.executescript("""
        DROP TABLE IF EXISTS procurement_submissions;
        DROP TABLE IF EXISTS episodes;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS suppliers;
    """)
    init_schema(conn)
    conn.commit()


def get_row_counts(conn: sqlite3.Connection) -> dict:
    """Return row counts for all tables."""
    tables = ["suppliers", "products", "episodes", "procurement_submissions"]
    counts = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
        counts[table] = row["cnt"]
    return counts
