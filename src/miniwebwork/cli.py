"""CLI for MiniWebWork-RL environment management."""

import json
import os
import sys

from .db import get_connection, get_db_path, init_schema, reset_db, get_row_counts
from .seed import seed_database, update_manifest, validate_seed
from .tasks import validate_tasks
from .verifier import verify_episode


def cmd_init_db():
    """Initialize database with schema and seed data."""
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = get_connection(db_path)
    init_schema(conn)
    seed_database(conn)

    counts = get_row_counts(conn)
    print(f"Database initialized: {db_path}")
    print(f"  Suppliers: {counts['suppliers']}")
    print(f"  Products: {counts['products']}")
    conn.close()


def cmd_reset_db():
    """Reset database (drop and re-seed)."""
    db_path = get_db_path()

    if not db_path.startswith(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "runtime")
    )):
        print(f"ERROR: Refusing to reset database outside project: {db_path}")
        sys.exit(1)

    conn = get_connection(db_path)
    old_counts = get_row_counts(conn)
    reset_db(conn)
    seed_database(conn)
    new_counts = get_row_counts(conn)

    print(f"Database reset: {db_path}")
    print(f"  Before: {old_counts}")
    print(f"  After:  {new_counts}")
    conn.close()


def cmd_validate_seed():
    """Validate seed data."""
    result = validate_seed()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["valid"]:
        sys.exit(1)


def cmd_validate_tasks():
    """Validate task definitions."""
    db_path = get_db_path()
    conn = get_connection(db_path)
    result = validate_tasks(conn)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    conn.close()
    if not result["valid"]:
        sys.exit(1)


def cmd_verify(task_id=None, episode_id=None):
    """Verify an episode."""
    if not task_id or not episode_id:
        print("Usage: python -m miniwebwork.cli verify --task-id TASK-XXX --episode-id EP-XXX")
        sys.exit(1)

    result = verify_episode(task_id, episode_id)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if not result.success:
        sys.exit(1)


def cmd_status():
    """Show database status."""
    db_path = get_db_path()
    conn = get_connection(db_path)
    counts = get_row_counts(conn)

    episodes = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM episodes GROUP BY status"
    ).fetchall()

    submissions = conn.execute(
        "SELECT decision_type, COUNT(*) as cnt FROM procurement_submissions GROUP BY decision_type"
    ).fetchall()

    print(f"Database: {db_path}")
    print(f"Counts: {json.dumps(counts)}")
    print(f"Episodes by status: {json.dumps({r['status']: r['cnt'] for r in episodes})}")
    print(f"Submissions by type: {json.dumps({r['decision_type']: r['cnt'] for r in submissions})}")
    conn.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m miniwebwork.cli <command>")
        print("Commands: init-db, reset-db, validate-seed, validate-tasks, verify, status")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "init-db":
        cmd_init_db()
    elif cmd == "reset-db":
        cmd_reset_db()
    elif cmd == "validate-seed":
        cmd_validate_seed()
    elif cmd == "validate-tasks":
        cmd_validate_tasks()
    elif cmd == "verify":
        task_id = None
        episode_id = None
        args = sys.argv[2:]
        for i, arg in enumerate(args):
            if arg == "--task-id" and i + 1 < len(args):
                task_id = args[i + 1]
            elif arg == "--episode-id" and i + 1 < len(args):
                episode_id = args[i + 1]
        cmd_verify(task_id=task_id, episode_id=episode_id)
    elif cmd == "status":
        cmd_status()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
