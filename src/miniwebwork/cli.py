"""Command-line utilities for MiniWebWork-RL data and runtime contracts."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import closing
from pathlib import Path

from .db import get_connection, get_db_path, get_row_counts, init_schema, reset_db
from .seed import seed_database, validate_seed
from .tasks import validate_tasks
from .verifier import verify_episode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = (PROJECT_ROOT / "data" / "runtime").resolve()


def _database_path(value: str | None = None) -> Path:
    return Path(value or get_db_path()).expanduser().resolve()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def cmd_init_db(args) -> int:
    path = _database_path(args.db_path)
    _ensure_parent(path)
    with closing(get_connection(str(path))) as connection:
        init_schema(connection)
        seed_database(connection)
        counts = get_row_counts(connection)
    _print({"database": str(path), "counts": counts, "status": "initialized"})
    return 0


def cmd_reset_db(args) -> int:
    path = _database_path(args.db_path)
    try:
        path.relative_to(RUNTIME_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Refusing to reset database outside {RUNTIME_ROOT}: {path}") from exc

    _ensure_parent(path)
    with closing(get_connection(str(path))) as connection:
        init_schema(connection)
        before = get_row_counts(connection)
        reset_db(connection)
        seed_database(connection)
        after = get_row_counts(connection)
    _print({"database": str(path), "before": before, "after": after, "status": "reset"})
    return 0


def cmd_validate_seed(args) -> int:
    result = validate_seed()
    _print(result)
    return 0 if result["valid"] else 1


def cmd_validate_tasks(args) -> int:
    path = _database_path(args.db_path)
    _ensure_parent(path)
    with closing(get_connection(str(path))) as connection:
        init_schema(connection)
        seed_database(connection)
        result = validate_tasks(connection, task_dir=args.task_dir)
    _print(result)
    return 0 if result["valid"] else 1


def cmd_verify(args) -> int:
    result = verify_episode(
        args.task_id,
        args.episode_id,
        db_path=str(_database_path(args.db_path)),
        task_dir=args.task_dir,
    )
    _print(result.to_dict())
    return 0 if result.success else 1


def cmd_status(args) -> int:
    path = _database_path(args.db_path)
    _ensure_parent(path)
    with closing(get_connection(str(path))) as connection:
        init_schema(connection)
        counts = get_row_counts(connection)
        episodes = connection.execute(
            "SELECT status, COUNT(*) AS cnt FROM episodes GROUP BY status"
        ).fetchall()
        submissions = connection.execute(
            "SELECT decision_type, COUNT(*) AS cnt "
            "FROM procurement_submissions GROUP BY decision_type"
        ).fetchall()
    _print(
        {
            "database": str(path),
            "counts": counts,
            "episodes_by_status": {row["status"]: row["cnt"] for row in episodes},
            "submissions_by_type": {
                row["decision_type"]: row["cnt"] for row in submissions
            },
        }
    )
    return 0


def _add_db_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db-path",
        default=None,
        help="Runtime SQLite path; defaults to MINIWEBWORK_DB_PATH/project runtime DB.",
    )


def _add_task_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=None,
        help="Exclusive directory containing *_public.jsonl and *_oracle.jsonl.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniWebWork-RL management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="Initialize and seed the runtime DB")
    _add_db_argument(init_parser)
    init_parser.set_defaults(handler=cmd_init_db)

    reset_parser = subparsers.add_parser("reset-db", help="Reset a project runtime DB")
    _add_db_argument(reset_parser)
    reset_parser.set_defaults(handler=cmd_reset_db)

    seed_parser = subparsers.add_parser("validate-seed", help="Validate seed manifests/data")
    seed_parser.set_defaults(handler=cmd_validate_seed)

    task_parser = subparsers.add_parser("validate-tasks", help="Validate public/Oracle tasks")
    _add_db_argument(task_parser)
    _add_task_dir_argument(task_parser)
    task_parser.set_defaults(handler=cmd_validate_tasks)

    verify_parser = subparsers.add_parser("verify", help="Verify one persisted episode")
    verify_parser.add_argument("--task-id", required=True)
    verify_parser.add_argument("--episode-id", required=True)
    _add_db_argument(verify_parser)
    _add_task_dir_argument(verify_parser)
    verify_parser.set_defaults(handler=cmd_verify)

    status_parser = subparsers.add_parser("status", help="Show runtime DB status")
    _add_db_argument(status_parser)
    status_parser.set_defaults(handler=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Preserve existing subprocess behavior when a task source is explicitly
    # selected, while library calls receive task_dir directly.
    if getattr(args, "task_dir", None) is not None:
        os.environ["MINIWEBWORK_TASK_DIR"] = str(args.task_dir.expanduser().resolve())
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
