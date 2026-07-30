import sqlite3
from contextlib import closing

import pytest

from miniwebwork.db import create_episode, create_submission, get_connection, init_schema
from miniwebwork.seed import seed_database


def _database(tmp_path):
    path = tmp_path / "contract.db"
    connection = get_connection(str(path))
    init_schema(connection)
    seed_database(connection)
    return connection


def test_quantity_must_be_positive_integer(tmp_path):
    connection = _database(tmp_path)
    episode = create_episode(connection, "TASK-001")

    with pytest.raises(ValueError, match="positive integer"):
        create_submission(
            connection,
            episode_id=episode,
            task_id="TASK-001",
            decision_type="select_product",
            product_id="P001",
            quantity=0,
        )
    connection.close()


def test_one_submission_per_episode_is_enforced(tmp_path):
    connection = _database(tmp_path)
    episode = create_episode(connection, "TASK-001")
    create_submission(
        connection,
        episode_id=episode,
        task_id="TASK-001",
        decision_type="select_product",
        product_id="P001",
    )

    with pytest.raises(ValueError, match="not active|already has"):
        create_submission(
            connection,
            episode_id=episode,
            task_id="TASK-001",
            decision_type="select_product",
            product_id="P001",
        )
    connection.close()


def test_unique_index_blocks_direct_duplicate_insert(tmp_path):
    connection = _database(tmp_path)
    episode = create_episode(connection, "TASK-001")
    create_submission(
        connection,
        episode_id=episode,
        task_id="TASK-001",
        decision_type="select_product",
        product_id="P001",
    )
    first = connection.execute(
        "SELECT * FROM procurement_submissions WHERE episode_id = ?",
        (episode,),
    ).fetchone()

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """INSERT INTO procurement_submissions
               (submission_id, episode_id, task_id, decision_type, product_id,
                quantity, justification, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "SUB-DUPLICATE",
                episode,
                "TASK-001",
                "select_product",
                "P001",
                1,
                "",
                first["submitted_at"],
            ),
        )
    connection.rollback()
    connection.close()
