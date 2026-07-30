from miniwebwork.db import create_episode, get_connection, init_schema
from miniwebwork.models import FAILURE_INVALID_EPISODE, FAILURE_MISSING_SUBMISSION
from miniwebwork.seed import seed_database
from miniwebwork.verifier import verify_episode


def _database(tmp_path):
    path = tmp_path / "verifier.db"
    conn = get_connection(str(path))
    init_schema(conn)
    seed_database(conn)
    return path, conn


def test_verifier_rejects_episode_from_different_task(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIWEBWORK_TASK_DIR", raising=False)
    path, conn = _database(tmp_path)
    episode_id = create_episode(conn, "TASK-001")
    conn.close()

    result = verify_episode("TASK-002", episode_id, str(path))

    assert result.success is False
    assert FAILURE_INVALID_EPISODE in result.failure_reasons
    assert "belongs to TASK-001" in result.details["error"]


def test_verifier_reports_missing_submission(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIWEBWORK_TASK_DIR", raising=False)
    path, conn = _database(tmp_path)
    episode_id = create_episode(conn, "TASK-001")
    conn.close()

    result = verify_episode("TASK-001", episode_id, str(path))

    assert result.success is False
    assert result.failure_reasons == [FAILURE_MISSING_SUBMISSION]
