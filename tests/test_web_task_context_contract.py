from contextlib import closing

from fastapi.testclient import TestClient

from miniwebwork.db import get_connection, init_schema
from miniwebwork.seed import seed_database
from miniwebwork.webapp import app


def _client(tmp_path, monkeypatch):
    database = tmp_path / "web.db"
    monkeypatch.setenv("MINIWEBWORK_DB_PATH", str(database))
    monkeypatch.delenv("MINIWEBWORK_TASK_DIR", raising=False)
    with closing(get_connection(str(database))) as connection:
        init_schema(connection)
        seed_database(connection)
    return TestClient(app), database


def _start(client, task_id="TASK-001"):
    response = client.post(f"/tasks/{task_id}/start", follow_redirects=False)
    assert response.status_code == 303
    location = response.headers["location"]
    episode_id = location.split("episode_id=", 1)[1].split("&", 1)[0]
    return episode_id


def test_products_reject_episode_task_mismatch(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    episode_id = _start(client, "TASK-001")

    response = client.get(
        "/products",
        params={"episode_id": episode_id, "task_id": "TASK-002"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Episode/task mismatch"


def test_supplier_links_preserve_episode_and_task(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    episode_id = _start(client, "TASK-001")

    response = client.get(
        "/products",
        params={"episode_id": episode_id, "task_id": "TASK-001"},
    )

    assert response.status_code == 200
    assert f"episode_id={episode_id}&amp;task_id=TASK-001" in response.text
    assert "/suppliers/" in response.text


def test_invalid_numeric_filter_is_not_silently_ignored(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    episode_id = _start(client, "TASK-001")

    response = client.get(
        "/products",
        params={
            "episode_id": episode_id,
            "task_id": "TASK-001",
            "max_price": "not-a-number",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid max_price"
