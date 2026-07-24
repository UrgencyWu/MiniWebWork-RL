"""Unit tests for the MiniWebWork-RL web application."""

import pytest
from fastapi.testclient import TestClient

from miniwebwork.webapp import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_returns_200(client):
    """GET / returns 200 and HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_index_contains_query_input(client):
    """Page contains #query input."""
    response = client.get("/")
    assert 'id="query"' in response.text


def test_index_contains_search_button(client):
    """Page contains #search-button."""
    response = client.get("/")
    assert 'id="search-button"' in response.text


def test_index_contains_result_div(client):
    """Page contains #result div."""
    response = client.get("/")
    assert 'id="result"' in response.text


def test_index_contains_ready_text(client):
    """Page initial result text is 'ready'."""
    response = client.get("/")
    assert "ready" in response.text


def test_health_returns_200(client):
    """GET /health returns 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_is_ok(client):
    """Health endpoint reports status ok."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"


def test_health_sqlite_available(client):
    """Health endpoint confirms SQLite is available."""
    response = client.get("/health")
    data = response.json()
    assert data["sqlite"]["available"] is True
    assert data["sqlite"]["version"] != "unknown"


def test_sqlite_smoke_direct():
    """Direct SQLite smoke test."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO test VALUES (1, 'smoke')")
    row = conn.execute("SELECT * FROM test").fetchone()
    conn.close()
    assert row == (1, "smoke")
