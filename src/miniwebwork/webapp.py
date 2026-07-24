"""
Minimal local web application for browser agent smoke testing.

Provides:
  GET /       - Smoke test page with input, button, and result area
  GET /health - Health check with SQLite smoke test
"""

import os
import sqlite3

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="MiniWebWork-RL Smoke Test")

SMOKE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MiniWebWork-RL Smoke Test</title>
    <style>
        body { font-family: sans-serif; margin: 2rem; }
        input, button { padding: 0.5rem; font-size: 1rem; }
        #result { margin-top: 1rem; padding: 1rem; border: 1px solid #ccc; min-height: 2rem; }
    </style>
</head>
<body>
    <h1>MiniWebWork-RL Smoke Test</h1>
    <input id="query" type="text" placeholder="Enter search term...">
    <button id="search-button" onclick="doSearch()">Search</button>
    <div id="result">ready</div>
    <script>
        function doSearch() {
            var q = document.getElementById('query').value;
            document.getElementById('result').textContent = 'searched: ' + q;
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return SMOKE_PAGE


@app.get("/health")
async def health():
    """Health check with SQLite smoke test."""
    sqlite_ok = False
    sqlite_version = "unknown"
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE smoke (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO smoke VALUES (1, 'test')")
        row = conn.execute("SELECT * FROM smoke").fetchone()
        conn.close()
        sqlite_ok = row == (1, "test")
        sqlite_version = sqlite3.sqlite_version
    except Exception:
        pass

    return {
        "status": "ok" if sqlite_ok else "degraded",
        "sqlite": {
            "available": sqlite_ok,
            "version": sqlite_version,
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MINIWEBWORK_HOST", "127.0.0.1")
    port = int(os.environ.get("MINIWEBWORK_PORT", "18080"))

    uvicorn.run(
        "miniwebwork.webapp:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )
