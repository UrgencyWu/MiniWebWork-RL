"""Deterministic FastAPI procurement site used by the browser environment."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .db import create_episode, create_submission, get_connection, get_db_path, init_schema
from .tasks import get_public_task, load_public_tasks

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app = FastAPI(title="MiniWebWork-RL Procurement")


def get_db():
    """Open an initialized connection for the process-selected runtime DB."""
    db_path = Path(get_db_path()).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = get_connection(str(db_path))
    init_schema(connection)
    return connection


def _parse_optional_float(name: str, value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {name}") from exc


def _parse_optional_int(name: str, value: str) -> Optional[int]:
    if value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {name}") from exc


def _validate_episode_task(connection, episode_id: str, task_id: str) -> None:
    """Require a live episode bound to the route's public task."""
    if not episode_id or not task_id:
        raise HTTPException(status_code=400, detail="episode_id and task_id are required")
    if get_public_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    episode = connection.execute(
        "SELECT task_id, status FROM episodes WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    if episode["task_id"] != task_id:
        raise HTTPException(status_code=409, detail="Episode/task mismatch")
    if episode["status"] not in {"active", "submitted"}:
        raise HTTPException(status_code=409, detail="Episode is not active")


def _optional_task_context(connection, episode_id: str, task_id: str):
    """Validate a task flow when either context identifier is supplied."""
    if not episode_id and not task_id:
        return None
    _validate_episode_task(connection, episode_id, task_id)
    return get_public_task(task_id)


@app.get("/health")
async def health():
    with closing(get_db()) as connection:
        supplier_count = connection.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
        product_count = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    sqlite_ok = False
    try:
        with closing(sqlite3.connect(":memory:")) as smoke_connection:
            smoke_connection.execute("CREATE TABLE smoke (id INTEGER, name TEXT)")
            smoke_connection.execute("INSERT INTO smoke VALUES (1, 'test')")
            sqlite_ok = smoke_connection.execute("SELECT * FROM smoke").fetchone() == (1, "test")
    except sqlite3.Error:
        sqlite_ok = False

    database_available = supplier_count > 0 and product_count > 0
    return {
        "status": "ok" if sqlite_ok and database_available else "degraded",
        "sqlite": {"available": sqlite_ok, "version": sqlite3.sqlite_version},
        "database_available": database_available,
        "supplier_count": supplier_count,
        "product_count": product_count,
        "task_count": len(load_public_tasks()),
    }


@app.get("/smoke", response_class=HTMLResponse)
async def smoke(request: Request):
    return templates.TemplateResponse(request=request, name="smoke.html", context={})


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"tasks": load_public_tasks()},
    )


@app.get("/tasks/{task_id}", response_class=HTMLResponse)
async def task_detail(request: Request, task_id: str):
    task = get_public_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return templates.TemplateResponse(
        request=request,
        name="task.html",
        context={"task": task},
    )


@app.post("/tasks/{task_id}/start")
async def start_task(task_id: str):
    if get_public_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    with closing(get_db()) as connection:
        episode_id = create_episode(connection, task_id)
    query = urlencode({"episode_id": episode_id, "task_id": task_id})
    return RedirectResponse(f"/products?{query}", status_code=303)


@app.get("/products", response_class=HTMLResponse)
async def product_list(
    request: Request,
    q: str = "",
    category: str = "",
    max_price: str = "",
    min_memory_gb: str = "",
    max_delivery_days: str = "",
    certified_only: str = "",
    min_supplier_rating: str = "",
    supplier_region: str = "",
    in_stock_only: str = "",
    min_warranty_months: str = "",
    episode_id: str = "",
    task_id: str = "",
):
    if certified_only not in {"", "0", "1"}:
        raise HTTPException(status_code=422, detail="Invalid certified_only")
    if in_stock_only not in {"", "0", "1"}:
        raise HTTPException(status_code=422, detail="Invalid in_stock_only")

    parsed_max_price = _parse_optional_float("max_price", max_price)
    parsed_min_memory = _parse_optional_int("min_memory_gb", min_memory_gb)
    parsed_delivery = _parse_optional_int("max_delivery_days", max_delivery_days)
    parsed_rating = _parse_optional_float("min_supplier_rating", min_supplier_rating)
    parsed_warranty = _parse_optional_int("min_warranty_months", min_warranty_months)

    query = """
        SELECT p.*, s.name AS supplier_name, s.rating,
               s.region AS supplier_region, s.certified,
               s.delivery_reliability
        FROM products p
        JOIN suppliers s ON p.supplier_id = s.supplier_id
        WHERE 1=1
    """
    params: list = []
    if q:
        query += " AND (p.name LIKE ? OR p.model_number LIKE ? OR p.description LIKE ?)"
        keyword = f"%{q}%"
        params.extend([keyword, keyword, keyword])
    if category:
        query += " AND p.category = ?"
        params.append(category)
    if parsed_max_price is not None:
        query += " AND p.price <= ?"
        params.append(parsed_max_price)
    if parsed_min_memory is not None:
        query += " AND p.memory_gb IS NOT NULL AND p.memory_gb >= ?"
        params.append(parsed_min_memory)
    if parsed_delivery is not None:
        query += " AND p.delivery_days <= ?"
        params.append(parsed_delivery)
    if certified_only:
        query += " AND s.certified = ?"
        params.append(int(certified_only))
    if parsed_rating is not None:
        query += " AND s.rating >= ?"
        params.append(parsed_rating)
    if supplier_region:
        query += " AND s.region = ?"
        params.append(supplier_region)
    if in_stock_only == "1":
        query += " AND p.stock > 0"
    if parsed_warranty is not None:
        query += " AND p.warranty_months >= ?"
        params.append(parsed_warranty)
    query += " ORDER BY p.price ASC, p.product_id ASC"

    with closing(get_db()) as connection:
        task = _optional_task_context(connection, episode_id, task_id)
        products = connection.execute(query, params).fetchall()

    return templates.TemplateResponse(
        request=request,
        name="products.html",
        context={
            "products": products,
            "task": task,
            "episode_id": episode_id,
            "q": q,
            "category": category,
            "max_price": max_price,
            "min_memory_gb": min_memory_gb,
            "max_delivery_days": max_delivery_days,
            "certified_only": certified_only,
            "min_supplier_rating": min_supplier_rating,
            "supplier_region": supplier_region,
            "in_stock_only": in_stock_only,
            "min_warranty_months": min_warranty_months,
        },
    )


@app.get("/products/{product_id}", response_class=HTMLResponse)
async def product_detail(
    request: Request,
    product_id: str,
    episode_id: str = "",
    task_id: str = "",
):
    with closing(get_db()) as connection:
        task = _optional_task_context(connection, episode_id, task_id)
        product = connection.execute(
            """SELECT p.*, s.name AS supplier_name, s.rating,
                      s.region AS supplier_region, s.certified,
                      s.delivery_reliability
               FROM products p
               JOIN suppliers s ON p.supplier_id = s.supplier_id
               WHERE p.product_id = ?""",
            (product_id,),
        ).fetchone()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return templates.TemplateResponse(
        request=request,
        name="product_detail.html",
        context={
            "product": product,
            "task": task,
            "episode_id": episode_id,
        },
    )


@app.get("/suppliers/{supplier_id}", response_class=HTMLResponse)
async def supplier_detail(
    request: Request,
    supplier_id: str,
    episode_id: str = "",
    task_id: str = "",
):
    with closing(get_db()) as connection:
        task = _optional_task_context(connection, episode_id, task_id)
        supplier = connection.execute(
            "SELECT * FROM suppliers WHERE supplier_id = ?",
            (supplier_id,),
        ).fetchone()
        if supplier is None:
            raise HTTPException(status_code=404, detail="Supplier not found")
        products = connection.execute(
            "SELECT * FROM products WHERE supplier_id = ? ORDER BY price, product_id",
            (supplier_id,),
        ).fetchall()
    return templates.TemplateResponse(
        request=request,
        name="supplier_detail.html",
        context={
            "supplier": supplier,
            "products": products,
            "task": task,
            "episode_id": episode_id,
        },
    )


@app.get("/procurement/new", response_class=HTMLResponse)
async def procurement_form(
    request: Request,
    episode_id: str,
    task_id: str,
    product_id: str = "",
):
    with closing(get_db()) as connection:
        _validate_episode_task(connection, episode_id, task_id)
        task = get_public_task(task_id)
        product = None
        if product_id:
            product = connection.execute(
                """SELECT p.*, s.name AS supplier_name
                   FROM products p
                   JOIN suppliers s ON p.supplier_id = s.supplier_id
                   WHERE p.product_id = ?""",
                (product_id,),
            ).fetchone()
            if product is None:
                raise HTTPException(status_code=404, detail="Product not found")
    return templates.TemplateResponse(
        request=request,
        name="procurement_form.html",
        context={
            "episode_id": episode_id,
            "task": task,
            "product": product,
        },
    )


@app.post("/procurement/submit", response_class=HTMLResponse)
async def submit_procurement(
    request: Request,
    episode_id: str = Form(...),
    task_id: str = Form(...),
    decision_type: str = Form(...),
    product_id: str = Form(default=""),
    quantity: int = Form(default=1),
    justification: str = Form(default=""),
):
    try:
        with closing(get_db()) as connection:
            _validate_episode_task(connection, episode_id, task_id)
            submission_id = create_submission(
                connection,
                episode_id=episode_id,
                task_id=task_id,
                decision_type=decision_type,
                product_id=product_id or None,
                quantity=quantity,
                justification=justification[:2000],
            )
            submission = connection.execute(
                "SELECT * FROM procurement_submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
    except ValueError as exc:
        return PlainTextResponse(f"Submission Error: {exc}", status_code=400)

    return templates.TemplateResponse(
        request=request,
        name="procurement_result.html",
        context={"submission": submission},
    )


@app.get("/procurement/result/{submission_id}", response_class=HTMLResponse)
async def procurement_result(request: Request, submission_id: str):
    with closing(get_db()) as connection:
        submission = connection.execute(
            "SELECT * FROM procurement_submissions WHERE submission_id = ?",
            (submission_id,),
        ).fetchone()
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    return templates.TemplateResponse(
        request=request,
        name="procurement_result.html",
        context={"submission": submission},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "miniwebwork.webapp:app",
        host=os.environ.get("MINIWEBWORK_HOST", "127.0.0.1"),
        port=int(os.environ.get("MINIWEBWORK_PORT", "18080")),
        log_level="info",
        reload=False,
    )
