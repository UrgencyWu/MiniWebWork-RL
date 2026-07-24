"""
MiniWebWork-RL Procurement Web Application.

FastAPI + Jinja2 server-side rendered procurement site.
Supports deterministic tasks, product search, supplier navigation,
episode lifecycle, and procurement submission.
"""

import os
import sqlite3
from typing import Optional

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .db import (
    create_episode,
    create_submission,
    get_connection,
    get_db_path,
    init_schema,
)
from .seed import load_products, load_suppliers, seed_database
from .tasks import get_public_task, load_public_tasks

from fastapi.templating import Jinja2Templates
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="MiniWebWork-RL Procurement")


def get_db():
    """Get a database connection, ensuring schema is initialized."""
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = get_connection(db_path)
    init_schema(conn)
    return conn


# ============================================================
# 1. Health
# ============================================================
@app.get("/health")
async def health():
    conn = get_db()
    supplier_count = conn.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
    product_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    task_count = len(load_public_tasks())

    sqlite_ok = False
    sqlite_version = "unknown"
    try:
        test_conn = sqlite3.connect(":memory:")
        test_conn.execute("CREATE TABLE smoke (id INTEGER, name TEXT)")
        test_conn.execute("INSERT INTO smoke VALUES (1, 'test')")
        row = test_conn.execute("SELECT * FROM smoke").fetchone()
        test_conn.close()
        sqlite_ok = row == (1, "test")
        sqlite_version = sqlite3.sqlite_version
    except Exception:
        pass

    db_available = supplier_count > 0 and product_count > 0

    return {
        "status": "ok" if (sqlite_ok and db_available) else "degraded",
        "sqlite": {"available": sqlite_ok, "version": sqlite_version},
        "database_available": db_available,
        "supplier_count": supplier_count,
        "product_count": product_count,
        "task_count": task_count,
    }


# ============================================================
# 2. Smoke Test (M1.0 compatibility)
# ============================================================
@app.get("/smoke", response_class=HTMLResponse)
async def smoke(request: Request):
    return templates.TemplateResponse("smoke.html", {"request": request})


# ============================================================
# 3. Home
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    tasks = load_public_tasks()
    return templates.TemplateResponse("home.html", {"request": request, "tasks": tasks})


# ============================================================
# 4. Task Detail
# ============================================================
@app.get("/tasks/{task_id}", response_class=HTMLResponse)
async def task_detail(request: Request, task_id: str):
    task = get_public_task(task_id)
    if task is None:
        return HTMLResponse("Task not found", status_code=404)
    return templates.TemplateResponse("task.html", {"request": request, "task": task})


# ============================================================
# 5. Start Task (Create Episode)
# ============================================================
@app.post("/tasks/{task_id}/start")
async def start_task(task_id: str):
    task = get_public_task(task_id)
    if task is None:
        return HTMLResponse("Task not found", status_code=404)

    conn = get_db()
    episode_id = create_episode(conn, task_id)
    conn.close()

    return RedirectResponse(
        f"/products?episode_id={episode_id}&task_id={task_id}",
        status_code=303,
    )


# ============================================================
# 6. Product Listing
# ============================================================
@app.get("/products", response_class=HTMLResponse)
async def product_list(
    request: Request,
    q: Optional[str] = None,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    min_memory_gb: Optional[int] = None,
    max_delivery_days: Optional[int] = None,
    certified_only: Optional[str] = None,
    min_supplier_rating: Optional[float] = None,
    supplier_region: Optional[str] = None,
    in_stock_only: Optional[str] = None,
    min_warranty_months: Optional[int] = None,
    episode_id: Optional[str] = None,
    task_id: Optional[str] = None,
):
    conn = get_db()

    query = """
        SELECT p.*, s.name as supplier_name, s.rating, s.region as supplier_region,
               s.certified, s.delivery_reliability
        FROM products p
        JOIN suppliers s ON p.supplier_id = s.supplier_id
        WHERE 1=1
    """
    params = []

    if q:
        query += " AND (p.name LIKE ? OR p.model_number LIKE ? OR p.description LIKE ?)"
        kw = f"%{q}%"
        params.extend([kw, kw, kw])

    if category:
        query += " AND p.category = ?"
        params.append(category)

    if max_price is not None:
        query += " AND p.price <= ?"
        params.append(max_price)

    if min_memory_gb is not None:
        query += " AND (p.memory_gb IS NOT NULL AND p.memory_gb >= ?)"
        params.append(min_memory_gb)

    if max_delivery_days is not None:
        query += " AND p.delivery_days <= ?"
        params.append(max_delivery_days)

    if certified_only == "1":
        query += " AND s.certified = 1"

    if min_supplier_rating is not None:
        query += " AND s.rating >= ?"
        params.append(min_supplier_rating)

    if supplier_region:
        query += " AND s.region = ?"
        params.append(supplier_region)

    if in_stock_only == "1":
        query += " AND p.stock > 0"

    if min_warranty_months is not None:
        query += " AND p.warranty_months >= ?"
        params.append(min_warranty_months)

    query += " ORDER BY p.price ASC"

    products = conn.execute(query, params).fetchall()
    conn.close()

    task = None
    if task_id:
        task = get_public_task(task_id)

    return templates.TemplateResponse(
        "products.html",
        {
            "request": request,
            "products": products,
            "task": task,
            "episode_id": episode_id or "",
            "q": q or "",
            "category": category or "",
            "max_price": max_price or "",
            "min_memory_gb": min_memory_gb or "",
            "max_delivery_days": max_delivery_days or "",
            "certified_only": certified_only or "",
            "min_supplier_rating": min_supplier_rating or "",
            "supplier_region": supplier_region or "",
            "in_stock_only": in_stock_only or "",
            "min_warranty_months": min_warranty_months or "",
        },
    )


# ============================================================
# 7. Product Detail
# ============================================================
@app.get("/products/{product_id}", response_class=HTMLResponse)
async def product_detail(
    request: Request,
    product_id: str,
    episode_id: Optional[str] = None,
    task_id: Optional[str] = None,
):
    conn = get_db()
    product = conn.execute(
        """SELECT p.*, s.name as supplier_name, s.rating, s.region as supplier_region,
                  s.certified, s.delivery_reliability
           FROM products p JOIN suppliers s ON p.supplier_id = s.supplier_id
           WHERE p.product_id = ?""",
        (product_id,),
    ).fetchone()
    conn.close()

    if product is None:
        return HTMLResponse("Product not found", status_code=404)

    task = None
    if task_id:
        task = get_public_task(task_id)

    return templates.TemplateResponse(
        "product_detail.html",
        {
            "request": request,
            "product": product,
            "task": task,
            "episode_id": episode_id or "",
        },
    )


# ============================================================
# 8. Supplier Detail
# ============================================================
@app.get("/suppliers/{supplier_id}", response_class=HTMLResponse)
async def supplier_detail(
    request: Request,
    supplier_id: str,
    episode_id: Optional[str] = None,
    task_id: Optional[str] = None,
):
    conn = get_db()
    supplier = conn.execute(
        "SELECT * FROM suppliers WHERE supplier_id = ?", (supplier_id,)
    ).fetchone()

    if supplier is None:
        conn.close()
        return HTMLResponse("Supplier not found", status_code=404)

    products = conn.execute(
        "SELECT * FROM products WHERE supplier_id = ?", (supplier_id,)
    ).fetchall()
    conn.close()

    task = None
    if task_id:
        task = get_public_task(task_id)

    return templates.TemplateResponse(
        "supplier_detail.html",
        {
            "request": request,
            "supplier": supplier,
            "products": products,
            "task": task,
            "episode_id": episode_id or "",
        },
    )


# ============================================================
# 9. Procurement Form
# ============================================================
@app.get("/procurement/new", response_class=HTMLResponse)
async def procurement_form(
    request: Request,
    episode_id: str = "",
    task_id: str = "",
    product_id: str = "",
):
    task = get_public_task(task_id)
    if not task:
        return HTMLResponse("Task not found", status_code=404)

    product = None
    if product_id:
        conn = get_db()
        product = conn.execute(
            """SELECT p.*, s.name as supplier_name
               FROM products p JOIN suppliers s ON p.supplier_id = s.supplier_id
               WHERE p.product_id = ?""",
            (product_id,),
        ).fetchone()
        conn.close()

    return templates.TemplateResponse(
        "procurement_form.html",
        {
            "request": request,
            "episode_id": episode_id,
            "task": task,
            "product": product,
        },
    )


# ============================================================
# 10. Submit Procurement
# ============================================================
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
    conn = get_db()

    try:
        submission_id = create_submission(
            conn,
            episode_id=episode_id,
            task_id=task_id,
            decision_type=decision_type,
            product_id=product_id if product_id else None,
            quantity=quantity,
            justification=justification,
        )

        submission = conn.execute(
            "SELECT * FROM procurement_submissions WHERE submission_id = ?",
            (submission_id,),
        ).fetchone()
        conn.close()

        return templates.TemplateResponse(
            "procurement_result.html",
            {"request": request, "submission": submission},
        )

    except ValueError as e:
        conn.close()
        return HTMLResponse(f"<h1>Submission Error</h1><p>{e}</p>", status_code=400)


# ============================================================
# 11. Procurement Result
# ============================================================
@app.get("/procurement/result/{submission_id}", response_class=HTMLResponse)
async def procurement_result(request: Request, submission_id: str):
    conn = get_db()
    submission = conn.execute(
        "SELECT * FROM procurement_submissions WHERE submission_id = ?",
        (submission_id,),
    ).fetchone()
    conn.close()

    if submission is None:
        return HTMLResponse("Submission not found", status_code=404)

    return templates.TemplateResponse(
        "procurement_result.html",
        {"request": request, "submission": submission},
    )


# ============================================================
# Main entry
# ============================================================
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
