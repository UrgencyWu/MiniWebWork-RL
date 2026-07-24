"""
End-to-end procurement tests using Playwright.

Runs 4 cases (in Slurm or directly):
  A: Exact product selection (TASK-001 -> PRD-001)
  B: Multi-filter cheapest feasible (TASK-010 -> PRD-002)
  C: No feasible product (TASK-004 -> no_solution)
  D: Wrong product submission (TASK-003, submit non-optimal)
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from .db import get_connection, get_db_path, reset_db
from .seed import seed_database
from .verifier import verify_episode

BASE_URL = os.environ.get("MINIWEBWORK_URL", "http://127.0.0.1:18080")
ARTIFACTS_DIR = os.environ.get(
    "MINIWEBWORK_ARTIFACTS",
    str(Path(__file__).resolve().parent.parent.parent / "artifacts"),
)
SCREENSHOTS_DIR = Path(ARTIFACTS_DIR) / "m1_1_screenshots"


def _w(sec=0.5):
    time.sleep(sec)


def _episode_id(page):
    try:
        el = page.locator('[data-testid="episode-id"]')
        return el.inner_text(timeout=3000).strip()
    except Exception:
        return ""


def _submission_id(page):
    try:
        text = page.locator('[data-testid="submission-result"]').inner_text(timeout=3000)
        for line in text.split("\n"):
            if line.strip().startswith("SUB-"):
                return line.strip()
    except Exception:
        pass
    return ""


def _navigate_with_url(page, path, task_id, episode_id, **params):
    """Navigate to a page with preserved episode/task context."""
    all_params = dict(params)
    if episode_id:
        all_params["episode_id"] = episode_id
    if task_id:
        all_params["task_id"] = task_id
    url = f"{BASE_URL}{path}"
    if all_params:
        url += "?" + urlencode(all_params)
    page.goto(url, timeout=15000)
    _w()


def setup_database():
    db_path = get_db_path()
    conn = get_connection(db_path)
    reset_db(conn)
    seed_database(conn)
    conn.close()
    print(f"Database reset: {db_path}")


def start_task(page, task_id):
    """Start a task via POST and return episode_id."""
    page.goto(f"{BASE_URL}/tasks/{task_id}", timeout=15000)
    _w()
    page.locator('[data-testid="start-task-button"]').click()
    _w()
    return _episode_id(page)


def run_case_a(page) -> dict:
    """Case A: Exact product. TASK-001 -> PRD-001."""
    print("\n=== Case A: Exact Product (TASK-001 -> PRD-001) ===")
    urls = []

    eid = start_task(page, "TASK-001")
    urls.append(page.url)
    print(f"  Episode: {eid}")

    # Navigate to filtered products page with keyword
    _navigate_with_url(page, "/products", "TASK-001", eid, q="CC-A100X")
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_a_list.png"))

    # Verify the product card
    assert page.locator('[data-testid="product-card-PRD-001"]').is_visible(), \
        f"PRD-001 not visible after search. URL: {page.url}"
    count_text = page.locator('[data-testid="result-count"]').inner_text(timeout=3000)
    print(f"  Results: {count_text}")

    # Go to product detail
    _navigate_with_url(page, "/products/PRD-001", "TASK-001", eid)
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_a_product.png"))

    # Go to supplier
    _navigate_with_url(page, "/suppliers/SUP-001", "TASK-001", eid)
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_a_supplier.png"))

    # Go to procurement form
    _navigate_with_url(page, "/procurement/new", "TASK-001", eid, product_id="PRD-001")
    urls.append(page.url)

    # Submit
    page.locator('[data-testid="submit-procurement"]').click()
    _w()
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_a_result.png"))

    sid = _submission_id(page)
    vr = verify_episode("TASK-001", eid)
    print(f"  Verifier: success={vr.success}")
    return {
        "case": "A", "task_id": "TASK-001", "episode_id": eid,
        "submission_id": sid, "verifier": vr.to_dict(),
        "success": vr.success, "urls": urls,
    }


def run_case_b(page) -> dict:
    """Case B: Multi-filter cheapest feasible. TASK-010 -> PRD-002."""
    print("\n=== Case B: Multi-Filter (TASK-010 -> PRD-002) ===")
    urls = []

    eid = start_task(page, "TASK-010")
    urls.append(page.url)
    print(f"  Episode: {eid}")

    # Apply all filters directly via URL
    _navigate_with_url(page, "/products", "TASK-010", eid,
                       category="GPU", min_memory_gb=48, max_delivery_days=14, in_stock_only="1")
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_b_list.png"))

    assert page.locator('[data-testid="product-card-PRD-002"]').is_visible(), \
        f"PRD-002 not visible. URL: {page.url}"
    count_text = page.locator('[data-testid="result-count"]').inner_text(timeout=3000)
    print(f"  Results: {count_text}")

    _navigate_with_url(page, "/products/PRD-002", "TASK-010", eid)
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_b_detail.png"))

    _navigate_with_url(page, "/procurement/new", "TASK-010", eid, product_id="PRD-002")
    page.locator('[data-testid="submit-procurement"]').click()
    _w()
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_b_result.png"))

    sid = _submission_id(page)
    vr = verify_episode("TASK-010", eid)
    print(f"  Verifier: success={vr.success}")
    return {
        "case": "B", "task_id": "TASK-010", "episode_id": eid,
        "submission_id": sid, "verifier": vr.to_dict(),
        "success": vr.success, "urls": urls,
    }


def run_case_c(page) -> dict:
    """Case C: No feasible product. TASK-004 -> no_solution."""
    print("\n=== Case C: No Feasible Product (TASK-004) ===")
    urls = []

    eid = start_task(page, "TASK-004")
    urls.append(page.url)
    print(f"  Episode: {eid}")

    # Apply strict filters
    _navigate_with_url(page, "/products", "TASK-004", eid,
                       category="GPU", min_memory_gb=64, max_price=50000, in_stock_only="1")
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_c_no_results.png"))

    count_text = page.locator('[data-testid="result-count"]').inner_text(timeout=3000)
    print(f"  Results: {count_text}")
    assert "0" in count_text, f"Expected 0 results, got: {count_text}"

    # Declare no solution
    assert page.locator('[data-testid="declare-no-solution"]').is_visible()
    page.locator('[data-testid="declare-no-solution"]').click()
    _w()
    urls.append(page.url)

    page.locator('[data-testid="submit-procurement"]').click()
    _w()
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_c_result.png"))

    sid = _submission_id(page)
    vr = verify_episode("TASK-004", eid)
    print(f"  Verifier: success={vr.success}")
    return {
        "case": "C", "task_id": "TASK-004", "episode_id": eid,
        "submission_id": sid, "verifier": vr.to_dict(),
        "success": vr.success, "urls": urls,
    }


def run_case_d(page) -> dict:
    """Case D: Wrong product. TASK-003, submit PRD-023 (not cheapest)."""
    print("\n=== Case D: Wrong Product (TASK-003) ===")
    urls = []

    eid = start_task(page, "TASK-003")
    urls.append(page.url)
    print(f"  Episode: {eid}")

    # Pick PRD-023 (25000) — not optimal, PRD-022 (18000) is cheapest
    _navigate_with_url(page, "/products/PRD-023", "TASK-003", eid)
    urls.append(page.url)
    page.locator('[data-testid="select-product"]').click()
    _w()

    page.locator('[data-testid="submit-procurement"]').click()
    _w()
    urls.append(page.url)
    page.screenshot(path=str(SCREENSHOTS_DIR / "case_d_result.png"))

    sid = _submission_id(page)
    vr = verify_episode("TASK-003", eid)
    print(f"  Verifier: success={vr.success}, reasons={vr.failure_reasons}")
    return {
        "case": "D", "task_id": "TASK-003", "episode_id": eid,
        "submission_id": sid, "verifier": vr.to_dict(),
        "success": not vr.success, "verifier_success": vr.success, "urls": urls,
    }


def main():
    setup_database()

    results = []
    overall_ok = True
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page()
            page.set_default_timeout(10000)

            cases = [
                ("A", run_case_a),
                ("B", run_case_b),
                ("C", run_case_c),
                ("D", run_case_d),
            ]

            for case_label, case_fn in cases:
                start = time.time()
                try:
                    result = case_fn(page)
                except Exception as e:
                    traceback.print_exc()
                    result = {
                        "case": case_label, "task_id": "", "episode_id": "",
                        "submission_id": "", "verifier": {}, "success": False,
                        "error": str(e), "urls": [],
                    }
                result["elapsed_s"] = round(time.time() - start, 2)
                results.append(result)
                if not result["success"]:
                    overall_ok = False

            browser.close()

    except Exception as e:
        traceback.print_exc()
        print(f"FATAL: {e}")
        overall_ok = False

    # Save results
    output = {"overall_success": overall_ok, "cases": results}
    output_path = Path(ARTIFACTS_DIR) / "m1_1_e2e_result.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nE2E results saved: {output_path}")
    print(f"Overall: {'PASS' if overall_ok else 'FAIL'}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
