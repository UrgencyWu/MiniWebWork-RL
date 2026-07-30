"""Shared constants and page classification for browser observations.

Observation extraction itself is implemented once, asynchronously, inside
`PlaywrightThreadManager._do_build_observation`.  The previous Sync Playwright
extractor was removed to prevent divergent element IDs and truncation rules.
"""

MAX_VISIBLE_TEXT = 8000

INTERACTIVE_ROLES = {
    "link",
    "button",
    "textbox",
    "searchbox",
    "checkbox",
    "combobox",
    "spinbutton",
    "textarea",
}


def _classify_page(path: str) -> str:
    """Map a URL path to the versioned Observation page type."""
    if path == "/":
        return "home"
    if path == "/smoke":
        return "smoke"
    if path == "/health":
        return "error"
    if path.startswith("/tasks/") and not path.endswith("/start"):
        return "task"
    if path == "/products":
        return "products"
    if path.startswith("/products/"):
        return "product_detail"
    if path.startswith("/suppliers/"):
        return "supplier_detail"
    if path == "/procurement/new":
        return "procurement_form"
    if path == "/procurement/submit" or path.startswith("/procurement/result/"):
        return "procurement_result"
    return "unknown"
