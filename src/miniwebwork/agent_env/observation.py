"""Observation extraction from Playwright page (batch-optimized)."""

import re
from urllib.parse import urlparse

from .schemas import ElementDescriptor, Observation

MAX_VISIBLE_TEXT = 8000

INTERACTIVE_ROLES = {"link", "button", "textbox", "searchbox", "checkbox", "combobox", "spinbutton", "textarea"}


def _classify_page(path: str) -> str:
    if path == "/":
        return "home"
    if path == "/smoke":
        return "smoke"
    if path == "/health":
        return "error"
    if path.startswith("/tasks/") and "/start" not in path:
        return "task"
    if path in ("/products",) or path.startswith("/products?"):
        return "products"
    if path.startswith("/products/"):
        return "product_detail"
    if path.startswith("/suppliers/"):
        return "supplier_detail"
    if path in ("/procurement/new",) or path.startswith("/procurement/new?"):
        return "procurement_form"
    if "/procurement/submit" in path:
        return "procurement_result"
    if "/procurement/result/" in path:
        return "procurement_result"
    return "unknown"


def _extract_role(tag: str, el_type: str) -> str:
    if tag == "select":
        return "combobox"
    if tag == "textarea":
        return "textarea"
    if tag == "button":
        return "button"
    if tag == "a":
        return "link"
    if tag == "input":
        if el_type == "search":
            return "searchbox"
        if el_type == "checkbox":
            return "checkbox"
        if el_type == "number":
            return "spinbutton"
        return "textbox"
    return "unknown"


def extract_visible_text(page) -> str:
    try:
        body = page.locator("body")
        text = body.inner_text(timeout=2000)
    except Exception:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)
    if len(text) > MAX_VISIBLE_TEXT:
        text = text[:MAX_VISIBLE_TEXT]
    return text.strip()


def extract_elements(page) -> list:
    """Extract all interactive elements in a single JS batch call for speed."""
    try:
        raw = page.evaluate("""() => {
            const selectors = 'a, button, input:not([type="hidden"]), select, textarea, [role="button"], [role="link"], [role="textbox"], [role="searchbox"], [role="checkbox"], [role="combobox"], [role="spinbutton"]';
            const els = document.querySelectorAll(selectors);
            const results = [];
            els.forEach((el, idx) => {
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 && rect.height === 0) return;
                const tag = el.tagName.toLowerCase();
                const type = (el.getAttribute('type') || '').toLowerCase();
                const testid = el.getAttribute('data-testid') || '';
                const domId = el.id || '';
                const name = el.getAttribute('name') || '';
                const disabled = el.disabled || false;
                const ariaLabel = el.getAttribute('aria-label') || '';
                let labelText = '';
                const lbl = document.querySelector('label[for="' + el.id + '"]');
                if (lbl) labelText = lbl.textContent.trim().substring(0, 100);
                let text = '';
                if (tag === 'a' || tag === 'button') {
                    text = (el.textContent || '').trim().substring(0, 200);
                }
                let value = el.value || '';
                let options = [];
                if (tag === 'select') {
                    for (let i = 0; i < el.options.length; i++) {
                        options.push({value: el.options[i].value || '', label: el.options[i].textContent.trim()});
                    }
                }
                let role = tag;
                if (tag === 'select') role = 'combobox';
                else if (tag === 'textarea') role = 'textarea';
                else if (tag === 'button') role = 'button';
                else if (tag === 'a') role = 'link';
                else if (tag === 'input') {
                    if (type === 'search') role = 'searchbox';
                    else if (type === 'checkbox') role = 'checkbox';
                    else if (type === 'number') role = 'spinbutton';
                    else role = 'textbox';
                }
                const displayName = ariaLabel || labelText || testid || name || domId || text || 'unnamed_' + tag;
                results.push({
                    tag, role, type, testid, domId, name, disabled, ariaLabel,
                    labelText, text: text.substring(0, 200), value: value.substring(0, 200),
                    options, displayName: displayName.substring(0, 100),
                    idx
                });
            });
            return results;
        }""")
    except Exception:
        return []

    elements = []
    seen_ids = set()
    for item in (raw or []):
        role = item.get("role", "unknown")
        if role not in INTERACTIVE_ROLES and role not in ("unknown",):
            pass  # Keep all interactive roles we might miss

        testid = item.get("testid", "")
        dom_id = item.get("domId", "")
        name_attr = item.get("name", "")
        tag = item.get("tag", "")
        base_id = testid or dom_id or name_attr or f"{tag}_{item['idx']}"
        element_id = base_id
        dedup = 0
        while element_id in seen_ids:
            dedup += 1
            element_id = f"{base_id}_{dedup}"
        seen_ids.add(element_id)

        elements.append(ElementDescriptor(
            element_id=element_id,
            role=role,
            tag=tag,
            name=item.get("displayName", "")[:100],
            text=item.get("text", "")[:200],
            value=item.get("value", "")[:200],
            input_type=item.get("type", ""),
            testid=testid,
            options=item.get("options", []),
            disabled=item.get("disabled", False),
        ))

    return elements


def build_observation(page, task_id: str, episode_id: str, instruction: str,
                      step_index: int, last_action_result: dict = None,
                      terminal: bool = False) -> Observation:
    url = page.url
    parsed = urlparse(url)
    path = parsed.path
    page_type = _classify_page(path)

    title = page.title()
    visible_text = extract_visible_text(page)
    text_truncated = len(visible_text) >= MAX_VISIBLE_TEXT
    elements = extract_elements(page)

    return Observation(
        task_id=task_id, episode_id=episode_id, instruction=instruction,
        step_index=step_index, url=url, path=path, page_type=page_type,
        title=title, visible_text=visible_text, text_truncated=text_truncated,
        elements=elements, last_action_result=last_action_result, terminal=terminal,
    )
