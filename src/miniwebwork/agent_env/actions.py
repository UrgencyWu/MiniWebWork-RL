"""Action validation and execution on Playwright page."""

from typing import Optional

from .schemas import (
    AgentAction, ActionResult, ElementDescriptor, Observation,
    VALID_ACTION_TYPES, ROLE_ACTION_COMPAT,
)
from .errors import InvalidActionError


def _find_element(page, element_id: str, elements: list) -> Optional[dict]:
    """Find a Playwright locator for an element_id."""
    for el in elements:
        if el.element_id == element_id:
            # Try to locate by testid first, then id, then name
            if el.testid:
                try:
                    loc = page.locator(f'[data-testid="{el.testid}"]').first
                    if loc.is_visible():
                        return {"locator": loc, "descriptor": el}
                except Exception:
                    pass
            if el.name and not el.name.startswith("unnamed_"):
                # Try by name attribute
                pass
            # Fallback: use nth match of same tag with same role
            try:
                # Find by data-testid as primary mechanism
                if el.testid:
                    loc = page.locator(f'[data-testid="{el.testid}"]').first
                    if loc.is_visible():
                        return {"locator": loc, "descriptor": el}
            except Exception:
                pass
            return None
    return None


def validate_action(action: AgentAction, observation: Observation) -> ActionResult:
    """Validate an action against the current observation. Returns error ActionResult or None."""
    # Check action type
    if action.action not in VALID_ACTION_TYPES:
        return ActionResult(False, "invalid_action_type",
                           f"Unknown action: {action.action}")

    # finish and back don't need target
    if action.action in ("finish", "back"):
        return ActionResult(True)

    # Other actions need target
    if not action.target:
        return ActionResult(False, "invalid_target", "Action requires a target element_id")

    # Find target element
    target_el = None
    for el in observation.elements:
        if el.element_id == action.target:
            target_el = el
            break

    if target_el is None:
        return ActionResult(False, "invalid_target",
                           f"Element '{action.target}' not found in current observation")

    if target_el.disabled:
        return ActionResult(False, "disabled_element",
                           f"Element '{action.target}' is disabled")

    # Check compatibility
    role = target_el.role
    allowed = ROLE_ACTION_COMPAT.get(role, set())
    if action.action not in allowed:
        return ActionResult(False, "incompatible_action",
                           f"Cannot {action.action} on {role} element '{action.target}'")

    # fill requires value
    if action.action == "fill" and not action.value:
        return ActionResult(False, "value_required",
                           "fill action requires a value")

    # Value length limit
    if len(action.value) > 500:
        return ActionResult(False, "value_too_long",
                           f"Value too long ({len(action.value)} > 500)")

    return ActionResult(True)


def execute_action(action: AgentAction, observation: Observation, page) -> ActionResult:
    """Execute a validated action on the page. Returns ActionResult."""
    # Validate first
    validation = validate_action(action, observation)
    if not validation.success:
        return validation

    try:
        if action.action == "finish":
            # No page operation needed
            return ActionResult(True, page_changed=False)

        if action.action == "back":
            page.go_back(timeout=5000)
            page.wait_for_timeout(200)
            return ActionResult(True, page_changed=True)

        # Find locator
        target_el = None
        for el in observation.elements:
            if el.element_id == action.target:
                target_el = el
                break

        loc_info = _find_element(page, action.target, [target_el])
        if loc_info is None:
            return ActionResult(False, "stale_target",
                               f"Cannot locate '{action.target}' on page")

        loc = loc_info["locator"]
        desc = loc_info["descriptor"]

        if action.action == "click":
            loc.click(timeout=5000)
            page.wait_for_timeout(300)
            return ActionResult(True, page_changed=True)

        elif action.action == "fill":
            loc.fill(action.value, timeout=5000)
            page.wait_for_timeout(100)
            return ActionResult(True, page_changed=True)

        elif action.action == "check":
            checked = action.checked if action.checked is not None else True
            if checked:
                loc.check(timeout=5000)
            else:
                loc.uncheck(timeout=5000)
            page.wait_for_timeout(100)
            return ActionResult(True, page_changed=True)

        elif action.action == "select":
            loc.select_option(action.value, timeout=5000)
            page.wait_for_timeout(100)
            return ActionResult(True, page_changed=True)

        elif action.action == "submit":
            loc.click(timeout=5000)
            page.wait_for_timeout(300)
            return ActionResult(True, page_changed=True)

        return ActionResult(False, "invalid_action_type", f"Unhandled: {action.action}")

    except Exception as e:
        return ActionResult(False, "browser_error", str(e)[:200], page_changed=False)
