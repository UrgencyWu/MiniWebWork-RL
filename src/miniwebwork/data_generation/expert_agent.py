"""OracleExpertProcurementAgent — uses Oracle for decisions, interacts only via Environment."""

from ..agent_env.schemas import AgentAction, Observation


class OracleExpertProcurementAgent:
    """Expert agent that reads Oracle for task constraints/answer,
    but executes all actions through the standard Environment interface."""

    def __init__(self, oracle: dict, max_steps=15):
        self._oracle = oracle
        self._max_steps = max_steps
        self._step = 0
        self._state = {}
        self._selection_done = False

    def reset(self):
        self._step = 0
        self._state = {}
        self._selection_done = False

    def act(self, observation: Observation) -> AgentAction:
        self._step += 1
        pt = observation.page_type
        elements = observation.elements
        c = self._oracle.get("constraints", {})
        expected_pid = self._oracle.get("expected_product_id", "")
        expected_decision = self._oracle.get("expected_decision_type", "select_product")
        obj = self._oracle.get("objective", "")

        if pt == "task":
            return self._click_testid(elements, "start-task-button")

        if pt == "products":
            return self._handle_products(elements, c, expected_pid, expected_decision, obj, observation.visible_text)

        if pt == "product_detail":
            if expected_decision == "select_product" and not self._selection_done:
                self._selection_done = True
                return self._click_testid(elements, "select-product")
            return AgentAction(action="back")

        if pt == "supplier_detail":
            return AgentAction(action="back")

        if pt == "procurement_form":
            if not self._state.get("justification_filled"):
                self._state["justification_filled"] = True
                for e in elements:
                    if e.testid == "procurement-justification":
                        return AgentAction(action="fill", target=e.element_id,
                                           value="根据任务要求完成采购。")
            return self._click_testid(elements, "submit-procurement")

        if pt == "procurement_result":
            return AgentAction(action="finish")

        return AgentAction(action="finish")

    def _handle_products(self, elements, c, expected_pid, expected_decision, obj, text):
        # Step 1: Fill keyword if exact_product
        if obj == "exact_product" and not self._state.get("keyword_filled"):
            kw = c.get("keyword", "")
            if kw:
                self._state["keyword_filled"] = True
                for e in elements:
                    if e.testid == "search-query":
                        return AgentAction(action="fill", target=e.element_id, value=kw)

        # Step 2: Fill filters
        filters = [
            ("category", "filter-category", "select"),
            ("max_price", "filter-max-price", "fill"),
            ("min_memory_gb", "filter-min-memory", "fill"),
            ("max_delivery_days", "filter-max-delivery", "fill"),
            ("min_supplier_rating", "filter-min-rating", "fill"),
            ("min_warranty_months", "filter-min-warranty", "fill"),
            ("supplier_region", "filter-region", "select"),
        ]
        for ckey, testid, atype in filters:
            if c.get(ckey) and not self._state.get(f"filled_{ckey}"):
                self._state[f"filled_{ckey}"] = True
                for e in elements:
                    if e.testid == testid:
                        if atype == "select":
                            return AgentAction(action="select", target=e.element_id, value=str(c[ckey]))
                        return AgentAction(action="fill", target=e.element_id, value=str(c[ckey]))

        # Step 3: Check certified_only
        if c.get("certified_only") and not self._state.get("checked_certified"):
            self._state["checked_certified"] = True
            for e in elements:
                if e.testid == "filter-certified":
                    return AgentAction(action="check", target=e.element_id, checked=True)

        # Step 4: Check in_stock_only
        if c.get("in_stock_only") and not self._state.get("checked_stock"):
            self._state["checked_stock"] = True
            for e in elements:
                if e.testid == "filter-in-stock":
                    return AgentAction(action="check", target=e.element_id, checked=True)

        # Step 5: Submit filters
        if not self._state.get("filters_submitted"):
            self._state["filters_submitted"] = True
            for e in elements:
                if e.testid == "apply-filters":
                    return AgentAction(action="click", target=e.element_id)

        # Step 6: If no_solution expected, declare it
        if expected_decision == "no_solution" and not self._state.get("declared"):
            self._state["declared"] = True
            for e in elements:
                if e.testid == "declare-no-solution":
                    return AgentAction(action="click", target=e.element_id)

        # Step 7: Click the expected product link
        if expected_pid and not self._selection_done:
            for e in elements:
                if e.testid == f"product-link-{expected_pid}":
                    self._selection_done = True
                    return AgentAction(action="click", target=e.element_id)
            # Try to find by any product link
            for e in elements:
                if e.testid and e.testid.startswith("product-link-"):
                    pid = e.testid.replace("product-link-", "")
                    if pid == expected_pid:
                        self._selection_done = True
                        return AgentAction(action="click", target=e.element_id)

        return AgentAction(action="finish")

    def _click_testid(self, elements, testid):
        for e in elements:
            if e.testid == testid and not e.disabled:
                return AgentAction(action="click", target=e.element_id)
        return AgentAction(action="finish")
