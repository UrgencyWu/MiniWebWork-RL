"""OracleExpertProcurementAgent v2 — proper state machine, no premature skip."""

from ..agent_env.schemas import AgentAction, Observation


class OracleExpertProcurementAgent:
    """Expert that reads Oracle constraints/answer, executes via Environment only."""

    def __init__(self, oracle: dict, max_steps=25):
        self._oracle = oracle
        self._max_steps = max_steps
        self.reset()

    def reset(self):
        self._step = 0
        self._state = {}          # Tracks which filters have been applied
        self._product_clicked = False  # True after clicking product link
        self._selection_confirmed = False  # True after clicking select-product

    def act(self, observation: Observation) -> AgentAction:
        self._step += 1
        pt = observation.page_type
        els = observation.elements
        c = self._oracle.get("constraints", {})
        expected_pid = self._oracle.get("expected_product_id", "")
        expected_decision = self._oracle.get("expected_decision_type", "select_product")
        obj = self._oracle.get("objective", "")

        if pt == "task":
            return self._click(els, "start-task-button")

        if pt == "products":
            return self._handle_products(els, c, expected_pid, expected_decision, obj)

        if pt == "product_detail":
            return self._handle_product_detail(els, expected_decision)

        if pt == "supplier_detail":
            return AgentAction(action="back")

        if pt == "procurement_form":
            return self._handle_form(els)

        if pt == "procurement_result":
            return AgentAction(action="finish")

        return AgentAction(action="finish")

    # ---- product_detail handler (FIXED: separate state from product link click) ----
    def _handle_product_detail(self, els, expected_decision):
        if expected_decision == "select_product" and not self._selection_confirmed:
            self._selection_confirmed = True
            result = self._click(els, "select-product")
            if result.action == "click":
                return result
            # Fallback: search for any button containing "选择"
            for e in els:
                if "选择" in (e.name or "") and e.role in ("link", "button"):
                    return AgentAction(action="click", target=e.element_id)
        return AgentAction(action="back")

    # ---- products page handler ----
    def _handle_products(self, els, c, expected_pid, expected_decision, obj):
        # 1) Fill keyword
        if obj == "exact_product" and not self._state.get("kw_filled"):
            kw = c.get("keyword", "")
            if kw:
                self._state["kw_filled"] = True
                e = self._find(els, "search-query")
                if e:
                    return AgentAction(action="fill", target=e.element_id, value=kw)

        # 2) Fill/select constraint fields (one per step)
        for ckey, testid, atype in [
            ("category", "filter-category", "select"),
            ("max_price", "filter-max-price", "fill"),
            ("min_memory_gb", "filter-min-memory", "fill"),
            ("max_delivery_days", "filter-max-delivery", "fill"),
            ("min_supplier_rating", "filter-min-rating", "fill"),
            ("min_warranty_months", "filter-min-warranty", "fill"),
            ("supplier_region", "filter-region", "select"),
        ]:
            if c.get(ckey) is not None and not self._state.get(f"f_{ckey}"):
                self._state[f"f_{ckey}"] = True
                e = self._find(els, testid)
                if e:
                    val = str(c[ckey])
                    if atype == "select":
                        return AgentAction(action="select", target=e.element_id, value=val)
                    return AgentAction(action="fill", target=e.element_id, value=val)

        # 3) Check certified
        if c.get("certified_only") and not self._state.get("chk_cert"):
            self._state["chk_cert"] = True
            e = self._find(els, "filter-certified")
            if e:
                return AgentAction(action="check", target=e.element_id, checked=True)

        # 4) Check in_stock
        if c.get("in_stock_only") and not self._state.get("chk_stock"):
            self._state["chk_stock"] = True
            e = self._find(els, "filter-in-stock")
            if e:
                return AgentAction(action="check", target=e.element_id, checked=True)

        # 5) Submit filters
        if not self._state.get("filt_done"):
            self._state["filt_done"] = True
            e = self._find(els, "apply-filters")
            if e:
                return AgentAction(action="click", target=e.element_id)

        # 6) If no_solution expected
        if expected_decision == "no_solution" and not self._state.get("declared"):
            self._state["declared"] = True
            e = self._find(els, "declare-no-solution")
            if e:
                return AgentAction(action="click", target=e.element_id)

        # 7) Click expected product link (sets _product_clicked, NOT _selection_confirmed)
        if expected_pid and not self._product_clicked:
            self._product_clicked = True
            e = self._find(els, f"product-link-{expected_pid}")
            if e:
                return AgentAction(action="click", target=e.element_id)
            # Try harder
            for elem in els:
                if elem.testid and elem.testid.endswith(f"-{expected_pid}"):
                    return AgentAction(action="click", target=elem.element_id)

        return AgentAction(action="finish")

    # ---- form handler ----
    def _handle_form(self, els):
        if not self._state.get("just_filled"):
            self._state["just_filled"] = True
            e = self._find(els, "procurement-justification")
            if e:
                return AgentAction(action="fill", target=e.element_id,
                                   value="根据任务要求完成采购。")
        e = self._find(els, "submit-procurement")
        if e:
            return AgentAction(action="click", target=e.element_id)
        return AgentAction(action="finish")

    # ---- helpers ----
    def _find(self, els, testid):
        for e in els:
            if e.testid == testid and not e.disabled:
                return e
        return None

    def _click(self, els, testid):
        e = self._find(els, testid)
        if e:
            return AgentAction(action="click", target=e.element_id)
        return AgentAction(action="finish")
