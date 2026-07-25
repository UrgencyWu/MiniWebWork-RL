"""
Rule-based procurement agent — simplified and robust.

Uses page-type dispatch and keyword/regex parsing from instruction.
Never accesses Oracle, database, or verifier.
"""

import re
from typing import Optional

from ..agent_env.schemas import AgentAction, Observation


class RuleBasedProcurementAgent:
    """Rule-based agent with page-type dispatch and instruction parsing."""

    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        self._step = 0
        self.reset()

    def reset(self):
        self._step = 0
        self._constraints = {}
        self._product_selected = False
        self._justification_filled = False
        self._keyword_filled = False
        self._filters_applied = False

    def act(self, observation: Observation) -> AgentAction:
        self._step += 1

        if self._step >= self.max_steps:
            return AgentAction(action="finish")

        pt = observation.page_type
        elements = observation.elements

        # Parse constraints once
        if not self._constraints:
            self._constraints = self._parse_instruction(observation.instruction)

        if pt == "task":
            return self._act_task(elements)
        elif pt == "products":
            return self._act_products(elements, observation.visible_text)
        elif pt == "product_detail":
            return self._act_product_detail(elements)
        elif pt == "supplier_detail":
            return AgentAction(action="back")
        elif pt == "procurement_form":
            return self._act_procurement_form(elements)
        elif pt == "procurement_result":
            return AgentAction(action="finish")
        else:
            return AgentAction(action="finish")

    def _act_task(self, elements) -> AgentAction:
        for e in elements:
            if e.testid == "start-task-button" and not e.disabled:
                return AgentAction(action="click", target=e.element_id)
        return AgentAction(action="finish")

    def _act_products(self, elements, text) -> AgentAction:
        c = self._constraints
        n_elements = len(elements)

        # If page has no elements, the extraction may have failed — try going back or finish
        if n_elements == 0:
            return AgentAction(action="finish")

        # Fill keyword if we have one
        if c.get("keyword") and not self._keyword_filled:
            for e in elements:
                if e.testid == "search-query" and not e.disabled:
                    self._keyword_filled = True
                    return AgentAction(action="fill", target=e.element_id,
                                        value=str(c["keyword"]))

        # Apply filters (once)
        if not self._filters_applied:
            for e in elements:
                if e.testid == "apply-filters" and not e.disabled:
                    self._filters_applied = True
                    return AgentAction(action="click", target=e.element_id)

        # Check for 0 results -> no_solution
        if "共 0 条" in text or "0 条结果" in text:
            for e in elements:
                if e.testid == "declare-no-solution" and not e.disabled:
                    return AgentAction(action="click", target=e.element_id)

        # Click first available product link
        if not self._product_selected:
            for e in elements:
                if e.testid and e.testid.startswith("product-link-") and not e.disabled:
                    self._product_selected = True
                    return AgentAction(action="click", target=e.element_id)

        return AgentAction(action="finish")

    def _act_product_detail(self, elements) -> AgentAction:
        # Try select-product by testid first
        for e in elements:
            if e.testid == "select-product" and not e.disabled:
                return AgentAction(action="click", target=e.element_id)
        # Fallback: look for any link/button with "选择" in name
        for e in elements:
            if ("选择" in e.name or "选择" in e.text) and e.role in ("link", "button") and not e.disabled:
                return AgentAction(action="click", target=e.element_id)
        # Fallback: if we see product detail content, try continue
        return AgentAction(action="finish")

    def _act_procurement_form(self, elements) -> AgentAction:
        # Fill justification if not done
        if not self._justification_filled:
            for e in elements:
                if e.testid == "procurement-justification" and not e.disabled:
                    self._justification_filled = True
                    return AgentAction(action="fill", target=e.element_id,
                                        value="按照任务要求进行采购。")
        # Submit
        for e in elements:
            if e.testid == "submit-procurement" and not e.disabled:
                return AgentAction(action="click", target=e.element_id)
        return AgentAction(action="finish")

    # ================================================================
    # Instruction parsing
    # ================================================================
    def _parse_instruction(self, text: str) -> dict:
        c = {}
        # Price
        m = re.search(r'价格不(?:超过|高于|大于)\s*(\d[\d,]*\.?\d*)\s*(?:元|万)?', text)
        if not m:
            m = re.search(r'预算(?:上限|不超过?)\s*(\d[\d,]*\.?\d*)\s*(?:元|万)?', text)
        if m:
            val = float(m.group(1).replace(",", ""))
            if "万" in m.group(0) or "万" in text[max(0,m.start()-5):m.end()+5]:
                val *= 10000
            c["max_price"] = val

        # Memory
        m = re.search(r'显存(?:至少|不小于?|≥|不少于)\s*(\d+)\s*GB', text)
        if m:
            c["min_memory_gb"] = int(m.group(1))

        # Delivery
        m = re.search(r'交付(?:时间)?(?:不超过?|≤|不大于)\s*(\d+)\s*天', text)
        if m:
            c["max_delivery_days"] = int(m.group(1))

        # Certification
        if "认证供应商" in text and ("只从" in text or "必须" in text):
            c["certified_only"] = True

        # Keyword (model number)
        m = re.search(r'型号(?:为|：|:)\s*([A-Za-z0-9\-]+)', text)
        if m:
            c["keyword"] = m.group(1)
        m = re.search(r'查找\s*([A-Z][A-Z0-9\-]{3,}[A-Z0-9])', text)
        if m:
            c["keyword"] = m.group(1)

        # No solution detection
        if "是否存在" in text or "判断是否存在" in text or "无可行" in text:
            c["no_solution_possible"] = True

        return c
