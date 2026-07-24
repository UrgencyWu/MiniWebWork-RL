"""Data models and constants for MiniWebWork-RL procurement environment."""

from dataclasses import dataclass, field
from typing import Optional

# Valid decision types
DECISION_SELECT_PRODUCT = "select_product"
DECISION_NO_SOLUTION = "no_solution"
VALID_DECISION_TYPES = {DECISION_SELECT_PRODUCT, DECISION_NO_SOLUTION}

# Valid episode statuses
EPISODE_ACTIVE = "active"
EPISODE_SUBMITTED = "submitted"
EPISODE_VERIFIED = "verified"
EPISODE_FAILED = "failed"
VALID_EPISODE_STATUSES = {EPISODE_ACTIVE, EPISODE_SUBMITTED, EPISODE_VERIFIED, EPISODE_FAILED}

# Valid task objectives
OBJECTIVE_EXACT_PRODUCT = "exact_product"
OBJECTIVE_CHEAPEST_FEASIBLE = "cheapest_feasible"
OBJECTIVE_HIGHEST_RATING_SUPPLIER = "highest_rating_supplier"
OBJECTIVE_NO_FEASIBLE_PRODUCT = "no_feasible_product"
VALID_OBJECTIVES = {
    OBJECTIVE_EXACT_PRODUCT,
    OBJECTIVE_CHEAPEST_FEASIBLE,
    OBJECTIVE_HIGHEST_RATING_SUPPLIER,
    OBJECTIVE_NO_FEASIBLE_PRODUCT,
}

# Verifier failure reason codes
FAILURE_MISSING_SUBMISSION = "missing_submission"
FAILURE_INVALID_EPISODE = "invalid_episode"
FAILURE_WRONG_DECISION_TYPE = "wrong_decision_type"
FAILURE_WRONG_PRODUCT = "wrong_product"
FAILURE_PRICE_CONSTRAINT_FAILED = "price_constraint_failed"
FAILURE_MEMORY_CONSTRAINT_FAILED = "memory_constraint_failed"
FAILURE_DELIVERY_CONSTRAINT_FAILED = "delivery_constraint_failed"
FAILURE_SUPPLIER_CERTIFICATION_FAILED = "supplier_certification_failed"
FAILURE_SUPPLIER_RATING_FAILED = "supplier_rating_failed"
FAILURE_REGION_CONSTRAINT_FAILED = "region_constraint_failed"
FAILURE_OUT_OF_STOCK = "out_of_stock"
FAILURE_WARRANTY_CONSTRAINT_FAILED = "warranty_constraint_failed"
FAILURE_OBJECTIVE_NOT_OPTIMAL = "objective_not_optimal"
FAILURE_FALSE_NO_SOLUTION = "false_no_solution"
FAILURE_EXPECTED_NO_SOLUTION = "expected_no_solution"


@dataclass
class TaskConstraints:
    """Structured constraints for a procurement task (Oracle side)."""
    category: Optional[str] = None
    keyword: Optional[str] = None
    max_price: Optional[float] = None
    min_memory_gb: Optional[int] = None
    max_delivery_days: Optional[int] = None
    min_supplier_rating: Optional[float] = None
    certified_only: Optional[bool] = None
    supplier_region: Optional[str] = None
    in_stock_only: bool = False
    min_warranty_months: Optional[int] = None


@dataclass
class VerificationResult:
    """Output of the deterministic verifier."""
    success: bool
    task_id: str
    episode_id: str
    submission_id: str = ""
    decision_type: str = ""
    selected_product_id: str = ""
    expected_decision_type: str = ""
    expected_product_id: str = ""
    constraints_satisfied: bool = False
    objective_satisfied: bool = False
    failure_reasons: list = field(default_factory=list)
    details: dict = field(default_factory=dict)
    verifier_version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "submission_id": self.submission_id,
            "decision_type": self.decision_type,
            "selected_product_id": self.selected_product_id,
            "expected_decision_type": self.expected_decision_type,
            "expected_product_id": self.expected_product_id,
            "constraints_satisfied": self.constraints_satisfied,
            "objective_satisfied": self.objective_satisfied,
            "failure_reasons": self.failure_reasons,
            "details": self.details,
            "verifier_version": self.verifier_version,
        }
