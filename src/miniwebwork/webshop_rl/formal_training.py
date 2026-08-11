"""Frozen contract and recovery primitives for the six M5 online runs."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..long_horizon_rl.contracts import SHA256_PATTERN, directory_sha256, sha256_file, sha256_json
from ..m5_webshop_protocol import eligible_goal_indices, task_id_for_goal_index
from .credit import ANCHOR_METHOD, BASELINE_METHOD

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAL_PLAN_PATH = PROJECT_ROOT / "data" / "m5_webshop_formal_plan_v1.json"
FORMAL_ROOT = PROJECT_ROOT / "outputs" / "m5_webshop_credit_assignment_v1" / "formal" / "online"
PLAN_SCHEMA = "m5_webshop_formal_plan_v1"
READINESS_SCHEMA = "m5_webshop_readiness_v1"
AUTHORIZATION_SCHEMA = "m5_webshop_formal_authorization_v1"
ITERATION_REPORT_SCHEMA = "m5_webshop_formal_iteration_v1"
RUN_REPORT_SCHEMA = "m5_webshop_formal_run_v1"
METHODS = (BASELINE_METHOD, ANCHOR_METHOD)
SEEDS = (20260801, 20260802, 20260803)
TASK_ORDER_NAMESPACE = "m5_webshop_formal_train_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return sha256_json(value)


def validate_self_hashed(payload: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == schema, f"M5 {schema} schema drift")
    _require(value.get("content_sha256") == self_hash(value), f"M5 {schema} self-hash drift")
    return value


def validate_formal_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(payload)
    _require(plan.get("schema_version") == PLAN_SCHEMA, "M5 formal plan schema drift")
    _require(plan.get("study_id") == "m5_webshop_credit_assignment_v1", "M5 formal plan study drift")
    _require(plan.get("lifecycle_state") == "ready_pending_user_authorization", "M5 formal plan lifecycle drift")
    _require(plan.get("formal_submission_allowed") is False, "M5 plan must not authorize submission")
    matrix = plan.get("matrix")
    _require(isinstance(matrix, Mapping), "M5 formal matrix is missing")
    _require(matrix.get("methods") == list(METHODS), "M5 formal methods drift")
    _require(matrix.get("seeds") == list(SEEDS), "M5 formal seeds drift")
    _require(matrix.get("logical_online_runs") == 6, "M5 formal run count drift")
    budget = plan.get("online_budget")
    _require(isinstance(budget, Mapping), "M5 formal budget is missing")
    maximum_group_tokens = (
        int(budget.get("group_size", 0))
        * int(budget.get("maximum_model_turns", 0))
        * int(budget.get("maximum_new_tokens_per_turn", 0))
    )
    _require(maximum_group_tokens == 9216, "M5 maximum atomic K4 token bound drift")
    _require(budget.get("maximum_atomic_group_token_reservation") == maximum_group_tokens, "M5 reservation drift")
    cap = budget.get("generated_action_token_cap_per_run")
    _require(cap == 500000, "M5 formal token cap drift")
    _require(budget.get("minimum_final_billed_action_tokens") == cap - maximum_group_tokens, "M5 terminal budget window drift")
    _require(budget.get("tasks_per_iteration") == 32, "M5 formal iteration width drift")
    _require(budget.get("parallel_lanes") == 32 and budget.get("shared_prefix_turns") == 1, "M5 formal branching drift")
    resources = plan.get("resource_contract")
    _require(isinstance(resources, Mapping), "M5 resource contract is missing")
    _require(
        resources.get("wall_time_per_allocation") == "24:00:00"
        and resources.get("gpus_per_run") == 1
        and resources.get("cpus_per_run") == 8
        and resources.get("memory_gib_per_run") == 32,
        "M5 formal per-run resources drift",
    )
    _require(resources.get("maximum_parallel_online_runs") == 6, "M5 formal parallelism drift")
    isolation = plan.get("data_isolation")
    _require(isinstance(isolation, Mapping) and "test" in isolation, "M5 data isolation contract is missing")
    success = plan.get("success_contract")
    _require(
        isinstance(success, Mapping)
        and success.get("attempt_infrastructure_valid_fraction_minimum") == 0.98
        and success.get("committed_token_fraction_minimum") == 0.90
        and success.get("cumulative_optimizer_updates_minimum") == 2,
        "M5 formal success contract drift",
    )
    ordering = plan.get("ordering_contract")
    _require(
        isinstance(ordering, Mapping)
        and ordering.get("explicit_user_authorization_before_any_formal_submission") is True
        and ordering.get("all_six_training_artifacts_before_frozen_test") is True,
        "M5 formal ordering contract drift",
    )
    evidence = plan.get("preflight_evidence")
    _require(
        isinstance(evidence, Mapping)
        and evidence.get("slurm_job_id") == 2139
        and evidence.get("state") == "COMPLETED"
        and evidence.get("exit_code") == "0:0"
        and evidence.get("result") == "passed",
        "M5 formal preflight evidence drift",
    )
    for field in ("protocol_sha256", "report_content_sha256", "sft_adapter_sha256"):
        _require(SHA256_PATTERN.fullmatch(str(evidence.get(field, ""))) is not None, f"M5 preflight {field} drift")
    return plan


def load_formal_plan(path: Path = FORMAL_PLAN_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = validate_formal_plan(_json(resolved))
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}


def formal_task_order(seed: int) -> tuple[str, ...]:
    _require(seed in SEEDS, "M5 formal seed drift")
    task_ids = tuple(task_id_for_goal_index(index) for index in eligible_goal_indices("train"))
    return tuple(
        sorted(
            task_ids,
            key=lambda task_id: (
                hashlib.sha256(f"{TASK_ORDER_NAMESPACE}|{seed}|{task_id}".encode("utf-8")).digest(),
                task_id,
            ),
        )
    )


def formal_run_root(method: str, seed: int, root: Path = FORMAL_ROOT) -> Path:
    _require(method in METHODS, "M5 formal method drift")
    _require(seed in SEEDS, "M5 formal seed drift")
    return Path(root).expanduser().resolve() / method / f"seed_{seed}"


def validate_readiness(payload: Mapping[str, Any], *, git_sha: str, plan_sha256: str) -> dict[str, Any]:
    readiness = validate_self_hashed(payload, schema=READINESS_SCHEMA)
    _require(readiness.get("decision") == "READY_PENDING_USER_AUTHORIZATION", "M5 readiness decision drift")
    _require(readiness.get("ready") is True, "M5 readiness is not complete")
    _require(readiness.get("formal_submission_allowed") is False, "M5 readiness cannot authorize submission")
    _require(readiness.get("git_sha") == git_sha, "M5 readiness Git drift")
    _require(readiness.get("formal_plan_sha256") == plan_sha256, "M5 readiness/plan drift")
    _require(not readiness.get("unmet_gates"), "M5 readiness has unmet gates")
    return readiness


def load_readiness(path: Path, *, git_sha: str, plan_sha256: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = validate_readiness(_json(resolved), git_sha=git_sha, plan_sha256=plan_sha256)
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}


def validate_authorization(
    payload: Mapping[str, Any],
    *,
    git_sha: str,
    plan_sha256: str,
    readiness_sha256: str,
) -> dict[str, Any]:
    authorization = validate_self_hashed(payload, schema=AUTHORIZATION_SCHEMA)
    _require(authorization.get("formal_submission_allowed") is True, "M5 formal authorization is closed")
    _require(authorization.get("approval_scope") == "six_online_runs_only", "M5 authorization scope drift")
    _require(authorization.get("git_sha") == git_sha, "M5 authorization Git drift")
    _require(authorization.get("formal_plan_sha256") == plan_sha256, "M5 authorization/plan drift")
    _require(authorization.get("readiness_sha256") == readiness_sha256, "M5 authorization/readiness drift")
    _require(authorization.get("methods") == list(METHODS), "M5 authorization method drift")
    _require(authorization.get("seeds") == list(SEEDS), "M5 authorization seed drift")
    return authorization


def load_authorization(
    path: Path,
    *,
    git_sha: str,
    plan_sha256: str,
    readiness_sha256: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = validate_authorization(
        _json(resolved),
        git_sha=git_sha,
        plan_sha256=plan_sha256,
        readiness_sha256=readiness_sha256,
    )
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}


@dataclass
class TokenReservation:
    budget: "AtomicTokenBudget"
    reservation_id: str
    capacity: int
    remaining: int
    released: bool = False

    def charge(self, count: int) -> None:
        self.budget._charge(self, count)

    def release(self) -> None:
        self.budget._release(self)


class AtomicTokenBudget:
    """Thread-safe worst-case reservation around one atomic K4 attempt."""

    def __init__(self, *, cap: int, spent: int, reservation_size: int):
        _require(cap > 0 and 0 <= spent <= cap and 0 < reservation_size <= cap, "invalid M5 token budget")
        self.cap = cap
        self.spent = spent
        self.reservation_size = reservation_size
        self._active: dict[str, TokenReservation] = {}
        self._lock = threading.Lock()

    @property
    def reserved(self) -> int:
        with self._lock:
            return sum(item.remaining for item in self._active.values())

    def reserve(self, reservation_id: str) -> TokenReservation | None:
        _require(isinstance(reservation_id, str) and reservation_id, "M5 reservation id is empty")
        with self._lock:
            _require(reservation_id not in self._active, "duplicate M5 active reservation")
            reserved = sum(item.remaining for item in self._active.values())
            if self.spent + reserved + self.reservation_size > self.cap:
                return None
            value = TokenReservation(
                budget=self,
                reservation_id=reservation_id,
                capacity=self.reservation_size,
                remaining=self.reservation_size,
            )
            self._active[reservation_id] = value
            return value

    def _charge(self, reservation: TokenReservation, count: int) -> None:
        _require(isinstance(count, int) and not isinstance(count, bool) and count >= 0, "invalid M5 token charge")
        with self._lock:
            _require(self._active.get(reservation.reservation_id) is reservation, "inactive M5 token reservation")
            _require(not reservation.released and count <= reservation.remaining, "M5 token reservation exceeded")
            _require(self.spent + count <= self.cap, "M5 generated-action token cap exceeded")
            reservation.remaining -= count
            self.spent += count

    def _release(self, reservation: TokenReservation) -> None:
        with self._lock:
            if reservation.released:
                return
            _require(self._active.get(reservation.reservation_id) is reservation, "inactive M5 token reservation")
            self._active.pop(reservation.reservation_id)
            reservation.released = True


def validate_iteration_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_self_hashed(payload, schema=ITERATION_REPORT_SCHEMA)
    _require(report.get("complete") is True and report.get("formal_training") is True, "M5 iteration is incomplete")
    _require(report.get("method") in METHODS and report.get("seed") in SEEDS, "M5 iteration identity drift")
    _require(report.get("iteration_index", -1) >= 0, "M5 iteration index drift")
    _require(report.get("generated_action_tokens", 0) > 0, "M5 iteration cost is empty")
    _require(report.get("group_count", 0) > 0, "M5 iteration group set is empty")
    return report


def validate_run_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_self_hashed(payload, schema=RUN_REPORT_SCHEMA)
    _require(report.get("complete") is True and report.get("formal_training") is True, "M5 formal run is incomplete")
    _require(report.get("passed") is True, "M5 formal run did not pass")
    _require(report.get("method") in METHODS and report.get("seed") in SEEDS, "M5 formal run identity drift")
    _require(not report.get("unmet_gates"), "M5 formal run has unmet gates")
    _require(
        directory_sha256(Path(report["final_adapter"])) == report["final_adapter_sha256"],
        "M5 formal final adapter hash drift",
    )
    _require(
        sha256_file(Path(report["final_optimizer"])) == report["final_optimizer_sha256"],
        "M5 formal final optimizer hash drift",
    )
    _require(
        sha256_file(Path(report["generated_turn_ledger"])) == report["generated_turn_ledger_sha256"],
        "M5 formal generated-turn ledger hash drift",
    )
    return report
