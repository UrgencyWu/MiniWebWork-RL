from __future__ import annotations

from pathlib import Path

import pytest

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent
from miniwebwork.data_generation.m4_long_horizon import build_long_horizon_dataset
from miniwebwork.models import FAILURE_REQUIRED_SUPPLIER_NOT_INSPECTED
from miniwebwork.tasks import get_oracle


@pytest.fixture(scope="module")
def long_horizon_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("m4-long-horizon-browser")
    task_root = root / "tasks"
    seed_root = root / "seed"
    build_long_horizon_dataset(task_root, seed_dir=seed_root)
    return task_root, seed_root


def _normalized_action(observation, action):
    payload = {"action": action.action}
    if action.target:
        matches = [
            element.testid
            for element in observation.elements
            if element.element_id == action.target
        ]
        assert len(matches) == 1
        payload["target_testid"] = matches[0]
    if action.value:
        payload["value"] = action.value
    if action.checked is not None:
        payload["checked"] = action.checked
    return payload


def _run(task_dir: Path, seed_dir: Path, task_id: str, oracle: dict):
    trace = []
    with ProcurementBrowserEnv(
        max_steps=20,
        run_id=f"long_horizon_browser_{task_id.lower()}",
        headless=True,
        keep_db=False,
        task_dir=task_dir,
        seed_dir=seed_dir,
    ) as environment:
        observation = environment.reset(task_id)
        expert = OracleExpertProcurementAgent(oracle, max_steps=20)
        terminal = None
        for _ in range(20):
            action = expert.act(observation)
            trace.append(_normalized_action(observation, action))
            terminal = environment.step(action)
            observation = terminal.observation
            if terminal.terminated or terminal.truncated:
                break
    assert terminal is not None
    return terminal, trace


@pytest.mark.browser
@pytest.mark.parametrize(
    ("suffix", "expected_horizon"),
    (
        ("EXACT_PRODUCT", 7),
        ("CHEAPEST_FEASIBLE", 12),
        ("HIGHEST_RELIABILITY_SUPPLIER", 18),
        ("NO_FEASIBLE_PRODUCT", 10),
    ),
)
def test_reference_trace_is_executable_and_exact(
    long_horizon_source, suffix: str, expected_horizon: int
):
    task_root, seed_root = long_horizon_source
    task_dir = task_root / "train"
    task_id = f"M4-LH-TRAIN-W001-{suffix}"
    oracle = get_oracle(task_id, task_dir=task_dir)
    assert oracle is not None
    terminal, trace = _run(task_dir, seed_root, task_id, oracle)
    assert terminal.terminated is True
    assert terminal.truncated is False
    assert terminal.reward == 1.0
    assert len(trace) == expected_horizon
    assert trace == oracle["reference_trace"]


@pytest.mark.browser
def test_correct_product_without_supplier_inspection_receives_zero_reward(
    long_horizon_source,
):
    task_root, seed_root = long_horizon_source
    task_dir = task_root / "train"
    task_id = "M4-LH-TRAIN-W001-HIGHEST_RELIABILITY_SUPPLIER"
    oracle = get_oracle(task_id, task_dir=task_dir)
    assert oracle is not None
    cheating_oracle = dict(oracle, workflow_requirements={})
    terminal, trace = _run(task_dir, seed_root, task_id, cheating_oracle)
    assert terminal.terminated is True
    assert terminal.truncated is False
    assert terminal.reward == 0.0
    assert FAILURE_REQUIRED_SUPPLIER_NOT_INSPECTED in terminal.info["failure_reasons"]
    assert not any(
        action.get("target_testid", "").startswith("supplier-link-")
        for action in trace
    )
