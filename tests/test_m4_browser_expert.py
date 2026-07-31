"""End-to-end contract: M4 constraints must be expressible through the real UI."""

from pathlib import Path

import pytest

from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent
from miniwebwork.tasks import get_oracle


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "data" / "tasks" / "m4_rlvr_v1" / "train"
SEED_DIR = ROOT / "data" / "seed_m4_rlvr_v1"


@pytest.mark.browser
@pytest.mark.parametrize(
    "task_suffix",
    ("EXACT_PRODUCT", "CHEAPEST_FEASIBLE", "HIGHEST_RATING_SUPPLIER", "NO_FEASIBLE_PRODUCT"),
)
def test_one_m4_world_per_task_family_is_executable_through_browser_ui(task_suffix: str):
    task_id = f"M4-TRAIN-W001-{task_suffix}"
    oracle = get_oracle(task_id, task_dir=TASK_DIR)
    assert oracle is not None
    with ProcurementBrowserEnv(
        max_steps=20,
        run_id=f"m4_expert_{task_suffix.lower()}",
        headless=True,
        keep_db=False,
        task_dir=TASK_DIR,
        seed_dir=SEED_DIR,
    ) as environment:
        observation = environment.reset(task_id)
        expert = OracleExpertProcurementAgent(oracle, max_steps=20)
        terminal = None
        for _ in range(20):
            terminal = environment.step(expert.act(observation))
            observation = terminal.observation
            if terminal.terminated or terminal.truncated:
                break
        assert terminal is not None
        assert terminal.terminated is True
        assert terminal.truncated is False
        assert terminal.reward == 1.0
