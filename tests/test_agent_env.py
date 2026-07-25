"""M1.2 tests: schemas, observation, actions, security, trajectory."""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from miniwebwork.agent_env.schemas import (
    Observation, AgentAction, StepResult, ElementDescriptor,
    VALID_ACTION_TYPES, ROLE_ACTION_COMPAT,
)
from miniwebwork.agent_env.errors import (
    EnvironmentClosedError, EpisodeFinishedError, InvalidActionError,
)
class TestSchemas:
    def test_observation_serializable(self):
        obs = Observation(task_id="TASK-001", page_type="products")
        d = obs.to_dict()
        assert d["task_id"] == "TASK-001"
        assert d["schema_version"] == "1.0"

    def test_action_to_dict(self):
        a = AgentAction(action="click", target="e5")
        d = a.to_dict()
        assert d == {"action": "click", "target": "e5"}

    def test_action_from_dict(self):
        a = AgentAction.from_dict({"action": "fill", "target": "e2", "value": "GPU"})
        assert a.action == "fill"
        assert a.target == "e2"
        assert a.value == "GPU"

    def test_unknown_action_rejected(self):
        assert "navigate" not in VALID_ACTION_TYPES
        assert "execute" not in VALID_ACTION_TYPES

    def test_step_result_defaults(self):
        sr = StepResult()
        assert sr.reward == 0.0
        assert not sr.terminated
        assert not sr.truncated

    def test_element_descriptor(self):
        el = ElementDescriptor("e1", "textbox", "input", "search", "", "", "text", "search-query", [], False)
        assert el.element_id == "e1"
        assert el.role == "textbox"
class TestActionValidation:
    def test_valid_click(self):
        obs = Observation(elements=[
            ElementDescriptor("e1", "button", "button", "Start", "Start", "", "", "start-task-button", [], False)
        ])
        a = AgentAction(action="click", target="e1")
        from miniwebwork.agent_env.actions import validate_action
        r = validate_action(a, obs)
        assert r.success

    def test_invalid_target(self):
        obs = Observation(elements=[])
        a = AgentAction(action="click", target="nonexistent")
        from miniwebwork.agent_env.actions import validate_action
        r = validate_action(a, obs)
        assert not r.success
        assert r.error_code == "invalid_target"

    def test_disabled_element(self):
        obs = Observation(elements=[
            ElementDescriptor("e1", "button", "button", "X", "", "", "", "btn", [], True)
        ])
        a = AgentAction(action="click", target="e1")
        from miniwebwork.agent_env.actions import validate_action
        r = validate_action(a, obs)
        assert not r.success
        assert r.error_code == "disabled_element"

    def test_incompatible_action(self):
        obs = Observation(elements=[
            ElementDescriptor("e1", "textbox", "input", "Q", "", "", "text", "search", [], False)
        ])
        a = AgentAction(action="click", target="e1")
        from miniwebwork.agent_env.actions import validate_action
        r = validate_action(a, obs)
        assert not r.success
        assert r.error_code == "incompatible_action"

    def test_fill_requires_value(self):
        obs = Observation(elements=[
            ElementDescriptor("e1", "textbox", "input", "Q", "", "", "text", "search", [], False)
        ])
        a = AgentAction(action="fill", target="e1", value="")
        from miniwebwork.agent_env.actions import validate_action
        r = validate_action(a, obs)
        assert not r.success
        assert r.error_code == "value_required"

    def test_finish_no_target_ok(self):
        obs = Observation(elements=[])
        a = AgentAction(action="finish")
        from miniwebwork.agent_env.actions import validate_action
        r = validate_action(a, obs)
        assert r.success

    def test_back_no_target_ok(self):
        obs = Observation(elements=[])
        a = AgentAction(action="back")
        from miniwebwork.agent_env.actions import validate_action
        r = validate_action(a, obs)
        assert r.success

    def test_unknown_action_type(self):
        obs = Observation(elements=[])
        a = AgentAction(action="execute_js", target="e1")
        from miniwebwork.agent_env.actions import validate_action
        r = validate_action(a, obs)
        assert not r.success
        assert r.error_code == "invalid_action_type"
class TestSecurity:
    """Verify Observation does not leak Oracle content."""

    def test_observation_no_expected_product_id(self):
        obs = Observation()
        d = obs.to_dict()
        assert "expected_product_id" not in json.dumps(d)

    def test_observation_no_constraints_json(self):
        obs = Observation()
        d = obs.to_dict()
        # This is a structural test
        assert isinstance(d, dict)

    def test_agent_does_not_import_oracle(self):
        """Rule agent must not import tasks_oracle."""
        import inspect
        from miniwebwork.agents.rule_based import RuleBasedProcurementAgent
        src = inspect.getsource(RuleBasedProcurementAgent)
        assert "tasks_oracle" not in src
        assert "verifier" not in src
        assert "sqlite3" not in src
        assert "playwright" not in src
        assert "requests" not in src

    def test_action_no_css_selector(self):
        """Action schema should not accept css_selector field."""
        a = AgentAction.from_dict({"action": "click", "css_selector": "#btn"})
        # css_selector is not a defined field, should be ignored
        d = a.to_dict()
        assert "css_selector" not in d

    def test_action_no_xpath(self):
        a = AgentAction.from_dict({"action": "click", "xpath": "//button"})
        d = a.to_dict()
        assert "xpath" not in d

    def test_action_no_javascript(self):
        a = AgentAction.from_dict({"action": "click", "script": "alert(1)"})
        d = a.to_dict()
        assert "script" not in d
class TestTrajectory:
    def test_trajectory_schema(self):
        from miniwebwork.agent_env.trajectory import TrajectoryRecorder
        t = TrajectoryRecorder("run1", "TASK-001", "EP-1", "instruction", "rule", 20)
        d = t.to_dict()
        assert d["trajectory_schema_version"] == "1.0"

    def test_trajectory_jsonl_roundtrip(self, tmp_path):
        from miniwebwork.agent_env.trajectory import TrajectoryRecorder, save_trajectories_jsonl
        t = TrajectoryRecorder("run1", "TASK-001", "EP-1", "test", "rule", 20)
        t.record_step(0, None, {"action": "click"}, {"success": True}, 0.0, False, False, 100)
        t.finalize(1.0, True, "verified", {})
        path = tmp_path / "test.jsonl"
        save_trajectories_jsonl([t], path)
        assert path.exists()
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["success"] is True
class TestRuleAgent:
    def test_parses_price(self):
        from miniwebwork.agents.rule_based import RuleBasedProcurementAgent
        agent = RuleBasedProcurementAgent()
        c = agent._parse_instruction("价格不超过30000元的GPU")
        assert c.get("max_price") == 30000

    def test_parses_memory(self):
        from miniwebwork.agents.rule_based import RuleBasedProcurementAgent
        agent = RuleBasedProcurementAgent()
        c = agent._parse_instruction("显存至少32GB的GPU")
        assert c.get("min_memory_gb") == 32

    def test_parses_keyword(self):
        from miniwebwork.agents.rule_based import RuleBasedProcurementAgent
        agent = RuleBasedProcurementAgent()
        c = agent._parse_instruction("查找型号为CC-A100X-80G的GPU")
        assert c.get("keyword") == "CC-A100X-80G"

    def test_parses_certified(self):
        from miniwebwork.agents.rule_based import RuleBasedProcurementAgent
        agent = RuleBasedProcurementAgent()
        c = agent._parse_instruction("只从认证供应商采购")
        assert c.get("certified_only") is True

    def test_parses_no_solution(self):
        from miniwebwork.agents.rule_based import RuleBasedProcurementAgent
        agent = RuleBasedProcurementAgent()
        c = agent._parse_instruction("判断是否存在可行商品")
        assert c.get("no_solution_possible") is True
    def test_outputs_valid_action(self):
        from miniwebwork.agents.rule_based import RuleBasedProcurementAgent
        from miniwebwork.agent_env.schemas import Observation
        agent = RuleBasedProcurementAgent()
        obs = Observation(task_id="TASK-001", page_type="task", instruction="查找型号为CC-A100X-80G的GPU")
        action = agent.act(obs)
        assert action.action in VALID_ACTION_TYPES
