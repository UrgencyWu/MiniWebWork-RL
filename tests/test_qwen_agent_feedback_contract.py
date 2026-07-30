from types import SimpleNamespace

from miniwebwork.agent_env.schemas import AgentAction, Observation, StepResult
from miniwebwork.model_agent.qwen_agent import ModelActionAttempt, QwenBrowserAgent


class _Unused:
    pass


def _agent():
    return QwenBrowserAgent(_Unused(), _Unused(), _Unused())


def test_step_result_history_keeps_only_action_result():
    agent = _agent()
    attempt = ModelActionAttempt(
        model_turn_index=1,
        schema_valid=True,
        action=AgentAction(action="click", target="x"),
    )
    step_result = StepResult(
        observation=Observation(visible_text="must not enter history"),
        info={
            "action_result": {
                "success": True,
                "error_code": "",
                "message": "clicked",
                "page_changed": True,
            }
        },
    )

    agent.record_feedback(attempt, step_result, "products")

    entry = agent.history[0]
    assert entry["result"] == {
        "success": True,
        "error_code": "",
        "message": "clicked",
        "page_changed": True,
    }
    assert "observation" not in entry["result"]
    assert "visible_text" not in str(entry)


def test_schema_failure_is_recorded_truthfully():
    agent = _agent()
    attempt = ModelActionAttempt(
        model_turn_index=1,
        schema_valid=False,
        errors=["unknown_action"],
    )

    agent.record_feedback(attempt, None, "products")

    assert agent.history[0]["result"]["success"] is False
    assert agent.history[0]["result"]["error_code"] == "schema_invalid"
    assert agent.history[0]["result"]["message"] == "unknown_action"
