from miniwebwork.agent_env.actions import validate_action
from miniwebwork.agent_env.schemas import AgentAction, ElementDescriptor, Observation


def test_submit_is_valid_for_button_role():
    observation = Observation(
        elements=[
            ElementDescriptor(
                element_id="submit-procurement",
                role="button",
                tag="button",
                name="提交采购决策",
                text="提交",
                value="",
                input_type="submit",
                testid="submit-procurement",
                options=[],
                disabled=False,
            )
        ]
    )

    result = validate_action(
        AgentAction(action="submit", target="submit-procurement"),
        observation,
    )

    assert result.success is True


def test_submit_still_rejects_non_button_role():
    observation = Observation(
        elements=[
            ElementDescriptor(
                element_id="query",
                role="textbox",
                tag="input",
                name="query",
                text="",
                value="",
                input_type="text",
                testid="query",
                options=[],
                disabled=False,
            )
        ]
    )

    result = validate_action(AgentAction(action="submit", target="query"), observation)

    assert result.success is False
    assert result.error_code == "incompatible_action"
