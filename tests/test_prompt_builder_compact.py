from types import SimpleNamespace

from miniwebwork.model_agent import prompt_builder


def _element(index: int, role: str, *, disabled: bool = False):
    return SimpleNamespace(
        element_id=f"e-{index}",
        role=role,
        name=f"name-{index}",
        testid=None,
        disabled=disabled,
        tag="button" if role == "button" else "a",
        value=None,
        options=None,
        text=f"text-{index}",
    )


def test_compact_prompt_selector_prioritizes_usable_controls_then_links_without_target_knowledge():
    controls = [_element(index, "button") for index in range(20)]
    controls.append(_element(100, "button", disabled=True))
    links = [_element(200 + index, "link") for index in range(20)]
    links.append(_element(300, "link", disabled=True))
    observation = SimpleNamespace(elements=[*controls, *links])

    serialized = prompt_builder._serialize_elements(observation)
    assert [row["element_id"] for row in serialized] == [
        *(f"e-{index}" for index in range(16)),
        *(f"e-{200 + index}" for index in range(16)),
    ]
    assert prompt_builder.visible_element_ids(observation) == {
        row["element_id"] for row in serialized
    }
    assert prompt_builder.context_contract() == {
        "prompt_version": "browser_agent_v3_compact",
        "max_visible_text_characters": 5000,
        "history_window": 5,
        "max_control_elements": 16,
        "max_link_elements": 16,
    }
