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


def _observation(page_type: str, path: str, visible_text: str, **hidden):
    values = {
        "elements": [],
        "task_id": "task-public",
        "instruction": "Compare public delivery reliability.",
        "url": f"http://127.0.0.1:8000{path}",
        "path": path,
        "page_type": page_type,
        "title": "Public supplier page",
        "step_index": 3,
        "visible_text": visible_text,
        "last_action_result": None,
    }
    values.update(hidden)
    return SimpleNamespace(**values)


def test_v4_public_evidence_memory_is_bounded_deduplicated_and_oracle_blind():
    memory = []
    first = _observation(
        "supplier_detail",
        "/suppliers/SUP-A?episode_id=private-run",
        "Supplier A Delivery reliability 93%",
        oracle={"expected_product_id": "SECRET-PRODUCT"},
        expected_product_id="SECRET-PRODUCT",
    )
    prompt_builder.update_evidence_memory(
        memory,
        first,
        version=prompt_builder.LONG_MEMORY_PROMPT_VERSION,
    )
    assert memory == [
        {
            "page_type": "supplier_detail",
            "path": "/suppliers/SUP-A",
            "title": "Public supplier page",
            "visible_text": "Supplier A Delivery reliability 93%",
        }
    ]

    refreshed = _observation(
        "supplier_detail",
        "/suppliers/SUP-A",
        "Supplier A Delivery reliability 94%",
    )
    prompt_builder.update_evidence_memory(
        memory,
        refreshed,
        version=prompt_builder.LONG_MEMORY_PROMPT_VERSION,
    )
    for index in range(9):
        prompt_builder.update_evidence_memory(
            memory,
            _observation(
                "supplier_detail",
                f"/suppliers/SUP-{index}",
                f"Supplier {index} Delivery reliability {80 + index}%",
            ),
            version=prompt_builder.LONG_MEMORY_PROMPT_VERSION,
        )
    assert len(memory) == prompt_builder.MAX_EVIDENCE_ENTRIES
    serialized = str(memory)
    assert "SECRET-PRODUCT" not in serialized
    assert "private-run" not in serialized


def test_v4_messages_expose_public_memory_while_v3_bytes_remain_memory_free():
    memory = [
        {
            "page_type": "supplier_detail",
            "path": "/suppliers/SUP-A",
            "title": "Supplier A",
            "visible_text": "Delivery reliability 0.93",
            "oracle_answer": "must never serialize",
        }
    ]
    current = _observation(
        "products", "/products?episode_id=private-current", "Catalogue"
    )
    v4 = prompt_builder.build_messages(
        current,
        [],
        version=prompt_builder.LONG_MEMORY_PROMPT_VERSION,
        evidence_memory=memory,
    )
    assert "## Public Evidence Memory (1 states; policy-visible only)" in v4[1]["content"]
    assert "Delivery reliability 0.93" in v4[1]["content"]
    assert "oracle_answer" not in v4[1]["content"]
    assert "private-current" not in v4[1]["content"]
    assert "127.0.0.1" not in v4[1]["content"]
    assert prompt_builder.context_contract(prompt_builder.LONG_MEMORY_PROMPT_VERSION) == {
        "prompt_version": "browser_agent_v4_long_memory",
        "max_visible_text_characters": 5000,
        "history_window": 5,
        "max_control_elements": 16,
        "max_link_elements": 16,
        "evidence_memory_contract": "public_observation_v1",
        "max_evidence_entries": 8,
        "max_evidence_text_characters_per_entry": 1000,
        "evidence_page_types": ["product_detail", "supplier_detail"],
        "current_url_contract": "path_only_without_origin_query_or_fragment",
    }

    v3 = prompt_builder.build_messages(current, [], evidence_memory=memory)
    assert "Public Evidence Memory" not in v3[1]["content"]
    assert "Delivery reliability 0.93" not in v3[1]["content"]
    assert "private-current" in v3[1]["content"]
