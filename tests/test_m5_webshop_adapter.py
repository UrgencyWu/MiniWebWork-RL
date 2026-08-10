from __future__ import annotations

import json

import httpx
import pytest

from miniwebwork.webshop_rl import prompt
from miniwebwork.webshop_rl.actions import WebShopCommand, parse_command_output
from miniwebwork.webshop_rl.credit import policy_context_signature, public_state_anchor_signature
from miniwebwork.webshop_rl.environment import (
    MAX_PUBLIC_ACTIONS,
    WebShopHTTPEnvironment,
    _bounded_public_actions,
)
from miniwebwork.webshop_rl.oracle import (
    MAX_ORACLE_TURNS,
    OraclePolicyFailure,
    _oracle_title_query,
    build_verified_oracle_trajectory,
)


class FakeWebShopServer:
    def __init__(self):
        self.step_commands: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/reset":
            assert body == {"goal_index": 1000}
            return httpx.Response(
                200,
                json={
                    "observation": "Amazon Shopping Game\nInstruction: buy a blue mug",
                    "env_state": {"page_type": "home", "query": "", "selected_options": {}},
                    "info": {
                        "goal_index": 1000,
                        "instruction": "buy a blue mug",
                        "asin": "HIDDEN-TARGET-ASIN",
                        "available_actions": ["search[<your query>]"],
                    },
                },
            )
        assert request.url.path == "/step"
        command = body["action"]
        self.step_commands.append(command)
        if command.startswith("search["):
            return httpx.Response(
                200,
                json={
                    "observation": "Search Results\n[1] B000TARGET | Blue mug",
                    "env_state": {"page_type": "search_results", "query": "blue mug", "selected_options": {}},
                    "reward": 0.0,
                    "done": False,
                    "info": {"available_actions": ["search[<your query>]", "click[B000TARGET]"]},
                },
            )
        return httpx.Response(
            200,
            json={
                "observation": "Episode complete.",
                "env_state": {"page_type": "done", "query": "blue mug", "asin": "B000TARGET"},
                "reward": 1.0,
                "done": True,
                "info": {
                    "success": True,
                    "task_score": 1.0,
                    "selected_asin": "B000TARGET",
                    "target_asin": "HIDDEN-TARGET-ASIN",
                    "available_actions": [],
                },
            },
        )


class OracleFakeServer:
    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/reset":
            return httpx.Response(
                200,
                json={
                    "observation": "Amazon Shopping Game\nInstruction: buy a navy blue mug",
                    "env_state": {"page_type": "home", "selected_options": {}},
                    "info": {
                        "instruction": "buy a navy blue mug",
                        "asin": "B000TARGET",
                        "available_actions": ["search[<your query>]"],
                    },
                },
            )
        command = body["action"]
        if command.startswith("search["):
            return httpx.Response(
                200,
                json={
                    "observation": "Search Results\nB000TARGET | Mug",
                    "env_state": {"page_type": "search_results", "query": "blue mug", "selected_options": {}},
                    "reward": 0.0,
                    "done": False,
                    "info": {"available_actions": ["search[<your query>]", "click[B000TARGET]"]},
                },
            )
        if command == "click[B000TARGET]":
            return httpx.Response(
                200,
                json={
                    "observation": "Product Page\nOptions: navy",
                    "env_state": {"page_type": "item", "asin": "B000TARGET", "selected_options": {}},
                    "reward": 0.0,
                    "done": False,
                    "info": {"available_actions": ["search[<your query>]", "click[navy]", "click[Buy Now]"]},
                },
            )
        if command == "click[navy]":
            return httpx.Response(
                200,
                json={
                    "observation": "Product Page\nOptions: navy selected",
                    "env_state": {
                        "page_type": "item",
                        "asin": "B000TARGET",
                        "selected_options": {"color": "navy"},
                    },
                    "reward": 0.0,
                    "done": False,
                    "info": {"available_actions": ["search[<your query>]", "click[navy]", "click[Buy Now]"]},
                },
            )
        assert command == "click[Buy Now]"
        return httpx.Response(
            200,
            json={
                "observation": "Episode complete.",
                "env_state": {"page_type": "done", "asin": "B000TARGET"},
                "reward": 1.0,
                "done": True,
                "info": {
                    "success": True,
                    "task_score": 1.0,
                    "selected_asin": "B000TARGET",
                    "target_asin": "B000TARGET",
                    "available_actions": [],
                },
            },
        )


def _environment(server: FakeWebShopServer) -> WebShopHTTPEnvironment:
    client = httpx.Client(transport=httpx.MockTransport(server), base_url="http://webshop.test")
    return WebShopHTTPEnvironment(split="train", client=client)


def test_parser_requires_one_command_key_and_normalizes_the_verb():
    parsed = parse_command_output('{"command":"SEARCH[blue mug]"}')
    assert parsed.schema_valid
    assert parsed.strict_json_success
    assert parsed.action == WebShopCommand("search[blue mug]")
    assert not parse_command_output('{"command":"search[x]","reason":"hidden"}').schema_valid
    assert not parse_command_output('{"command":"open[x]"}').schema_valid


def test_adapter_hides_target_fields_and_rejects_unlisted_click_without_http_step():
    server = FakeWebShopServer()
    environment = _environment(server)
    observation = environment.reset("webshop_goal_01000")
    rendered = json.dumps(prompt.build_messages(observation), ensure_ascii=False)
    assert "HIDDEN-TARGET-ASIN" not in rendered
    assert "goal_index" not in rendered

    rejected = environment.step(WebShopCommand("click[HIDDEN-TARGET-ASIN]"))
    assert not rejected.terminated
    assert rejected.info["policy_error"] == "click_target_not_public"
    assert server.step_commands == []


def test_public_search_then_listed_click_reaches_terminal_without_target_leakage():
    server = FakeWebShopServer()
    environment = _environment(server)
    environment.reset("webshop_goal_01000")
    searched = environment.step(WebShopCommand("search[blue mug]"))
    assert not searched.terminated
    assert searched.observation is not None
    assert "click[B000TARGET]" in searched.observation.available_actions
    terminal = environment.step(WebShopCommand("click[B000TARGET]"))
    assert terminal.terminated
    assert terminal.reward == 1.0
    assert terminal.info["task_score"] == 1.0
    assert "target_asin" not in terminal.info
    assert "HIDDEN-TARGET-ASIN" not in json.dumps(terminal.observation.to_dict())


def test_dense_terminal_score_is_preserved_but_rl_reward_is_binary():
    class PartialPurchaseServer(FakeWebShopServer):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/reset":
                return super().__call__(request)
            command = request.read().decode("utf-8")
            assert "search[blue mug]" in command
            return httpx.Response(
                200,
                json={
                    "observation": "Episode complete with a partial match.",
                    "env_state": {"page_type": "done"},
                    "reward": 0.625,
                    "done": True,
                    "info": {
                        "success": False,
                        "task_score": 0.625,
                        "selected_asin": "B000PARTIAL",
                        "available_actions": [],
                    },
                },
            )

    environment = _environment(PartialPurchaseServer())
    environment.reset("webshop_goal_01000")
    terminal = environment.step(WebShopCommand("search[blue mug]"))
    assert terminal.terminated
    assert terminal.reward == 0.0
    assert terminal.info["task_score"] == 0.625
    assert terminal.info["success"] is False


def test_public_anchor_ignores_episode_identity_but_binds_visible_action_list():
    left_server = FakeWebShopServer()
    right_server = FakeWebShopServer()
    left = _environment(left_server).reset("webshop_goal_01000")
    right = _environment(right_server).reset("webshop_goal_01000")
    assert left.episode_id != right.episode_id
    assert public_state_anchor_signature(left.to_dict()) == public_state_anchor_signature(right.to_dict())
    right.available_actions = (*right.available_actions, "click[new-public-action]")
    assert public_state_anchor_signature(left.to_dict()) != public_state_anchor_signature(right.to_dict())
    assert policy_context_signature([1, 2]) != policy_context_signature([1, 3])


def test_public_action_bound_preserves_terminal_controls():
    raw = ["search[<your query>]", *[f"click[option-{index}]" for index in range(300)], "click[Buy Now]"]
    bounded = _bounded_public_actions(raw)
    assert len(bounded) == MAX_PUBLIC_ACTIONS
    assert bounded[0] == "search[<your query>]"
    assert "click[Buy Now]" in bounded


def test_environment_fails_closed_on_split_mismatch():
    server = FakeWebShopServer()
    environment = _environment(server)
    with pytest.raises(ValueError, match="task/split"):
        environment.reset("webshop_goal_00500")


def test_environment_and_prompt_preserve_public_text_truncation_evidence():
    long_text = "A" * (prompt.MAX_OBSERVATION_CHARACTERS + 25)

    def server(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/reset"
        return httpx.Response(
            200,
            json={
                "observation": long_text,
                "env_state": {"page_type": "home"},
                "info": {
                    "instruction": "buy a blue mug",
                    "asin": "hidden",
                    "available_actions": ["search[<your query>]"],
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(server), base_url="http://webshop.test")
    observation = WebShopHTTPEnvironment(split="train", client=client).reset("webshop_goal_01000")
    assert len(observation.visible_text) == prompt.MAX_OBSERVATION_CHARACTERS
    assert observation.text_truncated
    rendered = prompt.build_messages(observation)[1]["content"]
    assert "observation_truncated: true" in rendered


def test_rejection_of_a_publicly_listed_action_is_infrastructure_invalid():
    class RejectingServer(FakeWebShopServer):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/reset":
                return super().__call__(request)
            return httpx.Response(
                200,
                json={
                    "observation": "Amazon Shopping Game",
                    "env_state": {"page_type": "home"},
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "error": "unexpected_server_rejection",
                        "available_actions": ["search[<your query>]"],
                    },
                },
            )

    server = RejectingServer()
    environment = _environment(server)
    environment.reset("webshop_goal_01000")
    with pytest.raises(ValueError, match="prior public state"):
        environment.step(WebShopCommand("search[blue mug]"))


def test_oracle_builds_only_public_executable_verified_turns():
    client = httpx.Client(transport=httpx.MockTransport(OracleFakeServer()), base_url="http://webshop.test")
    environment = WebShopHTTPEnvironment(split="train", client=client)
    goal = {
        "goal_index": 1000,
        "query": "blue mug",
        "name": "Blue Mug",
        "asin": "B000TARGET",
        "goal_options": ["navy"],
    }
    trajectory = build_verified_oracle_trajectory(environment, goal).to_dict()
    assert trajectory["verified_reward"] == 1.0
    assert [turn["command"] for turn in trajectory["turns"]] == [
        "search[Blue Mug]",
        "click[B000TARGET]",
        "click[navy]",
        "click[Buy Now]",
    ]
    assert all(set(json.loads(turn["completion"])) == {"command"} for turn in trajectory["turns"])
    assert "goal_options" not in json.dumps(trajectory["turns"])

    second_client = httpx.Client(transport=httpx.MockTransport(OracleFakeServer()), base_url="http://webshop.test")
    second_environment = WebShopHTTPEnvironment(split="train", client=second_client)
    second = build_verified_oracle_trajectory(second_environment, goal).to_dict()
    assert second_environment._episode_id != environment._episode_id
    assert second["content_sha256"] == trajectory["content_sha256"]


def test_oracle_title_query_is_bounded_sanitized_and_does_not_emit_target_asin():
    query = _oracle_title_query(
        {
            "name": "  Blue [B000TARGET]\nMug " + ("x" * 300),
            "asin": "B000TARGET",
        }
    )
    assert query.startswith("Blue Mug")
    assert len(query) == 200
    assert "[" not in query and "]" not in query
    assert "B000TARGET" not in query
    assert MAX_ORACLE_TURNS == 15


def test_oracle_refuses_the_frozen_test_split_before_reset():
    client = httpx.Client(transport=httpx.MockTransport(OracleFakeServer()), base_url="http://webshop.test")
    environment = WebShopHTTPEnvironment(split="test", client=client)
    with pytest.raises(OraclePolicyFailure, match="frozen_test"):
        build_verified_oracle_trajectory(
            environment,
            {
                "goal_index": 0,
                "query": "blue mug",
                "name": "Blue Mug",
                "asin": "B000TARGET",
                "goal_options": [],
            },
        )
