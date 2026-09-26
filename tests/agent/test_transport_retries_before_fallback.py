"""``agent.transport_retries_before_fallback``: how long a transport failure stays on the primary.

A transport failure (disconnect, timeout, overload) switched to ``fallback_providers`` after two
primary retries, a few seconds of backoff in all. A self-hosted primary that sheds load for a
minute or two (a hot GPU host refusing new connections until it cools) then hands every such turn
to the fallback chain, even though the primary is back moments later. The setting keeps the turn on
the primary, with the usual jittered exponential backoff, for as many retries as configured.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


class ServerDisconnected(Exception):
    """Stands in for httpx.RemoteProtocolError; the classifier matches it by message."""


def _agent():
    tool = {"type": "function", "function": {"name": "web_search", "description": "s",
                                             "parameters": {"type": "object", "properties": {}}}}
    with (
        patch("model_tools.get_tool_definitions", return_value=[tool]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12", base_url="http://gateway.local:8080/v1", provider="custom",
            model="local-model", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[{"provider": "custom", "model": "fallback-model",
                             "base_url": "http://fallback.local/v1"}],
        )
        agent.client = MagicMock()
        return agent


def _ok(text):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="m", usage=None)


def _run(agent, failures: int):
    calls = []

    def fake_api_call(api_kwargs):
        calls.append(agent.model)
        if len(calls) <= failures:
            raise ServerDisconnected("Server disconnected without sending a response.")
        return _ok(f"answered by {agent.model}")

    fb_client = MagicMock()
    fb_client.api_key, fb_client.base_url = "k", "http://fallback.local/v1"
    fb_client._custom_headers = fb_client.default_headers = None
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch("agent.retry_utils.jittered_backoff", return_value=0.0),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(fb_client, "fallback-model")),
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("hello")
    return result, calls


def test_default_still_falls_back_after_two_primary_retries():
    agent = _agent()
    agent._api_max_retries = 8
    result, calls = _run(agent, failures=2)
    assert calls == ["local-model", "local-model", "fallback-model"]
    assert result["final_response"] == "answered by fallback-model"


def test_configured_retries_ride_out_a_short_outage_on_the_primary():
    agent = _agent()
    agent._api_max_retries = 8
    agent._transport_retries_before_fallback = 6
    result, calls = _run(agent, failures=4)
    assert calls == ["local-model"] * 5
    assert result["final_response"] == "answered by local-model"
    assert agent._fallback_activated is False


@pytest.mark.parametrize("section,expected", [
    ({"transport_retries_before_fallback": 5}, 5), ({}, 2), ({"transport_retries_before_fallback": "x"}, 2),
])
def test_config_value_reaches_the_agent(section, expected):
    with patch("hermes_cli.config.load_config_readonly", return_value={"agent": section}):
        agent = _agent()
    assert agent._transport_retries_before_fallback == expected


@pytest.mark.parametrize("raw,expected", [(3, 3), (0, 0), (-1, 2), ("x", 2), (None, 2)])
def test_reader_is_defensive(raw, expected):
    from agent.turn_recovery import transport_retries_before_fallback

    assert transport_retries_before_fallback(SimpleNamespace(_transport_retries_before_fallback=raw)) == expected
