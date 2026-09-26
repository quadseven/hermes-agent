"""DISCORD_ALLOWED_BOTS: name the bots and webhooks that may instruct the agent.

DISCORD_ALLOW_BOTS=mentions admits every bot that mentions the agent, and an admitted bot skips the
human allowlist. A supervisor bot needs that door open, but any other bot or webhook in the server
then walks through it too. The allowlist narrows the door to named IDs at both gates (adapter
admission and gateway authorization); DISCORD_BOT_ALLOWLIST_REQUIRED keeps it shut while the list
is still empty.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import discord
import pytest

from gateway.bot_allowlist import bot_author_admitted
from gateway.platforms.helpers import MessageDeduplicator
from gateway.session import Platform, SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter

SUPERVISOR, STRANGER = "1500000000000000001", "1500000000000000002"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("DISCORD_ALLOW_BOTS", "DISCORD_ALLOWED_USERS", "DISCORD_ALLOWED_BOTS",
                "DISCORD_BOT_ALLOWLIST_REQUIRED", "DISCORD_ALLOWED_ROLES", "DISCORD_ALLOW_ALL_USERS",
                "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "DISCORD_BOTS_REQUIRE_INLINE_MENTION"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("allowed,required,author,expected", [
    ("", "", STRANGER, True),                    # no list: upstream behaviour
    (SUPERVISOR, "", SUPERVISOR, True),
    (SUPERVISOR, "", STRANGER, False),
    (f"{SUPERVISOR}, 7", "", "7", True),
    ([SUPERVISOR], "", SUPERVISOR, True),
    ("", "true", SUPERVISOR, False),             # required and empty: fail closed
    (SUPERVISOR, "true", SUPERVISOR, True),
    (SUPERVISOR, "", "", False),
])
def test_bot_author_admitted(allowed, required, author, expected):
    assert bot_author_admitted(allowed, required, author) is expected


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _bot_source(bot_id):
    return SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="thread",
                         user_id=bot_id, user_name="operator-assistant", is_bot=True)


def test_gateway_admits_only_the_listed_bot(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")
    monkeypatch.setenv("DISCORD_ALLOWED_BOTS", SUPERVISOR)
    runner = _runner()
    assert runner._is_user_authorized(_bot_source(SUPERVISOR)) is True
    assert runner._is_user_authorized(_bot_source(STRANGER)) is False


def test_gateway_fails_closed_when_required_and_empty(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_BOT_ALLOWLIST_REQUIRED", "true")
    assert _runner()._is_user_authorized(_bot_source(SUPERVISOR)) is False


def test_humans_are_unaffected(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")
    monkeypatch.setenv("DISCORD_ALLOWED_BOTS", SUPERVISOR)
    human = SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="thread",
                          user_id="100200300", is_bot=False)
    assert _runner()._is_user_authorized(human) is True


def _adapter(extra):
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(extra=extra)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._get_allow_bots = Mock(return_value="mentions")
    adapter._is_allowed_user = Mock(return_value=True)
    adapter._text_batch_delay_seconds = 0.6
    adapter._text_batch_split_delay_seconds = 2.0
    adapter._bot_tag_debounce_until = {}
    return adapter


def _bot_message(adapter, author_id):
    return SimpleNamespace(id=int(author_id[-3:]) + 1000, author=SimpleNamespace(id=int(author_id), bot=True),
                           channel=SimpleNamespace(id=7), content="<@99> /goal status",
                           mentions=[adapter._client.user], type=discord.MessageType.default)


def test_adapter_admission_checks_the_list():
    adapter = _adapter({"allowed_bots": [SUPERVISOR]})
    assert adapter._discord_message_admission(_bot_message(adapter, SUPERVISOR), claim=False)[0] is True
    assert adapter._discord_message_admission(_bot_message(adapter, STRANGER), claim=False)[0] is False


def test_adapter_admission_fails_closed_when_required_and_empty():
    adapter = _adapter({"bot_allowlist_required": "true"})
    assert adapter._discord_message_admission(_bot_message(adapter, SUPERVISOR), claim=False)[0] is False


def test_adapter_without_a_list_keeps_upstream_behaviour():
    adapter = _adapter({})
    assert adapter._discord_message_admission(_bot_message(adapter, STRANGER), claim=False)[0] is True


def test_yaml_keys_are_seeded_into_extra(monkeypatch):
    import importlib
    import sys

    monkeypatch.delenv("DISCORD_ALLOWED_BOTS", raising=False)
    monkeypatch.delenv("DISCORD_BOT_ALLOWLIST_REQUIRED", raising=False)
    sys.modules.pop("plugins.platforms.discord.adapter", None)
    mod = importlib.import_module("plugins.platforms.discord.adapter")
    seeded = mod._apply_yaml_config({}, {"allowed_bots": [SUPERVISOR, "7"], "bot_allowlist_required": True})
    assert seeded["allowed_bots"] == f"{SUPERVISOR},7"
    assert seeded["bot_allowlist_required"] == "true"
