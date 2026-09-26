"""/goal re-arm: an active goal left idle (a restart killed its turn, or a turn ended with no reply)
is kicked again by the gateway instead of waiting forever for a human message.

Before this, a goal only advanced from the post-turn hook of a completed turn. A pod restart killed
the in-flight turn and nothing re-entered the session, so the goal sat ``active`` with no worker.
"""

import asyncio
import contextlib
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from hermes_cli import goals
from hermes_cli.goals import GoalManager
from hermes_state import SessionDB


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=home / "state.db")
    monkeypatch.setattr(goals, "_DB_CACHE", {str(home): db})
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {"goals": {"rearm_idle_seconds": 600}})
    config = GatewayConfig()
    store = SessionStore(home / "sessions", config)
    adapter = SimpleNamespace(_message_handler=object(), _active_sessions={}, handle_message=AsyncMock())
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner._running = True
    runner._run_in_executor_with_context = asyncio.to_thread
    runner._restored_source = lambda entry: entry.origin
    runner._profile_scope_for_source = lambda source: contextlib.nullcontext()
    runner._delivery_adapter_for = lambda source: adapter
    busy: set = set()
    runner._is_session_running = lambda key: key in busy
    runner._queue_depth = lambda key, adapter=None: 0
    try:
        yield SimpleNamespace(runner=runner, store=store, adapter=adapter, busy=busy)
    finally:
        store.close_all_db_handles()
        db.close()


def _session_with_goal(store, thread: str, *, status: str = "active", idle_for: float = 3600.0):
    source = SessionSource(platform=Platform.DISCORD, chat_id="chan", chat_type="thread",
                           thread_id=thread, user_id="owner")
    entry = store.get_or_create_session(source)
    mgr = GoalManager(entry.session_id)
    mgr.set("open a PR for issue 746")
    mgr.state.created_at = mgr.state.last_turn_at = time.time() - idle_for
    mgr._save()
    if status == "paused":
        mgr.pause("operator")
    return entry


@pytest.mark.asyncio
async def test_an_idle_active_goal_is_kicked_once_and_charged(env):
    entry = _session_with_goal(env.store, "1")
    fired = await env.runner._goal_rearm_scan(600)
    assert fired == 1
    env.adapter.handle_message.assert_awaited_once()
    event = env.adapter.handle_message.await_args.args[0]
    assert "open a PR for issue 746" in event.text
    assert event.metadata["gateway_session_key"] == entry.session_key
    assert GoalManager(entry.session_id).state.turns_used == 1
    # The kick refreshed last_turn_at, so the next scan leaves it alone.
    assert await env.runner._goal_rearm_scan(600) == 0


@pytest.mark.asyncio
async def test_recent_busy_and_paused_goals_are_left_alone(env):
    _session_with_goal(env.store, "fresh", idle_for=10)
    _session_with_goal(env.store, "paused", status="paused")
    busy = _session_with_goal(env.store, "busy")
    env.busy.add(busy.session_key)
    assert await env.runner._goal_rearm_scan(600) == 0
    env.adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_watcher_is_off_by_default(env, monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    _session_with_goal(env.store, "1")
    await asyncio.wait_for(env.runner._goal_rearm_watcher(interval=0.01), timeout=2)
    env.adapter.handle_message.assert_not_awaited()


def test_watcher_is_spawned_with_the_other_post_reconnect_watchers():
    assert "_goal_rearm_watcher" in GatewayRunner._POST_RECONNECT_WATCHERS
    assert callable(getattr(GatewayRunner, "_goal_rearm_watcher"))
