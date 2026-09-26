"""``goals.judge: gates`` and ``goals.default_gates``: a deterministic judge that is not the worker.

With the LLM judge, the auxiliary model decides DONE after the gates pass, and unconfigured it is the
same model that did the work. Under ``judge: gates`` the gates are the whole verdict. Default gates
attach to every goal from config.yaml, so no gateway admin has to add them per goal, and each gate is
told which goal it guards through HERMES_GOAL_* environment variables.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from hermes_cli import goals
from hermes_cli.goals import GoalGate, GoalManager

_PY = sys.executable


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _config(monkeypatch, goals_cfg):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {"goals": goals_cfg})


def _judge_must_not_run(*_a, **_kw):
    raise AssertionError("the LLM judge ran under goals.judge: gates")


def _passing() -> str:
    return f'"{_PY}" -c "raise SystemExit(0)"'


def _failing(message: str = "no PR yet") -> str:
    return f'"{_PY}" -c "print(\'{message}\'); raise SystemExit(1)"'


class TestGatesOnlyJudge:
    def test_all_gates_pass_is_done_without_any_model_call(self, hermes_home, monkeypatch):
        _config(monkeypatch, {"judge": "gates"})
        mgr = GoalManager("s-pass")
        mgr.set("ship the fix")
        mgr.add_gate(_passing())
        with patch.object(goals, "judge_goal", side_effect=_judge_must_not_run):
            decision = mgr.evaluate_after_turn("Done! LOOP_COMPLETE")
        assert decision["status"] == "done"
        assert decision["verdict"] == "done"
        assert decision["should_continue"] is False
        assert GoalManager("s-pass").state.status == "done"

    def test_a_failing_gate_keeps_the_goal_open_with_its_output(self, hermes_home, monkeypatch):
        _config(monkeypatch, {"judge": "gates"})
        mgr = GoalManager("s-fail")
        mgr.set("ship the fix")
        mgr.add_gate(_failing("no PR yet"), max_retries=50)
        with patch.object(goals, "judge_goal", side_effect=_judge_must_not_run):
            decision = mgr.evaluate_after_turn("I am done, LOOP_COMPLETE")
        assert decision["status"] == "active"
        assert decision["should_continue"] is True
        assert "no PR yet" in decision["continuation_prompt"]
        assert GoalManager("s-fail").state.status == "active"

    def test_a_claim_of_done_with_no_gates_never_completes(self, hermes_home, monkeypatch):
        _config(monkeypatch, {"judge": "gates"})
        mgr = GoalManager("s-none")
        mgr.set("ship the fix", max_turns=2)
        with patch.object(goals, "judge_goal", side_effect=_judge_must_not_run):
            first = mgr.evaluate_after_turn("Goal complete.")
            second = mgr.evaluate_after_turn("Goal complete, really.")
        assert first["status"] == "active" and first["should_continue"] is True
        assert "no gates" in first["reason"]
        assert second["status"] == "paused"
        assert GoalManager("s-none").state.status == "paused"

    def test_default_mode_still_asks_the_llm_judge(self, hermes_home, monkeypatch):
        _config(monkeypatch, {})
        mgr = GoalManager("s-llm")
        mgr.set("ship the fix")
        mgr.add_gate(_passing())
        with patch.object(goals, "judge_goal", return_value=("continue", "not yet", False, None, False)) as judge:
            decision = mgr.evaluate_after_turn("working")
        judge.assert_called_once()
        assert decision["status"] == "active"

    def test_unknown_mode_falls_back_to_llm(self, monkeypatch):
        _config(monkeypatch, {"judge": "vibes"})
        assert goals.goal_judge_mode() == goals.JUDGE_MODE_LLM


class TestDefaultGates:
    def test_config_gates_attach_to_every_new_goal(self, hermes_home, monkeypatch):
        _config(monkeypatch, {"default_gates": [
            "check-a",
            {"command": "check-b", "timeout_seconds": 90, "max_retries": 40},
            {"command": "   "},
            42,
        ]})
        state = GoalManager("s-default").set("ship it")
        assert [g.command for g in state.gates] == ["check-a", "check-b"]
        assert state.gates[1].timeout_seconds == 90
        assert state.gates[1].max_retries == 40
        reloaded = GoalManager("s-default").state
        assert [g.command for g in reloaded.gates] == ["check-a", "check-b"]

    def test_no_default_gates_leaves_goals_ungated(self, hermes_home, monkeypatch):
        _config(monkeypatch, {})
        assert GoalManager("s-plain").set("ship it").gates == []

    def test_the_gate_is_told_which_goal_it_guards(self, hermes_home, monkeypatch):
        _config(monkeypatch, {"judge": "gates"})
        mgr = GoalManager("sess-env")
        state = mgr.set("ship the env fix")
        check = (
            "import os, sys; "
            "ok = os.environ['HERMES_GOAL_SESSION_ID'] == 'sess-env' "
            f"and os.environ['HERMES_GOAL_CREATED_AT'] == '{int(state.created_at)}' "
            "and os.environ['HERMES_GOAL_TEXT'] == 'ship the env fix'; "
            "sys.exit(0 if ok else 1)"
        )
        mgr.add_gate(f'"{_PY}" -c "{check}"')
        decision = mgr.evaluate_after_turn("done")
        assert decision["status"] == "done", decision


class TestRearmCharge:
    def test_rearm_charges_one_turn_and_returns_the_continuation(self, hermes_home, monkeypatch):
        _config(monkeypatch, {})
        mgr = GoalManager("s-rearm")
        mgr.set("ship it", max_turns=2)
        prompt = mgr.note_rearm()
        assert prompt and "ship it" in prompt
        assert GoalManager("s-rearm").state.turns_used == 1

    def test_rearm_past_the_budget_pauses_instead_of_kicking(self, hermes_home, monkeypatch):
        _config(monkeypatch, {})
        mgr = GoalManager("s-budget")
        mgr.set("ship it", max_turns=1)
        assert mgr.note_rearm() is not None
        assert mgr.note_rearm() is None
        assert GoalManager("s-budget").state.status == "paused"

    def test_rearm_ignores_paused_goals(self, hermes_home, monkeypatch):
        _config(monkeypatch, {})
        mgr = GoalManager("s-paused")
        mgr.set("ship it")
        mgr.pause("operator")
        assert mgr.note_rearm() is None
        assert GoalManager("s-paused").state.turns_used == 0


def test_rearm_idle_seconds_reader(monkeypatch):
    _config(monkeypatch, {"rearm_idle_seconds": 900})
    assert goals.goal_rearm_idle_seconds() == 900.0
    _config(monkeypatch, {"rearm_idle_seconds": "soon"})
    assert goals.goal_rearm_idle_seconds() == 0.0
    _config(monkeypatch, {})
    assert goals.goal_rearm_idle_seconds() == 0.0


def test_gate_retries_default_is_unchanged():
    assert GoalGate(command="x").max_retries == goals.DEFAULT_GATE_MAX_RETRIES
