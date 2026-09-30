"""Regression tests for Stage25StripExecutorAgent per-turn diagnostics retention.

The agent used to ``copy.deepcopy`` the whole per-turn diagnostics tree on every
primitive turn even though only the final turn of each day survives in the
``days`` archive.  The retained-by-reference design must produce exactly the
same ``diagnostics_json()`` payload as the eager per-turn snapshot, including
the live counters that the controller keeps mutating.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from executor_v0.strip_executor import StripExecutorConfig  # noqa: E402
from rl_manager.executor_factory import Stage25StripExecutorAgent  # noqa: E402

from test_executor_v0_agent import (  # noqa: E402
    make_obs, recording_provider, simple_plan,
)

DAY = 6


def _agent(seat: int = 0) -> Stage25StripExecutorAgent:
    return Stage25StripExecutorAgent(
        provider=recording_provider(simple_plan()),
        seat=seat,
        strip_config=replace(StripExecutorConfig(), acting_seat=seat),
        profile={"name": "test"},
    )


def _drive(agent: Stage25StripExecutorAgent, days: tuple[int, ...],
           hours: int = 3) -> None:
    plan = simple_plan()
    for day in days:
        for hour in range(hours):
            agent(make_obs(day=day, hour=hour))


def test_retained_diagnostics_match_eager_per_turn_snapshot():
    """Deferred snapshot must equal the eager per-turn deepcopy archive."""

    agent = _agent()
    _drive(agent, (DAY, DAY + 1))
    deferred = agent.diagnostics_json()

    eager_agent = _agent()
    plan = simple_plan()
    eager_days: dict[str, dict] = {}
    for day in (DAY, DAY + 1):
        for hour in range(3):
            result = eager_agent.controller.act(
                make_obs(day=day, hour=hour), plan)
            eager_days[str(day)] = copy.deepcopy(result.diagnostics)
    eager_agent._days = eager_days

    assert sorted(deferred["days"]) == sorted(eager_agent._days)
    for key, value in eager_agent._days.items():
        assert deferred["days"][key] == value, f"day {key} diagnostics diverged"


def test_pending_day_is_flushed_exactly_once():
    agent = _agent()
    _drive(agent, (DAY,))
    first = agent.diagnostics_json()
    second = agent.diagnostics_json()
    assert first["days"] == second["days"]
    assert str(DAY) in first["days"]
    assert agent._pending_day is None


def test_retained_tree_is_detached_from_live_counters():
    """A retained day tree must not observe later counter mutations."""

    agent = _agent()
    _drive(agent, (DAY,))
    retained = agent.diagnostics_json()["days"][str(DAY)]
    live = agent.controller._claim_timings
    live["sentinel_injected"] = 1234.0
    assert "sentinel_injected" not in retained.get("row_claim_timing_ms", {})


def test_archive_is_not_shared_with_returned_payload():
    agent = _agent()
    _drive(agent, (DAY, DAY + 1))
    payload = agent.diagnostics_json()
    payload["days"][str(DAY)]["injected"] = True
    assert "injected" not in agent.diagnostics_json()["days"][str(DAY)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
