from __future__ import annotations

import pytest

from opening_book.agent import make_opening_agent
from opening_book.trace import (
    TraceError,
    action_for,
    load_built_in_trace,
    trace_handoff_phase,
)


def _obs(day: int, hour: int, hand_count: int, seat: int = 0) -> dict:
    farms = [
        {
            "money": 3000.0,
            "hands": [],
            "tiles": [[None] * 10 for _ in range(10)],
            "unlocked_quadrants": ["NW"],
        },
        {
            "money": 3000.0,
            "hands": [],
            "tiles": [[None] * 10 for _ in range(10)],
            "unlocked_quadrants": ["NW"],
        },
    ]
    farms[seat]["hands"] = [[0, 0] for _ in range(hand_count)]
    return {
        "day": day,
        "hour": hour,
        "farms": farms,
        "private": {"shed": {}},
    }


def test_legacy_trace_keeps_d4h0_handoff() -> None:
    trace = load_built_in_trace("tetsuya_s1")
    assert len(trace["turns"]) == 96
    assert trace_handoff_phase(trace) == (4, 0)


def test_dsm_trace_is_148_turn_partial_day_opening() -> None:
    trace = load_built_in_trace("dsm_d0_d6h3")
    assert len(trace["turns"]) == 148
    assert trace_handoff_phase(trace) == (6, 4)
    final = action_for(trace, 6, 3)
    assert ["BUY_LAND"] in final["market"]
    with pytest.raises(TraceError, match="handoff is \\(6,4\\)"):
        action_for(trace, 6, 4)


def test_dsm_agent_replays_full_prefix_then_delegates_at_d6h4() -> None:
    delegated = []

    def downstream(obs):
        delegated.append((obs["day"], obs["hour"]))
        return {"farmer": ["PASS"], "hands": [], "market": []}

    trace = load_built_in_trace("dsm_d0_d6h3")
    agent = make_opening_agent("dsm_d0_d6h3", downstream=downstream, seat=0)

    for turn in trace["turns"]:
        action = turn["action"]
        emitted = agent(_obs(
            turn["day"], turn["hour"], len(action["hands"]), seat=0))
        assert emitted == action

    assert delegated == []
    emitted = agent(_obs(6, 4, 0, seat=0))
    assert emitted == {"farmer": ["PASS"], "hands": [], "market": []}
    assert delegated == [(6, 4)]

    diag = agent.diagnostics_json()
    assert diag["turns_replayed"] == 148
    assert diag["handoff"]["clean_handoff"] is True
    assert diag["handoff"]["clean_d4h0_handoff"] is False
    assert diag["handoff"]["turn"] == [6, 4]
    assert diag["handoff"]["expected_turn"] == [6, 4]
