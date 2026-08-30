from __future__ import annotations

from scripts.audit_fast_official_parity import (
    _TraceAgent,
    _divergence_payload,
    _pass_action,
)
from oracle.replay import DivergenceError, DivergenceReport


def test_pass_action_tracks_current_hand_count():
    observation = {"farms": [{"hands": [[], []]}, {"hands": []}]}
    assert _pass_action(observation, 0) == {
        "farmer": ["PASS"],
        "hands": [["PASS"], ["PASS"]],
        "market": [],
    }


def test_trace_agent_returns_exact_saved_actions_without_aliasing():
    actions = [[
        {"farmer": ["NORTH"], "hands": [], "market": []},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]]
    agent = _TraceAgent(actions, 0)
    result = agent({})
    result["farmer"][0] = "SOUTH"
    assert actions[0][0]["farmer"] == ["NORTH"]


def test_divergence_payload_records_same_action_hierarchy():
    report = DivergenceReport(
        seed=17,
        step=144,
        day=6,
        hour=0,
        field_path="state.farms[0].tiles[37].consecutive_unwatered",
        official_value=0,
        fast_value=1,
        p0_action={"farmer": ["WATER"]},
        p1_action={"farmer": ["PASS"]},
        phase="turn",
        turn_index=143,
    )
    payload = _divergence_payload(DivergenceError(report))
    assert payload["pre_state_equal"] is True
    assert payload["pre_policy_observations_equal"] is True
    assert payload["joint_action"] == [report.p0_action, report.p1_action]
