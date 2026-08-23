"""Focused offline tests for the opening_book runtime wrapper (issue #4, stage 2).

No raw sample data, no engine, no network. Uses a recording fake downstream
and minimal synthetic observations shaped like 1.32.7 observations.
"""

from __future__ import annotations

import copy
import json

import pytest

from opening_book import DEFAULT_IDENTITY, TraceError, load_built_in_trace
from opening_book.agent import (
    DIVERGENCE_HAND_COUNT,
    DIVERGENCE_MALFORMED_PHASE,
    DIVERGENCE_PHASE_MISMATCH,
    OpeningAgent,
    make_opening_agent,
)

HORIZON = [(d, h) for d in range(4) for h in range(24)]


class RecordingDownstream:
    """Fake downstream: records obs objects, returns a canned action."""

    def __init__(self, result=None):
        self.calls: list = []
        self.result = result if result is not None else {
            "farmer": ["PASS"], "hands": [], "market": []}

    def __call__(self, obs):
        self.calls.append(obs)
        return self.result


def make_obs(trace, day, hour, seat=0, *, hands_override=None, farms=None,
             private=None, extra=None):
    """Minimal observation whose hand count matches the trace for (day,hour)."""
    if farms is None:
        hands = hands_override
        if hands is None:
            idx = day * 24 + hour
            n = len(trace["turns"][idx]["action"]["hands"]) \
                if 0 <= idx < len(trace["turns"]) else 0
            hands = [None] * n
        farms = [{}, {}]
        farms[seat] = {
            "money": 100.0 + day,
            "tiles": [],
            "unlocked_quadrants": ["NE"],
            "hands": hands,
        }
    obs = {"day": day, "hour": hour, "farms": farms, "player": seat}
    if private is not None:
        obs["private"] = private
    if extra:
        obs.update(extra)
    return obs


@pytest.fixture()
def trace():
    return load_built_in_trace("standard_mixed")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_make_opening_agent_rejects_invalid_inputs():
    downstream = RecordingDownstream()
    with pytest.raises(ValueError, match="downstream"):
        make_opening_agent(opening="standard_mixed", seat=0)
    with pytest.raises(ValueError, match="downstream"):
        make_opening_agent(opening="standard_mixed", downstream=None, seat=0)
    with pytest.raises(ValueError, match="seat"):
        make_opening_agent(downstream=downstream, seat=None)
    with pytest.raises(ValueError, match="seat"):
        make_opening_agent(downstream=downstream, seat=2)
    with pytest.raises(TypeError, match="callable"):
        OpeningAgent(load_built_in_trace(), "not callable", 0)
    with pytest.raises(TraceError, match="unknown opening identity"):
        make_opening_agent(opening="wheat_only", downstream=downstream, seat=0)


# ---------------------------------------------------------------------------
# Normal sequence: exact 96 scripted turns then d4h0 handoff
# ---------------------------------------------------------------------------

def test_full_96_turn_playback_then_exact_d4h0_handoff(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(opening="standard_mixed",
                               downstream=downstream, seat=0)
    for day, hour in HORIZON:
        obs = make_obs(trace, day, hour, seat=0)
        action = agent(obs)
        assert action == trace["turns"][day * 24 + hour]["action"]
    assert downstream.calls == []
    assert agent._turns_replayed == 96

    handoff_obs = make_obs(trace, 4, 0, seat=0)
    returned = agent(handoff_obs)
    assert returned is downstream.result
    assert downstream.calls == [handoff_obs]  # exact original object
    diag = agent.diagnostics_json()
    assert diag["turns_replayed"] == 96
    assert diag["divergence"]["occurred"] is False
    assert diag["fallback_active"] is False
    assert diag["handoff"]["turn"] == [4, 0]
    assert diag["handoff"]["clean_d4h0_handoff"] is True


def test_d3h23_boundary_and_all_later_calls_delegated(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    for day, hour in HORIZON[:-1]:
        agent(make_obs(trace, day, hour))
    last = make_obs(trace, 3, 23)
    assert agent(last) == trace["turns"][95]["action"]
    assert downstream.calls == []
    for turn in [(4, 0), (4, 1), (5, 0), (29, 23)]:
        obs = make_obs(trace, *turn)
        agent(obs)
    assert [list((o["day"], o["hour"])) for o in downstream.calls] == \
        [[4, 0], [4, 1], [5, 0], [29, 23]]


# ---------------------------------------------------------------------------
# Seats and provenance
# ---------------------------------------------------------------------------

def test_both_runtime_seats_same_identity_provenance_separate(trace):
    for seat in (0, 1):
        downstream = RecordingDownstream()
        agent = make_opening_agent(opening="pasture_heavy",
                                   downstream=downstream, seat=seat)
        ph = load_built_in_trace("pasture_heavy")
        for day, hour in HORIZON[:3]:
            obs = make_obs(ph, day, hour, seat=seat)
            assert agent(obs) == ph["turns"][day * 24 + hour]["action"]
        agent(make_obs(ph, 4, 0, seat=seat))
        diag = agent.diagnostics_json()
        assert diag["seat"] == seat
        assert diag["opening"] == "pasture_heavy"
        prov = diag["source_provenance"]
        assert prov["source_seat"] == 0  # source seat recorded separately
        assert prov["episode"] == 95055022
        assert prov["player"] == "ReCurSiON"
        assert len(prov["source_replay_sha256"]) == 64


def test_seat_selects_observation_state_not_trace_data(trace):
    downstream = RecordingDownstream()
    before = json.dumps(trace, sort_keys=True)
    agent = make_opening_agent(downstream=downstream, seat=1)
    # seat=1 reads farms[1]; give seat-1 the wrong hand count -> divergence
    obs1_bad = make_obs(trace, 0, 0, seat=1, hands_override=[None] * 99)
    agent(obs1_bad)
    assert downstream.calls == [obs1_bad]
    assert agent.diagnostics_json()["divergence"]["reason"].startswith(
        DIVERGENCE_HAND_COUNT)
    # trace data untouched by any wrapper activity
    assert json.dumps(load_built_in_trace("standard_mixed"), sort_keys=True) \
        == before


# ---------------------------------------------------------------------------
# Divergence guards: one-way permanent fallback
# ---------------------------------------------------------------------------

def _assert_permanent_fallback(agent, downstream, first_bad_obs):
    assert downstream.calls[-1] is first_bad_obs
    diag = agent.diagnostics_json()
    assert diag["divergence"]["occurred"] is True
    assert diag["fallback_active"] is True
    # even a perfectly-phased later call must stay delegated
    good = make_obs(load_built_in_trace("standard_mixed"), 0, 1)
    n = len(downstream.calls)
    agent(good)
    assert len(downstream.calls) == n + 1 and downstream.calls[-1] is good


def test_skipped_hour_diverges(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    agent(make_obs(trace, 0, 0))
    bad = make_obs(trace, 0, 2)  # skipped h1
    agent(bad)
    diag = agent.diagnostics_json()
    assert diag["divergence"]["occurred"] is True
    assert diag["divergence"]["observed_turn"] == [0, 2]
    assert DIVERGENCE_PHASE_MISMATCH in diag["divergence"]["reason"]
    _assert_permanent_fallback(agent, downstream, bad)


def test_repeated_and_out_of_order_calls_diverge(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    agent(make_obs(trace, 0, 0))
    repeated = make_obs(trace, 0, 0)
    agent(repeated)
    assert DIVERGENCE_PHASE_MISMATCH in \
        agent.diagnostics_json()["divergence"]["reason"]
    _assert_permanent_fallback(agent, downstream, repeated)


def test_malformed_phase_diverges():
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    bad = {"farms": [{}, {}]}  # no day/hour
    agent(bad)
    assert DIVERGENCE_MALFORMED_PHASE in \
        agent.diagnostics_json()["divergence"]["reason"]
    _assert_permanent_fallback(
        agent, downstream, bad)


def test_first_call_mid_opening_diverges_with_zero_scripted(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    mid = make_obs(trace, 2, 5)
    agent(mid)
    diag = agent.diagnostics_json()
    assert diag["turns_replayed"] == 0
    assert diag["divergence"]["occurred"] is True
    assert diag["divergence"]["observed_turn"] == [2, 5]
    _assert_permanent_fallback(agent, downstream, mid)


def test_first_call_d4h0_or_later_is_clean_zero_turn_handoff(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    obs = make_obs(trace, 4, 0)
    agent(obs)
    diag = agent.diagnostics_json()
    assert diag["turns_replayed"] == 0
    assert diag["divergence"]["occurred"] is False
    assert diag["handoff"]["turn"] == [4, 0]
    assert downstream.calls == [obs]

    late = make_opening_agent(downstream=RecordingDownstream(), seat=0)
    late(make_obs(trace, 7, 12))
    assert late.diagnostics_json()["handoff"]["turn"] == [7, 12]


def test_hand_cardinality_mismatch_diverges(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    bad = make_obs(trace, 0, 0, hands_override=[None] * 3)
    agent(bad)
    diag = agent.diagnostics_json()
    assert DIVERGENCE_HAND_COUNT in diag["divergence"]["reason"]
    assert diag["turns_replayed"] == 0
    _assert_permanent_fallback(agent, downstream, bad)


def test_missing_farm_for_seat_diverges(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=1)
    bad = {"day": 0, "hour": 0, "farms": [{"hands": []}]}  # farms[1] missing
    agent(bad)
    assert "observation_incomplete" in \
        agent.diagnostics_json()["divergence"]["reason"]
    _assert_permanent_fallback(agent, downstream, bad)


def test_market_cap_validation_reused_via_doctored_trace(trace):
    doctored = copy.deepcopy(trace)
    doctored["turns"][0]["action"]["market"] = [["HIRE"]] * 11
    downstream = RecordingDownstream()
    agent = OpeningAgent(doctored, downstream, 0)
    bad = make_obs(doctored, 0, 0)
    agent(bad)
    diag = agent.diagnostics_json()
    assert "action_invalid" in diag["divergence"]["reason"]
    assert "max is 10" in diag["divergence"]["reason"]
    _assert_permanent_fallback(agent, downstream, bad)


# ---------------------------------------------------------------------------
# Defensive copies / no mutation leakage
# ---------------------------------------------------------------------------

def test_returned_actions_are_defensive_copies(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    a1 = agent(make_obs(trace, 0, 0))
    frozen = copy.deepcopy(a1)
    a1["market"].append(["HACK"])
    a1["farmer"][0] = "HACKED"
    a2 = agent(make_obs(trace, 0, 1))
    assert a2 == trace["turns"][1]["action"]
    assert a2 != a1
    # committed data unchanged
    fresh = load_built_in_trace("standard_mixed")
    assert fresh["turns"][0]["action"] == frozen


# ---------------------------------------------------------------------------
# Farm summary extraction
# ---------------------------------------------------------------------------

def representative_tiles():
    return [
        [{"kind": "PLANT", "crop": "WHEAT"}, None,
         {"kind": "PLANT", "crop": "MELON"}],
        [{"animal": "COW"}, {"animal": "COW"}, {"animal": "SHEEP"}],
    ]


def test_farm_summary_full_observation(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    farms = [{}, {}]
    farms[0] = {
        "money": 42.5,
        "tiles": representative_tiles(),
        "unlocked_quadrants": ["NE"],
        "hands": [],
    }
    obs = make_obs(trace, 4, 0, farms=farms,
                   private={"shed": {"WHEAT": 6}})
    agent(obs)
    summary = agent.diagnostics_json()["handoff"]["farm_summary"]
    assert summary["money"] == 42.5
    assert summary["crops"] == {"WHEAT": 1, "MELON": 1}
    assert summary["animals"] == {"COW": 2, "SHEEP": 1}
    assert summary["land_count"] == 1
    assert summary["shed_wheat"] == 6


def test_farm_summary_missing_fields_best_effort(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=0)
    obs = {"day": 4, "hour": 0, "farms": [{}, {}]}
    agent(obs)
    summary = agent.diagnostics_json()["handoff"]["farm_summary"]
    assert summary == {
        "money": None, "crops": {}, "animals": {},
        "land_count": None, "shed_wheat": None,
    }


def test_farm_summary_uses_configured_seat(trace):
    downstream = RecordingDownstream()
    agent = make_opening_agent(downstream=downstream, seat=1)
    farms = [
        {"money": 1.0, "tiles": [{"kind": "PLANT", "crop": "WHEAT"}],
         "unlocked_quadrants": ["NE"], "hands": []},
        {"money": 2.0, "tiles": [[{"animal": "COW"}]],
         "unlocked_quadrants": ["NE", "SW"], "hands": []},
    ]
    obs = make_obs(trace, 4, 0, farms=farms)
    agent(obs)
    summary = agent.diagnostics_json()["handoff"]["farm_summary"]
    assert summary["money"] == 2.0
    assert summary["animals"] == {"COW": 1}
    assert summary["crops"] == {}
    assert summary["land_count"] == 2


# ---------------------------------------------------------------------------
# Deterministic JSON diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_deterministic_and_serializable(trace):
    def run_game():
        downstream = RecordingDownstream()
        agent = make_opening_agent(downstream=downstream, seat=0)
        for day, hour in HORIZON[:5]:
            agent(make_obs(trace, day, hour))
        agent(make_obs(trace, 0, 9))  # divergence
        return agent

    a, b = run_game(), run_game()
    ja = json.dumps(a.diagnostics_json(), sort_keys=True)
    jb = json.dumps(b.diagnostics_json(), sort_keys=True)
    assert ja == jb
    parsed = json.loads(ja)
    assert parsed["opening"] == DEFAULT_IDENTITY
    assert parsed["turns_replayed"] == 5
    assert parsed["delegated_calls"] == 1
