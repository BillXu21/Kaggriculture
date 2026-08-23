"""Focused oracle tests against the pinned official 1.32.7 engine.

Skipped unless ``kaggle_environments`` is installed AND passes the provenance
guard (exact version + interpreter file hashes at the pinned upstream commit).
"""

from __future__ import annotations

import pytest

from oracle import DivergenceError, run_same_action_replay
from oracle.provenance import ProvenanceError, verify_official_provenance

try:
    verify_official_provenance()
    OFFICIAL_AVAILABLE = True
    _SKIP_REASON = ""
except ProvenanceError as error:
    OFFICIAL_AVAILABLE = False
    _SKIP_REASON = str(error)

pytestmark = pytest.mark.skipif(not OFFICIAL_AVAILABLE, reason=_SKIP_REASON)


def pass_action() -> dict[str, object]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def act(**overrides) -> dict[str, object]:
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    action.update(overrides)
    return action


def test_initial_full_state_parity() -> None:
    run_same_action_replay({"seed": 7}, [])


def test_short_legal_trace_no_divergence() -> None:
    trace = [
        [act(market=[["BUY_SEED", "WHEAT", 2]]), pass_action()],
        [act(farmer=["PLANT", "WHEAT"]), pass_action()],
        [act(farmer=["WATER"]), pass_action()],
        [pass_action(), pass_action()],
    ]
    result = run_same_action_replay({"seed": 7}, trace)
    assert result.turns_executed == 4
    assert result.official_statuses == result.fast_statuses == ["ACTIVE", "ACTIVE"]


def test_both_seat_observation_privacy_comparison() -> None:
    from oracle.backend import make_backend
    from oracle.canonical import canonical_state_official

    official = make_backend("official", {"seed": 7})
    fast = make_backend("fast", {"seed": 7})
    pair = [act(market=[["BUY_SEED", "WHEAT", 2]]), pass_action()]
    official.reset(), fast.reset()
    official.step(pair), fast.step(pair)

    official_canonical = official.canonical_state()
    # Each seat's private block must reflect only its own shed/seeds.
    assert official_canonical["privates"][0]["seeds"]["WHEAT"] == 2
    assert official_canonical["privates"][1]["seeds"]["WHEAT"] == 0
    # Public blocks are identical to both seats on both engines; only the
    # private blocks differ per seat.
    assert official.env.state[0].observation.farms == official.env.state[1].observation.farms
    assert fast.observations()[0]["farms"] == fast.observations()[1]["farms"]

    # A privacy leak (seat 1 seeing seat 0's seeds) must be caught at an exact path.
    def leak(canonical: dict, turn: int) -> None:
        if turn == 0:
            canonical["privates"][1]["seeds"]["WHEAT"] = 2

    with pytest.raises(DivergenceError) as excinfo:
        run_same_action_replay(
            {"seed": 7}, [pair], fast_mutator=leak,
        )
    assert excinfo.value.report.field_path == "state.privates[1].seeds.WHEAT"


def test_deliberate_corruption_stops_at_first_divergent_turn_and_path() -> None:
    trace = [
        [pass_action(), pass_action()],
        [pass_action(), pass_action()],
        [act(market=[["BUY_SEED", "CARROT", 1]]), pass_action()],
        [pass_action(), pass_action()],
    ]

    def corrupt(canonical: dict, turn: int) -> None:
        if turn == 2:
            canonical["farms"][1]["hires_today"] = 99

    with pytest.raises(DivergenceError) as excinfo:
        run_same_action_replay({"seed": 7}, trace, fast_mutator=corrupt)
    report = excinfo.value.report
    assert report.turn_index == 2
    assert report.step == 3 and report.day == 0 and report.hour == 3
    assert report.field_path == "state.farms[1].hires_today"
    assert report.official_value == 0 and report.fast_value == 99
    assert report.seed == 7
    assert report.p0_action == trace[2][0] and report.p1_action == trace[2][1]


def test_terminal_rewards_and_statuses_match() -> None:
    configuration = {"seed": 7, "episodeSteps": 3}
    trace = [[pass_action(), pass_action()], [pass_action(), pass_action()]]
    result = run_same_action_replay(configuration, trace)
    assert result.official_statuses == result.fast_statuses == ["DONE", "DONE"]
    assert result.official_rewards == result.fast_rewards == [3000.0, 3000.0]


def test_provenance_guard_rejects_tampered_hash(monkeypatch) -> None:
    from oracle.provenance import OFFICIAL_FILE_SHA256

    monkeypatch.setitem(
        OFFICIAL_FILE_SHA256, "kaggriculture.py", "0" * 64
    )
    with pytest.raises(ProvenanceError):
        verify_official_provenance()


def test_official_backend_rejects_wrong_action_count() -> None:
    from oracle.backend import make_backend

    official = make_backend("official", {"seed": 7})
    official.reset()
    with pytest.raises(ValueError):
        official.step([pass_action()])
