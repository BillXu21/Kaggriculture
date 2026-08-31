"""Exact E economic-history semantics through the runner seam (issue #9 A2).

The runner feeds `bc_manager.live.encode_live_inputs` via the stateless
`economic_prev_start=(prev_day, prev_cash)` path. These tests pin that path
against the authoritative `EconomicHistory` tracker for: initial episode
state, exact adjacent day, day gaps, backwards days, and per-episode reset —
for both seats. No formulas are copied: both paths run the authoritative
encoder and tracker.
"""

import numpy as np

from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    E_HISTORY_LEGACY,
    EconomicHistory,
    previous_net_cash,
)
from bc_manager.live import encode_live_inputs


def _obs(day: int, money_seat0: float, money_seat1: float):
    def farm(money: float) -> dict:
        return {"farmer": [0, 0], "hands": [], "hires_today": 0,
                "money": money, "tiles": [[None] * 10 for _ in range(10)],
                "unlocked_quadrants": ["NW"]}

    return {
        "day": day, "hour": 0, "step": day * 24,
        "farms": [farm(money_seat0), farm(money_seat1)],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


def _encode(obs, seat, prev_start):
    return encode_live_inputs(
        obs, seat, {"workers_hired": 0, "hire_cost": 0}, step=obs["day"] * 24,
        economic_prev_start=prev_start)


def _encode_history(obs, seat, history: EconomicHistory):
    return encode_live_inputs(
        obs, seat, {"workers_hired": 0, "hire_cost": 0}, step=obs["day"] * 24,
        economic_history=history)


def _context(inputs) -> np.ndarray:
    context = inputs["economic_context"]
    assert context.shape == (1, 14)
    return context[0]


def test_initial_episode_state_matches_tracker_invalid_delta():
    # First observation of an episode/seat: no valid previous daily start.
    obs = _obs(3, 2500.0, 3100.0)
    history = EconomicHistory()
    _context(_encode_history(obs, 0, history))  # observes into the tracker
    # The runner never passes a prev-start before the first observed day;
    # encode_live_inputs emits no economic_context without either argument.
    raw = encode_live_inputs(
        obs, 0, {"workers_hired": 0, "hire_cost": 0}, step=obs["day"] * 24)
    assert "economic_context" not in raw
    # Tracker's first observe is invalid-delta; the runner reproduces the
    # same invalid-delta context once a (None -> first-day) handoff occurs.
    delta, valid = history.observe(3, 2500.0)
    assert valid is False and delta == 0.0


def test_exact_adjacent_day_matches_tracker_both_seats():
    money = [2100.5, 3300.25]
    histories = [EconomicHistory(), EconomicHistory()]
    for seat in (0, 1):
        histories[seat].observe(3, money[seat])
    obs = _obs(4, money[0] + 120.0, money[1] - 55.0)
    for seat in (0, 1):
        via_prev = _context(_encode(obs, seat, (3, money[seat])))
        via_hist = _context(_encode_history(obs, seat, histories[seat]))
        assert np.array_equal(via_prev, via_hist), seat
        # Valid-delta flag set; signed-log delta channel reflects movement.
        assert float(via_prev[13]) == 1.0
        if money[seat] != 0.0:
            assert float(via_prev[12]) != 0.0


def test_day_gap_yields_invalid_delta_matching_tracker():
    histories = [EconomicHistory(), EconomicHistory()]
    for seat in (0, 1):
        histories[seat].observe(1, 2800.0)
    obs = _obs(4, 3000.0, 2600.0)
    for seat in (0, 1):
        via_prev = _context(_encode(obs, seat, (1, 2800.0)))
        via_hist = _context(_encode_history(obs, seat, histories[seat]))
        assert np.array_equal(via_prev, via_hist), seat


def test_backwards_day_yields_invalid_delta_matching_tracker():
    history = EconomicHistory()
    history.observe(6, 2400.0)
    obs = _obs(4, 2300.0, 2900.0)
    assert np.array_equal(
        _context(_encode(obs, 0, (6, 2400.0))),
        _context(_encode_history(obs, 0, history)))


def test_reset_between_episodes_matches_fresh_tracker():
    obs = _obs(4, 2750.0, 3050.0)
    stale = EconomicHistory()
    stale.observe(29, 40000.0)  # previous episode's last daily start
    fresh = EconomicHistory()
    fresh.observe(3, 2700.0)
    # A new episode must behave like the fresh tracker at its first day-4
    # decision only when the runner hands it the fresh d3 daily start.
    # NOTE: encode_live_inputs(economic_history=...) observes INTO the
    # tracker, so each comparison below needs its own fresh tracker at the
    # same episode-local state (d3 start observed, d4 being encoded).
    assert np.array_equal(
        _context(_encode(obs, 0, (3, 2700.0))),
        _context(_encode_history(obs, 0, fresh)))
    # And must NOT reuse the stale cross-episode previous start: prev day 29
    # is a gap before day 4, so its delta is invalid while the fresh tracker
    # records an exact adjacent d3->d4 delta.
    fresh_adjacent = EconomicHistory()
    fresh_adjacent.observe(3, 2700.0)
    assert not np.array_equal(
        _context(_encode(obs, 0, (29, 40000.0))),
        _context(_encode_history(obs, 0, fresh_adjacent)))


def test_runner_style_daily_start_tracking_is_adjacent_only():
    """Mirror of the runner's per-seat daily_start bookkeeping: deltas are
    valid exactly when the previous daily start was the immediately
    preceding day of the SAME episode/seat."""
    history = EconomicHistory()
    daily_start = None
    contexts = []
    for day in range(28, 31):
        money = 1000.0 + 10.0 * day
        obs = _obs(day % 30, money, money)
        if daily_start is not None:
            inputs = _encode(obs, 0, daily_start)
        else:
            inputs = encode_live_inputs(
                obs, 0, {"workers_hired": 0, "hire_cost": 0},
                step=obs["day"] * 24)
        if "economic_context" in inputs:
            contexts.append((day, _context(inputs)))
        daily_start = (obs["day"], money)
        history.observe(obs["day"], money)
    # Day 28 had no previous start -> no context row recorded by the runner.
    # Day 29 is exactly adjacent -> a valid-delta row. The wrapped day-0
    # observation still emits a row (the encoder always emits once a
    # prev-start exists) but with an INVALID delta, because prev day 29 is
    # not day-1 of day 0 — matching the tracker's gap semantics.
    assert [day for day, _ in contexts] == [29, 30]
    assert [float(row[13]) for _, row in contexts] == [1.0, 0.0]
    assert float(contexts[1][1][12]) == 0.0  # invalid delta channel stays 0


def test_history_versions_are_explicit_and_legacy_is_zero_invalid():
    assert previous_net_cash(
        E_HISTORY_CORRECTED_V1, 5, 125.0, (4, 100.0)) == (25.0, True)
    assert previous_net_cash(
        E_HISTORY_CORRECTED_V1, 4, 125.0, (3, 100.0)) == (25.0, True)
    assert previous_net_cash(
        E_HISTORY_LEGACY, 5, 125.0, (4, 100.0)) == (0.0, False)

    obs = _obs(5, 125.0, 225.0)
    legacy = _context(encode_live_inputs(
        obs, 0, {"workers_hired": 0, "hire_cost": 0}, step=120,
        economic_prev_start=(4, 100.0),
        e_history_version=E_HISTORY_LEGACY))
    assert legacy[12] == 0.0 and legacy[13] == 0.0
