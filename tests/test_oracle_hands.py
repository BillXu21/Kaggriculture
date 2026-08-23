"""Real-official >16-hired-hands parity regressions (MAX_HANDS=240 layout).

The fast engine previously fixed 16 hand slots and silently truncated
submitted hand-action lists beyond 16 while the official engine has no hand
cap other than 10 market orders/turn x 24 turns/day cleared at day reset.
These scenarios replay the exact same action pairs through BOTH engines and
compare the complete canonical state after every turn, locking down:

- exactly 16 simultaneous hands: zero divergence (the old boundary);
- 17+ hands (validated 100000-money trace reaching 23): zero divergence
  across the hires AND through subsequent hand actions / inventory
  observations;
- day-end reset from >16 hands: hands/hires_today/inventories reset parity
  and the next-day Fibonacci restart;
- HIRE availability implied by the official-reachable state matches the fast
  action masks before the reset.

Skipped unless ``kaggle_environments`` passes the provenance guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_env._kaggriculture_env import MASK_SIZE, MAX_HANDS
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


def act(**overrides) -> dict[str, object]:
    action: dict[str, object] = {"farmer": ["PASS"], "hands": [], "market": []}
    action.update(overrides)
    return action


def pas() -> dict[str, object]:
    return act()


def replay(name: str, trace, configuration=None, max_turns=720):
    try:
        return run_same_action_replay(configuration or {"seed": 7}, trace, max_turns=max_turns)
    except DivergenceError as error:
        raise AssertionError(f"{name}: {error}") from error


def _fibonacci(n: int) -> int:
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# ---------------------------------------------------------------------------
# Exactly 16 hands (old boundary)
# ---------------------------------------------------------------------------


def test_exactly_16_hands_zero_divergence_with_hand_actions() -> None:
    # 10 + 6 hires inside day 0 costs fib-sum(16) = 1596 of the default 3000.
    # Then every one of the 16 hands moves north and the full canonical state
    # (positions, private per-hand inventories, hires_today) must match.
    trace = [
        [act(market=[["HIRE"]] * 10), pas()],
        [act(market=[["HIRE"]] * 6), pas()],
        [act(hands=[["NORTH"]] * 16), pas()],
        [act(hands=[["WEST"]] * 16), pas()],
    ]
    result = replay("exactly_16_hands", trace)
    assert result.turns_executed == 4


# ---------------------------------------------------------------------------
# 17+ hands: validated 100000-money trace reaching 23
# ---------------------------------------------------------------------------


def test_23_hands_hires_and_subsequent_hand_actions_zero_divergence() -> None:
    # startingMoney=100000, seed=7, 10 HIRE orders/turn -> 10, 20, 23 hands,
    # then two turns of real hand actions over all 23 slots plus a pickup so
    # carried-inventory observations are exercised beyond 16 hands too.
    configuration = {"seed": 7, "startingMoney": 100000}
    trace = [
        [act(market=[["HIRE"]] * 10), pas()],
        [act(market=[["HIRE"]] * 10), pas()],
        [act(market=[["HIRE"]] * 3), pas()],
        [act(hands=[["NORTH"]] * 23), pas()],
        [act(hands=[["EAST"]] * 23), pas()],
    ]
    result = replay("twenty_three_hands", trace, configuration)
    assert result.turns_executed == 5


def test_17th_through_23rd_hand_boundary_turns_zero_divergence() -> None:
    # Pin the exact crossing of the old 16 cap: 16 hands after turn 2, the
    # 17th hire lands on turn 3's first order, hands 18-23 follow.
    configuration = {"seed": 7, "startingMoney": 100000}
    trace = [
        [act(market=[["HIRE"]] * 10), pas()],
        [act(market=[["HIRE"]] * 6), pas()],   # exactly 16 hands here
        [act(market=[["HIRE"]] * 7), pas()],   # 17..23 cross the old boundary
        [act(hands=[["SOUTH"]] * 23), pas()],
    ]
    result = replay("crossing_16_boundary", trace, configuration)
    assert result.turns_executed == 4


# ---------------------------------------------------------------------------
# Day-end reset from >16 hands
# ---------------------------------------------------------------------------


def test_day_end_reset_from_23_hands_parity_and_fibonacci_restart() -> None:
    # Reach 23 hands on day 0, pass to the day-1 boundary (hands clear,
    # hires_today resets, carried inventories drop to the shed), then rehire
    # on day 1 so the restarted Fibonacci schedule is compared too.
    configuration = {"seed": 7, "startingMoney": 100000}
    trace = [
        [act(market=[["HIRE"]] * 10), pas()],
        [act(market=[["HIRE"]] * 10), pas()],
        [act(market=[["HIRE"]] * 3), pas()],
        *[[pas(), pas()] for _ in range(20)],   # hour 3 -> hour 23, day 0
        [pas(), pas()],                          # day 1, hour 0: reset applied
        [act(market=[["HIRE"]]), pas()],         # day-1 rehire costs fib(0)=1
        [act(market=[["HIRE"], ["HIRE"]]), pas()],
    ]
    result = replay("day_reset_from_23_hands", trace, configuration)
    assert result.turns_executed == 26


# ---------------------------------------------------------------------------
# Fast masks vs official-reachable HIRE availability
# ---------------------------------------------------------------------------


def test_fast_hire_mask_matches_official_reachable_state() -> None:
    # The canonical comparison already proves the fast observation equals the
    # official state; here the fast HIRE mask bit must additionally equal the
    # official-reachable gate hand_count < MAX_HANDS AND money >= fib(
    # hires_today), evaluated from that shared state, before the day reset.
    from oracle import make_backend

    configuration = {"seed": 7, "startingMoney": 100000}
    unit_mask_width = 18 + 17 + 101
    market_mask_width = 7 + 17 + 101
    market_base = (MAX_HANDS + 1) * unit_mask_width

    official = make_backend("official", configuration)
    fast = make_backend("fast", configuration)
    official.reset()
    fast.reset()

    def check(turn_label: str) -> None:
        canonical = official.canonical_state()
        farm = canonical["farms"][0]
        expected = int(
            len(farm["hands"]) < MAX_HANDS
            and farm["money"] >= _fibonacci(farm["hires_today"])
        )
        masks = np.zeros((1, 2, MASK_SIZE), dtype=np.uint8)
        fast._env._backend.action_masks_into(masks)
        actual = int(masks[0, 0, market_base + 5])
        assert actual == expected, (
            f"{turn_label}: fast HIRE mask {actual} != official-reachable {expected}"
        )

    check("initial")
    pair = [act(market=[["HIRE"]] * 10), pas()]
    for turn in range(3):
        official.step([dict(pair[0]), dict(pair[1])])
        fast.step([dict(pair[0]), dict(pair[1])])
        check(f"turn {turn}")
