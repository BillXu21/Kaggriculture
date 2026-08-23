"""Stage-2b focused mechanic parity probes: worker inventory, same-turn
ordering, hiring, and market processing against the pinned official 1.32.7
engine.

Every scenario replays the exact same action pair through both engines and
compares the complete canonical state after every submitted pair
(``run_same_action_replay`` raises at the FIRST divergent field). Semantic
assertions pin the official mechanic itself so a jointly-wrong fast engine
cannot pass silently.

Skipped unless ``kaggle_environments`` passes the provenance guard.

Regressions are named for the exact divergence they lock down:

- ``test_regression_money_decode_f32_noise``: fast observation decoding
  recovered money as ``raw * 10000`` without rounding, so every money change
  produced spurious canonical divergences (official 2993.0 vs fast
  2992.999755859375).
- ``test_regression_buy_seed_quantity_clamped_at_100``: the fast engine
  clamped market order quantities to MAX_QUANTITY=100; official order
  quantities are unbounded (BUY_SEED WHEAT 150 costs 1500 and grants 150
  seeds).
- ``test_regression_malformed_actions_must_be_silent_noops``: the fast wire
  translation raised ValueError for inputs the official engine silently
  ignores (11th market order, unknown unit op, seed-name PICKUP, unknown
  PLANT crop, non-dict action, missing farmer).
- ``test_regression_place_animal_zero_quantity_noop``: the fast PLACE
  animal-shed fallback clamped n<=0 up to 1 and moved an animal that the
  official engine leaves carried.
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


# ---------------------------------------------------------------------------
# Worker inventory
# ---------------------------------------------------------------------------


def test_worker_inventory_pickup_arbitrary_n_subject_to_shed() -> None:
    result = replay("pickup_arbitrary_n", [
        [act(market=[["BUY_PRODUCT", "WHEAT", 10]]), pas()],
        [act(farmer=["PICKUP", "WHEAT", 999]), pas()],  # clamped to stock, not a slot count
        [act(farmer=["DROP"]), pas()],
    ])
    assert result.turns_executed == 3


def test_worker_inventory_multi_item_quantity_semantics() -> None:
    # Item->quantity dict semantics: two product types carried and dropped
    # together; canonical compare proves no fixed slot count/multiplicity loss.
    replay("multi_item_carry_drop", [
        [act(market=[["BUY_PRODUCT", "WHEAT", 5], ["BUY_PRODUCT", "FERTILIZER", 3]]), pas()],
        [act(farmer=["PICKUP", "WHEAT", 5]), pas()],
        [act(farmer=["PICKUP", "FERTILIZER", 3]), pas()],
        [act(farmer=["DROP"]), pas()],
    ])


def test_seeds_are_never_worker_carried() -> None:
    # Seeds live in private["seeds"]; PICKUP of a seed name finds nothing in
    # the shed and must be a silent no-op on both engines.
    replay("pickup_seed_name_noop", [
        [act(market=[["BUY_SEED", "WHEAT", 4]]), pas()],
        [act(farmer=["PICKUP", "WHEAT_SEED", 4]), pas()],
    ])


def test_day_end_carried_inventories_drop_into_shed_and_hiring_resets() -> None:
    # Crossing the day boundary: carried wheat lands in the shed, hands and
    # hires_today reset, and the next hire restarts Fibonacci at 1.
    result = replay("day_end_inventory_drop_hire_reset", [
        [act(market=[["HIRE"], ["BUY_PRODUCT", "WHEAT", 4]]), pas()],
        [act(farmer=["PICKUP", "WHEAT", 4], hands=[["EAST"]]), pas()],
        *[[pas(), pas()] for _ in range(22)],
        [act(market=[["HIRE"]]), pas()],  # day 1, hour 0
    ])
    assert result.turns_executed == 25


def test_shed_overflow_is_discarded_on_day_end_drop() -> None:
    # A near-full shed plus a full carry: the day-end drop keeps only what
    # fits; the remainder is discarded (not retained by workers).
    configuration = {"seed": 7, "shedCapacity": 20}
    replay("day_end_overflow_discarded", [
        [act(market=[["BUY_PRODUCT", "WHEAT", 30]]), pas()],       # shed caps at 20
        [act(farmer=["PICKUP", "WHEAT", 20]), pas()],              # shed 0, carry 20
        [act(market=[["BUY_PRODUCT", "WHEAT", 30]]), pas()],       # shed 20 again
        *[[pas(), pas()] for _ in range(21)],
        [pas(), pas()],                                            # day boundary drop
    ], configuration)


# ---------------------------------------------------------------------------
# Same-turn ordering: workers before market
# ---------------------------------------------------------------------------


def test_same_turn_market_buy_cannot_be_picked_up() -> None:
    # Workers act first: a PICKUP submitted the same turn as the BUY_PRODUCT
    # sees an empty shed and must stay empty-handed.
    replay("same_turn_buy_not_pickup", [
        [act(farmer=["PICKUP", "WHEAT", 5], market=[["BUY_PRODUCT", "WHEAT", 5]]), pas()],
    ])


def test_pickup_frees_shed_capacity_before_same_turn_buy() -> None:
    # Shed capped at 20/20; picking up 15 frees room so 6 bought units land
    # (5 fit, the 6th aborts). If market ran first, the whole buy would abort.
    configuration = {"seed": 7, "shedCapacity": 20}
    replay("pickup_frees_room_before_buy", [
        [act(market=[["BUY_PRODUCT", "WHEAT", 25]]), pas()],
        [act(farmer=["PICKUP", "WHEAT", 15], market=[["BUY_PRODUCT", "WHEAT", 6]]), pas()],
    ], configuration)


def test_full_shed_buy_aborts_without_pickup_control() -> None:
    # Control for the ordering probe: with the shed still full and no pickup,
    # the same buy commits nothing.
    configuration = {"seed": 7, "shedCapacity": 20}
    replay("full_shed_buy_aborts", [
        [act(market=[["BUY_PRODUCT", "WHEAT", 25]]), pas()],
        [act(market=[["BUY_PRODUCT", "WHEAT", 1]]), pas()],
    ], configuration)


def test_worker_deposit_lands_before_same_turn_sell() -> None:
    # DROP executes before market SELL, so goods deposited this turn are
    # sellable this turn.
    result = replay("deposit_then_sell_same_turn", [
        [act(market=[["BUY_PRODUCT", "WHEAT", 3]]), pas()],
        [act(farmer=["PICKUP", "WHEAT", 3]), pas()],
        [act(farmer=["DROP"], market=[["SELL", "WHEAT", 3]]), pas()],
    ])
    assert result.turns_executed == 3


# ---------------------------------------------------------------------------
# Hiring
# ---------------------------------------------------------------------------


def test_hire_fibonacci_prices_repeated_hires_and_new_hand_timing() -> None:
    # Costs 1,1,2 then 3,5,8 (mult=1); six new hands act one turn after being
    # hired, never on the hire turn itself.
    result = replay("hire_fibonacci_repeat", [
        [act(market=[["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"]]), pas()],
        [act(market=[["HIRE"], ["HIRE"]]), pas()],
        [act(hands=[["EAST"]] * 6), pas()],
    ])
    assert result.turns_executed == 3


def test_hire_unaffordable_stops_silently() -> None:
    # 20 HIRE orders in one day outrun 3000 money long before slot 10;
    # unaffordable hires are silent no-ops that stop the Fibonacci counter.
    replay("hire_unaffordable_stops", [
        [act(market=[["HIRE"]] * 10), pas()],
        [act(market=[["HIRE"]] * 10), pas()],
    ])


def test_hire_fibonacci_scales_with_cost_multiplier() -> None:
    replay("hire_mult_3", [
        [act(market=[["HIRE"], ["HIRE"], ["HIRE"]]), pas()],
    ], {"seed": 7, "farmHandCostMult": 3})


def test_hire_at_hour_zero_then_daily_reset() -> None:
    # Hires at hour 0 of day 0 and again at hour 0 of day 1: hires_today
    # reset makes the second day's first hand cost 1 again.
    replay("hire_hour0_daily_reset", [
        [act(market=[["HIRE"], ["HIRE"]]), pas()],
        *[[pas(), pas()] for _ in range(22)],
        [act(market=[["HIRE"], ["HIRE"]]), pas()],
    ])


# ---------------------------------------------------------------------------
# Market processing
# ---------------------------------------------------------------------------


def test_market_orders_truncated_to_ten_slots() -> None:
    result = replay("eleven_orders_truncated", [
        [act(market=[["BUY_SEED", "WHEAT", 1]] * 11), pas()],
    ])
    assert result.turns_executed == 1
    # Semantic pin: exactly the first 10 orders committed.
    from oracle.backend import make_backend

    fast = make_backend("fast", {"seed": 7})
    fast.reset()
    fast.step([act(market=[["BUY_SEED", "WHEAT", 1]] * 11), pas()])
    canonical = fast.canonical_state()
    assert canonical["privates"][0]["seeds"]["WHEAT"] == 10
    assert canonical["farms"][0]["money"] == 2900.0


def test_market_both_players_lockstep_per_slot() -> None:
    replay("both_players_lockstep", [
        [act(market=[["BUY_SEED", "WHEAT", 4]]), act(market=[["BUY_SEED", "WHEAT", 4]])],
        [act(market=[["BUY_PRODUCT", "WHEAT", 30]]), act(market=[["BUY_PRODUCT", "WHEAT", 30]])],
        [act(market=[["SELL", "WHEAT", 30]]), act(market=[["SELL", "WHEAT", 30]])],
    ])


def test_market_cross_buy_sell_same_slot_same_precommit_price() -> None:
    # Player 0 sells wheat while player 1 buys it in the same slot: both
    # units quote from the same pre-commit inventory each step.
    replay("cross_buy_sell_same_turn", [
        [act(market=[["SELL", "WHEAT", 2]]), act(market=[["BUY_PRODUCT", "WHEAT", 2]])],
    ])


def test_market_insufficient_funds_commits_then_aborts_order() -> None:
    # MELON seeds cost 80: 40 requested commits 37 (2960 of 3000), then the
    # order aborts; later slots still run.
    replay("insufficient_funds_partial_commit", [
        [act(market=[["BUY_SEED", "MELON", 40], ["BUY_SEED", "WHEAT", 1]]), pas()],
    ])


def test_market_shed_capacity_aborts_order_midway() -> None:
    configuration = {"seed": 7, "shedCapacity": 20}
    replay("shed_capacity_partial_commit", [
        [act(market=[["BUY_PRODUCT", "WHEAT", 25]]), pas()],
    ], configuration)


def test_market_large_quantities_bound_by_resources_not_constants() -> None:
    replay("large_quantities", [
        [act(market=[["BUY_PRODUCT", "WHEAT", 50]]), pas()],
        [act(market=[["SELL", "WHEAT", 200]]), pas()],
        [act(market=[["BUY_SEED", "WHEAT", 150]]), pas()],
    ])


def test_market_same_turn_roundtrip_nets_zero() -> None:
    # BUY quotes at post-buy inventory, so a same-turn BUY 1 / SELL 1 against
    # an unchanged market must net exactly zero money.
    result = replay("roundtrip_nets_zero", [
        [act(market=[["BUY_PRODUCT", "WHEAT", 1], ["SELL", "WHEAT", 1]]), pas()],
    ])
    assert result.turns_executed == 1
    from oracle.backend import make_backend

    fast = make_backend("fast", {"seed": 7})
    fast.reset()
    fast.step([act(market=[["BUY_PRODUCT", "WHEAT", 1], ["SELL", "WHEAT", 1]]), pas()])
    assert fast.canonical_state()["farms"][0]["money"] == 3000.0


def test_market_buy_land_atomic_twice_per_turn() -> None:
    # Two BUY_LAND orders unlock NE (1000) and SW (2000) in one turn; each
    # order is atomic and ignores any extra arguments.
    replay("double_buy_land_atomic", [
        [act(market=[["BUY_LAND"], ["BUY_LAND", "junk", 99]]), pas()],
    ])


def test_market_hire_and_buy_land_atomic_mix() -> None:
    replay("hire_buy_land_atomic_mix", [
        [act(market=[["HIRE"], ["BUY_LAND"], ["HIRE"]]), pas()],
        [act(hands=[["EAST"], ["WEST"]]), pas()],
    ])


def test_market_mixed_ten_order_turn_with_truncation() -> None:
    replay("mixed_ten_order_slots", [
        [act(market=[
            ["HIRE"],
            ["BUY_SEED", "WHEAT", 3],
            ["BUY_PRODUCT", "WHEAT", 10],
            ["BUY_ANIMAL", "GOOSE", 1],
            ["SELL", "FERTILIZER", 1],
            ["BUY_LAND"],
            ["BUY_SEED", "CARROT", 2],
            ["BUY_PRODUCT", "FERTILIZER", 2],
            ["HIRE"],
            ["BUY_SEED", "TOMATO", 1],
            ["BUY_SEED", "MELON", 1],  # 11th: truncated
        ]), pas()],
        [act(hands=[["EAST"], ["WEST"]]), pas()],
    ])


# ---------------------------------------------------------------------------
# Regressions named for the exact divergences they lock down
# ---------------------------------------------------------------------------


def test_regression_money_decode_f32_noise() -> None:
    # First found as: official money 2993.0 vs fast 2992.999755859375 after
    # four hires (3000 - (1+1+2+3)); the f32 normalize(10000) observation
    # round-trip must be inverted by rounding, not trusted raw.
    result = replay("money_decode_exactness", [
        [act(market=[["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"]]), pas()],
    ])
    assert result.turns_executed == 1
    from oracle.backend import make_backend

    fast = make_backend("fast", {"seed": 7})
    fast.reset()
    fast.step([act(market=[["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"]]), pas()])
    assert fast.canonical_state()["farms"][0]["money"] == 2993.0


def test_regression_buy_seed_quantity_clamped_at_100() -> None:
    # First found as: official money 1500.0 vs fast 2000.0 after
    # BUY_SEED WHEAT 150 (fast granted only MAX_QUANTITY=100 seeds).
    result = replay("buy_seed_150", [
        [act(market=[["BUY_SEED", "WHEAT", 150]]), pas()],
    ])
    assert result.turns_executed == 1
    from oracle.backend import make_backend

    fast = make_backend("fast", {"seed": 7})
    fast.reset()
    fast.step([act(market=[["BUY_SEED", "WHEAT", 150]]), pas()])
    canonical = fast.canonical_state()
    assert canonical["privates"][0]["seeds"]["WHEAT"] == 150
    assert canonical["farms"][0]["money"] == 1500.0


def test_regression_malformed_actions_must_be_silent_noops() -> None:
    # First found as ValueError from the fast wire translation for inputs the
    # official engine silently ignores.
    result = replay("malformed_silent_noops", [
        [act(farmer=["FLY"]), pas()],
        [{"hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}, ""],  # missing farmer / non-dict
        [act(farmer=["PLANT", "NOPE"], market=[["BUY_SEED", "WHEAT", 1]]), pas()],
        [act(farmer=["PICKUP", "WHEAT_SEED"]), pas()],
        [act(market=[["BUY_SEED", "WHEAT", "lots"], ["BUY_SEED", "WHEAT", 2]]), pas()],
    ])
    assert result.turns_executed == 5


def test_regression_place_animal_zero_quantity_noop() -> None:
    # First found by inspection: the fast PLACE animal-shed fallback clamped
    # n<=0 up to 1, moving an animal the official engine leaves carried.
    replay("place_animal_zero_noop", [
        [act(market=[["BUY_ANIMAL", "GOOSE", 1]]), pas()],
        [act(farmer=["PICKUP", "GOOSE", 1]), pas()],
        [act(farmer=["PLACE", "GOOSE", 0]), pas()],
        [act(farmer=["DROP"]), pas()],
    ])
