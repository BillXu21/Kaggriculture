"""Stage-2b focused animal/structure/fertilizer lifecycle parity probes against
the pinned official 1.32.7 engine.

Every scenario replays the exact same action pair through both engines and
compares the complete canonical state after every submitted pair
(``run_same_action_replay`` raises at the FIRST divergent field). Semantic
assertions pin the official mechanic itself (values taken from the pinned
source symbols cited inline) so a jointly-wrong fast engine cannot pass
silently.

Official source of truth (kaggle-environments 1.32.7, commit
28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c,
``kaggle_environments/envs/kaggriculture/kaggriculture.py``):

- ``ANIMALS`` constant table (cost/structure/first_yield_day/interval/
  max_held/product per species);
- ``_new_animal`` (fresh tile dict: ``placed_day=day``, ``yield_units=0``,
  ``consecutive_unfed=0``, ``fed_today=False``, ``cared_today=False``,
  ``fertilizer_available=False``, ``pending_care_bonus=0``);
- ``_apply_unit_action`` arms ``BUILD_COOP``/``BUILD_PASTURE`` (standing tile
  must be owned and completely empty -- no plant, weed, or structure; the
  build itself is FREE), ``PLACE`` (exactly 1 carried animal onto a matching
  unoccupied structure; otherwise falls through to the shed-drop path, which
  needs shed adjacency and silently no-ops on ``n <= 0`` or zero room),
  ``FEED`` (animal tile, not already fed today, consumes 1 carried WHEAT),
  ``CARE`` (animal tile, once per day, free),
  ``COLLECT_FERTILIZER`` (animal tile with ``fertilizer_available``; adds 1
  FERTILIZER to the acting unit's carried inventory),
  ``DIG`` (silent no-op on a tile holding a placed animal; otherwise clears
  plants/weeds/empty structures), ``HARVEST`` (drains ALL accumulated
  ``yield_units`` into the acting unit's inventory);
- ``_commit_unit`` arm ``BUY_ANIMAL`` (per-unit commit: fails when
  ``farm["money"] < price`` OR the shed is at ``shedCapacity``; animals land
  in the private SHED, never the market inventory);
- ``_daily_refresh_animals`` (end-of-day, in order: unfed-streak update;
  ``consecutive_unfed >= 2`` -> animal ESCAPES and the bare structure dict
  REMAINS; else production ``days_since_first = next_day - placed_day -
  first_yield_day`` hitting ``% interval == 0`` adds ``base=1`` plus the
  pending-care bonus ONLY when ``fed_today`` (an unfed production day still
  resets ``pending_care_bonus`` to 0 -- accumulated bonuses are LOST);
  ``cared_today and fed_today`` accrues ``pending_care_bonus += 1``;
  ``fertilizer_available = True`` regenerates every day for surviving
  animals; ``fed_today``/``cared_today`` reset);
- ``_drop_inventories_to_shed`` (end-of-day: farmer then hands, each
  inventory in insertion order, shed capacity bounded, overflow DISCARDED);
- ``_spawn_hand`` (first free shed-access tile, NWSE preference).

Skipped unless ``kaggle_environments`` passes the provenance guard.
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

# Literal transcription of the pinned official ``ANIMALS`` table (see module
# docstring). Locked against fast-engine behavior by the probes below.
OFFICIAL_ANIMALS_TABLE = {
    "GOOSE": {"cost": 300, "structure": "COOP", "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW": {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

CONFIG = {"seed": 7}


def act(**overrides) -> dict[str, object]:
    action: dict[str, object] = {"farmer": ["PASS"], "hands": [], "market": []}
    action.update(overrides)
    return action


def pas() -> dict[str, object]:
    return act()


def build_turns(events, total_days) -> list[list[dict[str, object]]]:
    """Turn grid from ``(day, hour, p0_action[, p1_action])`` events."""
    normalized = []
    for event in events:
        day, hour, p0 = event[0], event[1], event[2]
        p1 = event[3] if len(event) > 3 else pas()
        normalized.append(((day, hour), [p0, p1]))
    grid = dict(normalized)
    assert len(grid) == len(normalized), "duplicate (day, hour) event"
    turns = []
    for day in range(total_days):
        for hour in range(24):
            pair = grid.get((day, hour))
            turns.append(pair if pair is not None else [pas(), pas()])
    return turns


def replay(name: str, trace, configuration=None, max_turns=720):
    try:
        return run_same_action_replay(configuration or CONFIG, trace, max_turns=max_turns)
    except DivergenceError as error:
        raise AssertionError(f"{name}: {error}") from error


def fast_timeline(configuration, trace):
    """Fast-only canonical states: index i is the state after i turn pairs."""
    from oracle.backend import make_backend

    fast = make_backend("fast", configuration)
    fast.reset()
    timeline = [fast.canonical_state()]
    for pair in trace:
        fast.step(pair)
        timeline.append(fast.canonical_state())
    return timeline


def tile(state, x: int, y: int, farm: int = 0):
    return state["farms"][farm]["tiles"][y][x]


def turn_index(day: int, hour: int) -> int:
    return day * 24 + hour + 1


# ---------------------------------------------------------------------------
# Constants table
# ---------------------------------------------------------------------------


def test_official_animal_constants_table_is_locked() -> None:
    # The probes below derive every expectation from this literal table; it
    # must keep matching the pinned official source symbol ``ANIMALS``.
    assert OFFICIAL_ANIMALS_TABLE["GOOSE"]["structure"] == "COOP"
    assert OFFICIAL_ANIMALS_TABLE["COW"]["structure"] == "PASTURE"
    assert OFFICIAL_ANIMALS_TABLE["SHEEP"]["structure"] == "PASTURE"
    assert sum(entry["cost"] for entry in OFFICIAL_ANIMALS_TABLE.values()) == 1200
    assert {entry["product"] for entry in OFFICIAL_ANIMALS_TABLE.values()} == {"EGG", "MILK", "WOOL"}


# ---------------------------------------------------------------------------
# GOOSE/COOP lifecycle: production timing, fertilizer, harvest drain
# ---------------------------------------------------------------------------


def test_goose_lifecycle_production_fertilizer_and_harvest() -> None:
    name = "goose_lifecycle"
    events = [
        # Day 0: buy, carry, build a FREE coop at (3,3), place, stock wheat.
        (0, 0, act(market=[["BUY_ANIMAL", "GOOSE", 1]])),
        (0, 1, act(farmer=["PICKUP", "GOOSE"])),
        (0, 2, act(farmer=["WEST"])),
        (0, 3, act(farmer=["NORTH"])),
        (0, 4, act(farmer=["BUILD_COOP"])),
        (0, 5, act(farmer=["PLACE", "GOOSE"])),
        (0, 6, act(market=[["BUY_PRODUCT", "WHEAT", 12]])),
    ]
    # Days 1..6: feed + care + collect fertilizer daily.
    for day in range(1, 7):
        events += [
            (day, 0, act(farmer=["PICKUP", "WHEAT", 1])),
            (day, 1, act(farmer=["NORTH"])),
            (day, 2, act(farmer=["WEST"])),
            (day, 3, act(farmer=["FEED"])),
            (day, 4, act(farmer=["CARE"])),
            (day, 5, act(farmer=["COLLECT_FERTILIZER"])),
            (day, 6, act(farmer=["EAST"])),
            (day, 7, act(farmer=["SOUTH"])),
        ]
    # Day 7: harvest drains ALL accumulated yield in one call.
    events += [
        (7, 1, act(farmer=["NORTH"])),
        (7, 2, act(farmer=["WEST"])),
        (7, 3, act(farmer=["HARVEST"])),
    ]
    trace = build_turns(events, 8)
    result = replay(name, trace)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(CONFIG, trace)
    # Placed day 0, first_yield_day 4, interval 1: first production at the
    # end of day 3 (days_since_first == 0), i.e. visible in the state that
    # follows the day-3 boundary refresh. Care+feed daily means the
    # pending-care bonus accrued on days 1-2 (bonus 2) is consumed on the
    # first fed production day, then re-accrues; the yield is capped at
    # max_held == 4 either way by the end of day 6.
    mid = timeline[turn_index(3, 23)]
    goose_mid = tile(mid, 3, 3)
    assert goose_mid["animal"] == "GOOSE"
    assert goose_mid["yield_units"] == 3  # 1 base + 2 pending-care bonus
    assert goose_mid["pending_care_bonus"] == 1  # re-accrued on day 3 itself
    final = timeline[-1]
    goose = tile(final, 3, 3)
    assert goose["kind"] == "COOP" and goose["animal"] == "GOOSE"
    # The day-7 HARVEST drained the accumulated 4; the day-7 boundary
    # refresh then produced +1 more (days_since_first == 4, interval 1).
    assert goose["yield_units"] == 1
    assert goose["consecutive_unfed"] == 1  # day 7 itself had no FEED event
    assert goose["fertilizer_available"] is True  # regenerated end of day 6
    # 4 production events (ends of days 3..6) capped at max_held 4, drained
    # by HARVEST and moved to the shed by the day-7 boundary drop.
    assert final["privates"][0]["shed"]["EGG"] == 4
    assert final["privates"][0]["inventories"][0]["EGG"] == 0
    # One COLLECT_FERTILIZER per day for six days, dropped to the shed nightly.
    assert final["privates"][0]["shed"]["FERTILIZER"] == 6
    # 12 wheat bought, 6 consumed by FEED.
    assert final["privates"][0]["shed"]["WHEAT"] == 6
    # The goose never starved and never escaped.
    assert final["privates"][0]["shed"]["GOOSE"] == 0


# ---------------------------------------------------------------------------
# COW+SHEEP/PASTURE: differing timings, care-bonus loss, max_held caps
# ---------------------------------------------------------------------------


def test_cow_sheep_pasture_timings_care_bonus_and_caps() -> None:
    name = "cow_sheep_timings"
    events = [
        (0, 0, act(market=[
            ["BUY_ANIMAL", "COW", 1],
            ["BUY_ANIMAL", "SHEEP", 1],
        ])),
        # Cow -> pasture at (3,3); sheep -> pasture at (2,3).
        (0, 1, act(farmer=["PICKUP", "COW"])),
        (0, 2, act(farmer=["WEST"])),
        (0, 3, act(farmer=["NORTH"])),
        (0, 4, act(farmer=["BUILD_PASTURE"])),
        (0, 5, act(farmer=["PLACE", "COW"])),
        (0, 6, act(farmer=["WEST"])),
        (0, 7, act(farmer=["BUILD_PASTURE"])),
        (0, 8, act(farmer=["EAST"])),
        (0, 9, act(farmer=["SOUTH"])),
        (0, 10, act(farmer=["EAST"])),
        (0, 11, act(farmer=["PICKUP", "SHEEP"])),
        (0, 12, act(farmer=["WEST"])),
        (0, 13, act(farmer=["NORTH"])),
        (0, 14, act(farmer=["WEST"])),
        (0, 15, act(farmer=["PLACE", "SHEEP"])),
        (0, 16, act(market=[["BUY_PRODUCT", "WHEAT", 40]])),
    ]
    for day in range(1, 13):
        events.append((day, 0, act(farmer=["PICKUP", "WHEAT", 3])))
        events.append((day, 1, act(farmer=["NORTH"])))
        events.append((day, 2, act(farmer=["WEST"])))  # cow tile (3,3)
        if day != 7:
            events.append((day, 3, act(farmer=["FEED"])))  # cow unfed ONLY on day 7
        if day in (5, 6):
            events.append((day, 4, act(farmer=["CARE"])))  # accrue bonus days 5-6
        events.append((day, 5, act(farmer=["WEST"])))  # sheep tile (2,3)
        events.append((day, 6, act(farmer=["FEED"])))
        events.append((day, 7, act(farmer=["CARE"])))
        events.append((day, 8, act(farmer=["COLLECT_FERTILIZER"])))
        events.append((day, 9, act(farmer=["EAST"])))
        events.append((day, 10, act(farmer=["EAST"])))
        events.append((day, 11, act(farmer=["SOUTH"])))
    events += [
        (13, 1, act(farmer=["NORTH"])),
        (13, 2, act(farmer=["WEST"])),
        (13, 3, act(farmer=["HARVEST"])),  # cow MILK
        (13, 4, act(farmer=["WEST"])),
        (13, 5, act(farmer=["HARVEST"])),  # sheep WOOL
    ]
    trace = build_turns(events, 14)
    result = replay(name, trace)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(CONFIG, trace)
    # COW: placed day 0, first_yield_day 8, interval 2 -> productions at the
    # ends of days 7, 9, 11. Care on days 5-6 (fed) accrues pending_care_bonus
    # 2, but day 7 is an UNFED production day: base +1 applies, the bonus is
    # NOT consumed, and pending_care_bonus is still reset to 0 (lost).
    after_day7 = timeline[turn_index(7, 23)]
    cow = tile(after_day7, 3, 3)
    assert cow["animal"] == "COW"
    assert cow["yield_units"] == 1
    assert cow["pending_care_bonus"] == 0  # bonus lost on the unfed production day
    assert cow["consecutive_unfed"] == 1  # one unfed day: not yet escaped
    # Fed again from day 8: productions at ends of days 9 and 11 add +1 each.
    final = timeline[-1]
    # The day-13 HARVEST drained 3 MILK and the night drop moved it to the
    # shed; the day-13 boundary refresh produced +1 more (days_since_first
    # == 6, still on schedule).
    assert tile(final, 3, 3)["yield_units"] == 1
    assert final["privates"][0]["shed"]["MILK"] == 3
    assert final["privates"][0]["inventories"][0]["MILK"] == 0
    # SHEEP: placed day 0, first_yield_day 6, interval 3 -> productions at the
    # ends of days 5, 8, 11. Fed+cared daily: pending 4 consumed on day 5
    # (yield 5), pending 3 consumed on day 8 (capped at max_held 6), day 11
    # stays capped. Harvest drains 6 WOOL == max_held.
    after_day5 = timeline[turn_index(5, 23)]
    sheep = tile(after_day5, 2, 3)
    assert sheep["animal"] == "SHEEP"
    assert sheep["yield_units"] == 5  # 1 base + 4 pending-care bonus
    # Harvested 6 == max_held on day 13; the night drop moved it to the shed
    # and the day-13 boundary produced nothing (8 % interval 3 != 0).
    assert final["privates"][0]["shed"]["WOOL"] == 6
    assert tile(final, 2, 3)["yield_units"] == 0
    # Twelve daily COLLECT_FERTILIZER calls at the sheep tile.
    assert final["privates"][0]["shed"]["FERTILIZER"] == 12


# ---------------------------------------------------------------------------
# BUY_ANIMAL: funds / shed-capacity partial fills, malformed orders
# ---------------------------------------------------------------------------


def test_buy_animal_partial_fill_on_insufficient_funds() -> None:
    name = "buy_animal_funds"
    config = {"seed": 7, "startingMoney": 700}
    events = [
        # 3 geese cost 900 > 700: the official per-unit commit loop buys 2
        # and aborts the order when the money check fails (_commit_unit).
        # Malformed orders ("DRAGON", a unit op in the market queue) are
        # skipped by _parse_order without touching state.
        (0, 0, act(market=[
            ["BUY_ANIMAL", "GOOSE", 3],
            ["BUY_ANIMAL", "DRAGON", 1],
            ["FEED"],
        ])),
    ]
    trace = build_turns(events, 2)
    result = replay(name, trace, config)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(config, trace)
    final = timeline[-1]
    assert final["privates"][0]["shed"]["GOOSE"] == 2
    assert final["farms"][0]["money"] == 100.0


def test_buy_animal_partial_fill_on_shed_capacity_and_overflow_discard() -> None:
    name = "buy_animal_capacity"
    config = {"seed": 7, "shedCapacity": 2}
    events = [
        # 3 geese into a 2-slot shed: 2 committed, third aborts.
        (0, 0, act(market=[["BUY_ANIMAL", "GOOSE", 3]])),
        (0, 1, act(farmer=["PICKUP", "GOOSE"])),
        # Shed room 1 again: this buy succeeds.
        (0, 2, act(market=[["BUY_ANIMAL", "GOOSE", 1]])),
        # PLACE off-structure falls through to the shed path: zero room ->
        # silent no-op, the goose stays carried.
        (0, 3, act(farmer=["PLACE", "GOOSE"])),
    ]
    trace = build_turns(events, 2)
    result = replay(name, trace, config)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(config, trace)
    after_place = timeline[turn_index(0, 3)]
    assert after_place["privates"][0]["inventories"][0]["GOOSE"] == 1
    # End-of-day drop has zero shed room and DISCARDS the carried goose.
    final = timeline[-1]
    assert final["privates"][0]["shed"]["GOOSE"] == 2
    assert final["privates"][0]["inventories"][0]["GOOSE"] == 0
    assert final["farms"][0]["money"] == 2100.0


# ---------------------------------------------------------------------------
# PLACE: invalid targets fall through to the shed path or no-op
# ---------------------------------------------------------------------------


def test_place_invalid_targets_fall_through_to_shed_path() -> None:
    name = "place_invalid"
    events = [
        (0, 0, act(market=[["BUY_ANIMAL", "GOOSE", 1], ["BUY_ANIMAL", "COW", 1]])),
        (0, 1, act(farmer=["PICKUP", "GOOSE"])),
        # Empty non-structure tile at the shed-access spawn: shed path.
        (0, 2, act(farmer=["PLACE", "GOOSE"])),
        (0, 3, act(farmer=["PICKUP", "GOOSE"])),
        # Locked tile (4,5) IS shed-adjacent: shed path applies there too.
        (0, 4, act(farmer=["SOUTH"])),
        (0, 5, act(farmer=["PLACE", "GOOSE"])),
        (0, 6, act(farmer=["NORTH"])),
        # COOP at (4,3); a COW does not match a COOP: no-op off-shed, the
        # cow stays carried; a GOOSE matches but placement consumes exactly
        # the carried animal.
        (0, 7, act(farmer=["PICKUP", "COW"])),
        (0, 8, act(farmer=["NORTH"])),
        (0, 9, act(farmer=["BUILD_COOP"])),
        (0, 10, act(farmer=["PLACE", "COW"])),
        (0, 11, act(farmer=["SOUTH"])),
        (0, 12, act(farmer=["PLACE", "COW"])),
        (0, 13, act(farmer=["PICKUP", "GOOSE"])),
        (0, 14, act(farmer=["NORTH"])),
        # The goose DOES match the coop: this placement succeeds and
        # consumes exactly the one carried animal.
        (0, 15, act(farmer=["PLACE", "GOOSE"])),
    ]
    trace = build_turns(events, 1)
    result = replay(name, trace)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(CONFIG, trace)
    after_cow_place = timeline[turn_index(0, 10)]
    # Structure mismatch + not shed-adjacent: silent no-op, cow still carried.
    assert after_cow_place["privates"][0]["inventories"][0]["COW"] == 1
    assert "animal" not in tile(after_cow_place, 4, 3)
    final = timeline[-1]
    # The cow ended in the shed; the goose ended placed on its coop.
    assert final["privates"][0]["shed"]["GOOSE"] == 0
    assert final["privates"][0]["shed"]["COW"] == 1
    goose_tile = tile(final, 4, 3)
    assert goose_tile["kind"] == "COOP" and goose_tile["animal"] == "GOOSE"
    assert goose_tile["placed_day"] == 0
    assert final["privates"][0]["inventories"][0] == {
        "WHEAT": 0, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0,
        "EGG": 0, "MILK": 0, "WOOL": 0, "FERTILIZER": 0,
        "GOOSE": 0, "COW": 0, "SHEEP": 0,
    }
    # Animals were paid at the official table costs: 3000 - GOOSE 300 - COW 400.
    assert final["farms"][0]["money"] == 2300.0


# ---------------------------------------------------------------------------
# FEED: requires carried WHEAT, once per day
# ---------------------------------------------------------------------------


def test_feed_requires_wheat_and_is_once_per_day() -> None:
    name = "feed_rules"
    events = [
        (0, 0, act(market=[["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 2]])),
        (0, 1, act(farmer=["PICKUP", "GOOSE"])),
        (0, 2, act(farmer=["WEST"])),
        (0, 3, act(farmer=["NORTH"])),
        (0, 4, act(farmer=["BUILD_COOP"])),
        (0, 5, act(farmer=["PLACE", "GOOSE"])),
        (0, 6, act(farmer=["EAST"])),
        (0, 7, act(farmer=["SOUTH"])),
        # Day 1: FEED without carried wheat is a no-op...
        (1, 1, act(farmer=["NORTH"])),
        (1, 2, act(farmer=["WEST"])),
        (1, 3, act(farmer=["FEED"])),
        (1, 4, act(farmer=["EAST"])),
        (1, 5, act(farmer=["SOUTH"])),
        (1, 6, act(farmer=["PICKUP", "WHEAT", 2])),
        (1, 7, act(farmer=["NORTH"])),
        (1, 8, act(farmer=["WEST"])),
        # ...the first FEED consumes 1 wheat and sets fed_today...
        (1, 9, act(farmer=["FEED"])),
        # ...the second FEED is a no-op (already fed today; wheat kept).
        (1, 10, act(farmer=["FEED"])),
        (1, 11, act(farmer=["EAST"])),
        (1, 12, act(farmer=["SOUTH"])),
        # Day 2: normal feed keeps the animal alive.
        (2, 0, act(farmer=["PICKUP", "WHEAT", 1])),
        (2, 1, act(farmer=["NORTH"])),
        (2, 2, act(farmer=["WEST"])),
        (2, 3, act(farmer=["FEED"])),
        (2, 4, act(farmer=["EAST"])),
        (2, 5, act(farmer=["SOUTH"])),
    ]
    trace = build_turns(events, 3)
    result = replay(name, trace)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(CONFIG, trace)
    after_no_wheat_feed = timeline[turn_index(1, 3)]
    assert tile(after_no_wheat_feed, 3, 3)["fed_today"] is False
    after_first_feed = timeline[turn_index(1, 9)]
    assert tile(after_first_feed, 3, 3)["fed_today"] is True
    assert after_first_feed["privates"][0]["inventories"][0]["WHEAT"] == 1
    after_second_feed = timeline[turn_index(1, 10)]
    assert tile(after_second_feed, 3, 3)["fed_today"] is True
    assert after_second_feed["privates"][0]["inventories"][0]["WHEAT"] == 1
    # Unfed placement day leaves streak 1 entering day 1; feeding stops it.
    entering_day1 = timeline[turn_index(0, 23)]
    assert tile(entering_day1, 3, 3)["consecutive_unfed"] == 1
    final = timeline[-1]
    assert tile(final, 3, 3)["consecutive_unfed"] == 0
    assert "animal" in tile(final, 3, 3)


# ---------------------------------------------------------------------------
# Escape timing, structure persistence, DIG semantics
# ---------------------------------------------------------------------------


def test_escape_timing_structure_remains_and_dig_semantics() -> None:
    name = "escape_and_dig"
    events = [
        (0, 0, act(market=[["BUY_ANIMAL", "GOOSE", 2], ["BUY_PRODUCT", "WHEAT", 4]])),
        (0, 1, act(farmer=["PICKUP", "GOOSE"])),
        (0, 2, act(farmer=["WEST"])),
        (0, 3, act(farmer=["NORTH"])),
        (0, 4, act(farmer=["BUILD_COOP"])),
        (0, 5, act(farmer=["PLACE", "GOOSE"])),
        # Goose #1 is never fed: streak 1 after day 0, 2 after day 1 ->
        # escapes at the END of day 1; the bare COOP remains.
        # Day 2: DIG the empty coop (removes it), rebuild, place goose #2,
        # then DIG the placed animal (silent no-op).
        (2, 1, act(farmer=["NORTH"])),
        (2, 2, act(farmer=["WEST"])),
        (2, 3, act(farmer=["DIG"])),
        (2, 4, act(farmer=["BUILD_COOP"])),
        (2, 5, act(farmer=["EAST"])),
        (2, 6, act(farmer=["SOUTH"])),
        (2, 7, act(farmer=["PICKUP", "GOOSE"])),
        (2, 8, act(farmer=["NORTH"])),
        (2, 9, act(farmer=["WEST"])),
        (2, 10, act(farmer=["PLACE", "GOOSE"])),
        (2, 11, act(farmer=["DIG"])),
        (2, 12, act(farmer=["EAST"])),
        (2, 13, act(farmer=["SOUTH"])),
        # Day 3: feed goose #2 (resets its streak after the day-2 unfed day).
        (3, 0, act(farmer=["PICKUP", "WHEAT", 1])),
        (3, 1, act(farmer=["NORTH"])),
        (3, 2, act(farmer=["WEST"])),
        (3, 3, act(farmer=["FEED"])),
        (3, 4, act(farmer=["EAST"])),
        (3, 5, act(farmer=["SOUTH"])),
        # Days 4-5 unfed: streak 1 after day 4, 2 after day 5 -> escapes at
        # the END of day 5.
    ]
    trace = build_turns(events, 6)
    result = replay(name, trace)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(CONFIG, trace)
    # Exact first escape timing: still placed through day 1's hours...
    before_escape = timeline[turn_index(1, 22)]
    assert tile(before_escape, 3, 3)["animal"] == "GOOSE"
    assert tile(before_escape, 3, 3)["consecutive_unfed"] == 1
    # ...gone at the day-1 boundary, structure left behind.
    after_escape = timeline[turn_index(1, 23)]
    assert tile(after_escape, 3, 3) == {"kind": "COOP"}
    # DIG removed the empty structure entirely.
    after_dig = timeline[turn_index(2, 3)]
    assert tile(after_dig, 3, 3) is None
    # DIG on the placed animal is a silent no-op.
    after_animal_dig = timeline[turn_index(2, 11)]
    assert tile(after_animal_dig, 3, 3)["animal"] == "GOOSE"
    assert tile(after_animal_dig, 3, 3)["kind"] == "COOP"
    # Second escape exactly at the end of day 5, structure remains again.
    before_second = timeline[turn_index(5, 22)]
    assert tile(before_second, 3, 3)["animal"] == "GOOSE"
    assert tile(before_second, 3, 3)["consecutive_unfed"] == 1
    final = timeline[-1]
    assert tile(final, 3, 3) == {"kind": "COOP"}
    assert final["privates"][0]["shed"]["GOOSE"] == 0


# ---------------------------------------------------------------------------
# COLLECT_FERTILIZER: availability, once per day, regeneration
# ---------------------------------------------------------------------------


def test_collect_fertilizer_availability_once_per_day_and_regeneration() -> None:
    name = "fertilizer_rules"
    events = [
        (0, 0, act(market=[["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 4]])),
        (0, 1, act(farmer=["PICKUP", "GOOSE"])),
        (0, 2, act(farmer=["WEST"])),
        (0, 3, act(farmer=["NORTH"])),
        (0, 4, act(farmer=["BUILD_COOP"])),
        (0, 5, act(farmer=["PLACE", "GOOSE"])),
        # Placement day: fertilizer_available is False until the first
        # day-end refresh; COLLECT_FERTILIZER is a no-op.
        (0, 6, act(farmer=["COLLECT_FERTILIZER"])),
        # Day 1: regenerate flag set by the day-0 refresh; first collect
        # succeeds, second same-day collect is a no-op. The goose is fed so
        # it survives to see the day-1 regeneration.
        (1, 0, act(farmer=["PICKUP", "WHEAT", 1])),
        (1, 1, act(farmer=["NORTH"])),
        (1, 2, act(farmer=["WEST"])),
        (1, 3, act(farmer=["FEED"])),
        (1, 4, act(farmer=["COLLECT_FERTILIZER"])),
        (1, 5, act(farmer=["COLLECT_FERTILIZER"])),
        (1, 6, act(farmer=["EAST"])),
        (1, 7, act(farmer=["SOUTH"])),
        # Day 2: the flag regenerated at the day-1 refresh.
        (2, 0, act(farmer=["PICKUP", "WHEAT", 1])),
        (2, 1, act(farmer=["NORTH"])),
        (2, 2, act(farmer=["WEST"])),
        (2, 3, act(farmer=["FEED"])),
        (2, 4, act(farmer=["COLLECT_FERTILIZER"])),
        (2, 5, act(farmer=["EAST"])),
        (2, 6, act(farmer=["SOUTH"])),
    ]
    trace = build_turns(events, 3)
    result = replay(name, trace)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(CONFIG, trace)
    after_placement_collect = timeline[turn_index(0, 6)]
    assert tile(after_placement_collect, 3, 3)["fertilizer_available"] is False
    assert after_placement_collect["privates"][0]["inventories"][0]["FERTILIZER"] == 0
    after_first_collect = timeline[turn_index(1, 4)]
    assert tile(after_first_collect, 3, 3)["fertilizer_available"] is False
    assert after_first_collect["privates"][0]["inventories"][0]["FERTILIZER"] == 1
    after_second_collect = timeline[turn_index(1, 5)]
    assert after_second_collect["privates"][0]["inventories"][0]["FERTILIZER"] == 1
    final = timeline[-1]
    # Two successful collects, dropped to the shed at each day-end.
    assert final["privates"][0]["shed"]["FERTILIZER"] == 2


# ---------------------------------------------------------------------------
# Hired hand performs the whole animal chore with its own inventory
# ---------------------------------------------------------------------------


def test_hand_places_feeds_and_collects_with_own_inventory() -> None:
    name = "hand_chore"
    events = [
        # Day 0: the farmer buys, builds the coop at (3,3), and places.
        (0, 0, act(market=[["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 6]])),
        (0, 1, act(farmer=["WEST"])),
        (0, 2, act(farmer=["NORTH"])),
        (0, 3, act(farmer=["BUILD_COOP"])),
        (0, 4, act(farmer=["EAST"])),
        (0, 5, act(farmer=["SOUTH"])),
        (0, 6, act(farmer=["PICKUP", "GOOSE"])),
        (0, 7, act(farmer=["WEST"])),
        (0, 8, act(farmer=["NORTH"])),
        (0, 9, act(farmer=["PLACE", "GOOSE"])),
        (0, 10, act(farmer=["EAST"])),
        (0, 11, act(farmer=["SOUTH"])),
        # Hands never survive a day boundary (`_end_of_day` clears
        # ``farm["hands"]`` and resets ``private["inventories"]``), so the
        # chore hand is hired on day 1. It spawns at the first free
        # shed-access tile (5,4) -- the farmer occupies (4,4).
        (1, 0, act(market=[["HIRE"]])),
        (1, 1, act(hands=[["PICKUP", "WHEAT", 1]])),
        (1, 2, act(hands=[["WEST"]])),
        (1, 3, act(hands=[["WEST"]])),
        (1, 4, act(hands=[["NORTH"]])),
        (1, 5, act(hands=[["FEED"]])),
        (1, 6, act(hands=[["CARE"]])),
        (1, 7, act(hands=[["COLLECT_FERTILIZER"]])),
    ]
    trace = build_turns(events, 2)
    result = replay(name, trace)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(CONFIG, trace)
    after_hire = timeline[turn_index(1, 0)]
    assert after_hire["farms"][0]["hands"] == [[5, 4]]
    after_place = timeline[turn_index(0, 9)]
    assert tile(after_place, 3, 3)["animal"] == "GOOSE"
    after_hand_feed = timeline[turn_index(1, 5)]
    assert tile(after_hand_feed, 3, 3)["fed_today"] is True
    # Still inside day 1: the fertilizer the hand collected sits in the
    # HAND's inventory (index 1) and the goose is cared today.
    after_hand_collect = timeline[turn_index(1, 7)]
    assert after_hand_collect["privates"][0]["inventories"][1]["FERTILIZER"] == 1
    assert tile(after_hand_collect, 3, 3)["cared_today"] is True
    assert tile(after_hand_collect, 3, 3)["fertilizer_available"] is False
    # The day-1 boundary clears the hands and drops their carry into the shed.
    final = timeline[-1]
    assert final["privates"][0]["shed"]["FERTILIZER"] == 1
    assert len(final["privates"][0]["inventories"]) == 1


# ---------------------------------------------------------------------------
# End-of-day inventory drop: insertion-order priority with animals carried
# ---------------------------------------------------------------------------


def test_day_end_drop_insertion_order_priority_with_animals() -> None:
    name = "day_end_drop_priority"
    config = {"seed": 7, "shedCapacity": 6}
    base_events = [
        # Shed budget (capacity 6): 4 geese + 2 wheat fill it exactly; the
        # wheat top-up orders below each partially fill (room-bound).
        (0, 0, act(market=[
            ["BUY_ANIMAL", "GOOSE", 4],
            ["BUY_PRODUCT", "WHEAT", 2],
        ])),
        (0, 1, act(farmer=["PICKUP", "GOOSE"])),
        (0, 2, act(farmer=["WEST"])),
        (0, 3, act(farmer=["NORTH"])),
        (0, 4, act(farmer=["BUILD_COOP"])),
        (0, 5, act(farmer=["PLACE", "GOOSE"])),
        (0, 6, act(farmer=["EAST"])),
        (0, 7, act(farmer=["SOUTH"])),
        # Days 1-4: feed the goose (base production needs no feed, but the
        # animal must stay alive), collecting fertilizer only on day 4.
        (1, 0, act(farmer=["PICKUP", "WHEAT", 1])),
        (1, 1, act(farmer=["NORTH"])),
        (1, 2, act(farmer=["WEST"])),
        (1, 3, act(farmer=["FEED"])),
        (1, 4, act(farmer=["EAST"])),
        (1, 5, act(farmer=["SOUTH"])),
        (1, 6, act(market=[["BUY_PRODUCT", "WHEAT", 3]])),  # room-bound: 2 of 3
        (2, 0, act(farmer=["PICKUP", "WHEAT", 1])),
        (2, 1, act(farmer=["NORTH"])),
        (2, 2, act(farmer=["WEST"])),
        (2, 3, act(farmer=["FEED"])),
        (2, 4, act(farmer=["EAST"])),
        (2, 5, act(farmer=["SOUTH"])),
        (2, 6, act(market=[["BUY_PRODUCT", "WHEAT", 2]])),  # room-bound: 1 of 2
        (3, 0, act(farmer=["PICKUP", "WHEAT", 1])),
        (3, 1, act(farmer=["NORTH"])),
        (3, 2, act(farmer=["WEST"])),
        (3, 3, act(farmer=["FEED"])),
        (3, 4, act(farmer=["EAST"])),
        (3, 5, act(farmer=["SOUTH"])),
        (4, 0, act(farmer=["PICKUP", "WHEAT", 1])),
        (4, 1, act(farmer=["NORTH"])),
        (4, 2, act(farmer=["WEST"])),
        (4, 3, act(farmer=["FEED"])),
        (4, 4, act(farmer=["COLLECT_FERTILIZER"])),
        (4, 5, act(farmer=["EAST"])),
        (4, 6, act(farmer=["SOUTH"])),
    ]

    def variant(first_op: str, second_op: str, label: str):
        events = list(base_events) + [
            (5, 1, act(farmer=["NORTH"])),
            (5, 2, act(farmer=["WEST"])),
            (5, 3, act(farmer=[first_op])),
            (5, 4, act(farmer=[second_op])),
        ]
        trace = build_turns(events, 6)
        replay(f"{name}:{label}", trace, config)
        timeline = fast_timeline(config, trace)
        # Entering day 5 the shed holds 3 GEESE + 1 WHEAT + 1 FERTILIZER
        # (5/6 used): productions at the ends of days 3-4 give yield 2, so
        # the farmer walks out carrying 2 EGG + 1 FERTILIZER but only ONE
        # shed slot is free at the day-5 boundary.
        entering = timeline[turn_index(4, 23)]
        assert entering["privates"][0]["shed"] == {
            "WHEAT": 1, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0,
            "EGG": 0, "MILK": 0, "WOOL": 0, "FERTILIZER": 1,
            "GOOSE": 3, "COW": 0, "SHEEP": 0,
        }
        return timeline[-1]

    # Whichever item entered the inventory FIRST wins the free slot; the
    # rest is DISCARDED (_drop_inventories_to_shed insertion order).
    harvest_first = variant("HARVEST", "COLLECT_FERTILIZER", "harvest_first")
    assert harvest_first["privates"][0]["shed"]["EGG"] == 1
    assert harvest_first["privates"][0]["shed"]["FERTILIZER"] == 1
    assert harvest_first["privates"][0]["inventories"][0]["EGG"] == 0
    collect_first = variant("COLLECT_FERTILIZER", "HARVEST", "collect_first")
    assert collect_first["privates"][0]["shed"]["FERTILIZER"] == 2
    # The fertilizer took the only free slot; BOTH eggs were discarded.
    assert collect_first["privates"][0]["shed"]["EGG"] == 0
    assert collect_first["privates"][0]["inventories"][0]["FERTILIZER"] == 0


# ---------------------------------------------------------------------------
# Structure build requirements: empty owned tile only, builds are free
# ---------------------------------------------------------------------------


def test_build_coop_pasture_blocking_rules_and_free_cost() -> None:
    name = "build_rules"
    events = [
        (0, 0, act(market=[["BUY_SEED", "WHEAT", 1]])),
        (0, 1, act(farmer=["NORTH"])),  # (4,3)
        (0, 2, act(farmer=["PLANT", "WHEAT"])),
        # Building on a plant tile is blocked (both kinds)...
        (0, 3, act(farmer=["BUILD_COOP"])),
        (0, 4, act(farmer=["BUILD_PASTURE"])),
        (0, 5, act(farmer=["WEST"])),  # (3,3)
        (0, 6, act(farmer=["BUILD_COOP"])),
        # ...building on an existing structure is blocked...
        (0, 7, act(farmer=["BUILD_PASTURE"])),
        (0, 8, act(farmer=["EAST"])),
        (0, 9, act(farmer=["SOUTH"])),  # (4,4)
        # ...and building on a LOCKED tile is blocked.
        (0, 10, act(farmer=["SOUTH"])),  # (4,5) locked
        (0, 11, act(farmer=["BUILD_COOP"])),
        (0, 12, act(farmer=["NORTH"])),
    ]
    trace = build_turns(events, 2)
    result = replay(name, trace)
    assert result.turns_executed == len(trace)

    timeline = fast_timeline(CONFIG, trace)
    after_plant = timeline[turn_index(0, 2)]
    assert after_plant["farms"][0]["tiles"][3][4]["kind"] == "PLANT"
    after_coop = timeline[turn_index(0, 6)]
    assert tile(after_coop, 3, 3) == {"kind": "COOP"}
    final = timeline[-1]
    assert tile(final, 3, 3) == {"kind": "COOP"}
    # The unwatered wheat plant died at the day-0 boundary (official
    # `_daily_refresh_plants`: consecutive_unwatered >= 2 -> WEED); it was
    # never built over.
    assert tile(final, 4, 3) == {"kind": "WEED"}
    assert tile(final, 4, 5) == "LOCKED"  # never built on
    # Builds are FREE: only the seed purchase (10) reduced the money.
    assert final["farms"][0]["money"] == 2990.0
