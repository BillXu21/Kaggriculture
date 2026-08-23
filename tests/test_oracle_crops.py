"""Stage-2b focused crop/seed/tile lifecycle parity probes against the pinned
official 1.32.7 engine.

Every scenario replays the exact same action pair through both engines and
compares the complete canonical state after every submitted pair
(``run_same_action_replay`` raises at the FIRST divergent field). Semantic
assertions pin the official mechanic itself (values taken from the pinned
source symbols cited inline) so a jointly-wrong fast engine cannot pass
silently.

Official source of truth (kaggle-environments 1.32.7, commit
28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c,
``kaggle_environments/envs/kaggriculture/kaggriculture.py``):

- ``CROPS`` constant table (seed/first_yield_day/max_yield_day/interval/
  max_yield/ongoing per crop);
- ``_new_plant`` (fresh tile dict: ``consecutive_unwatered=1`` -- planting day
  counts as unwatered, ``yield_units`` 1 non-ongoing / 0 ongoing,
  ``max_lifespan_step`` ``-1`` ongoing else ``(day+max_yield_day+1)*24``,
  ``fertilized_until_day=-1``, ``watered_today=False``);
- ``_apply_unit_action`` arms ``PLANT`` (tile must be empty unlocked, seed
  consumed only on success), ``WATER`` (once per day; single-harvest bonus
  ``window_start=(max_yield_day+1)//2 .. max_yield_day``, bonus 2 when
  ``fertilized_until_day >= day`` else 1), ``HARVEST`` (silent no-op before
  ``first_yield_day`` even with positive yield; drains all units; non-ongoing
  crops are removed from the tile), ``FERTILIZE`` (consumes 1 carried
  fertilizer, ``fertilized_until_day = max(old, day+2)`` -- active on the
  application day and the following two), ``DIG`` (clears plants/weeds/
  structures but never a placed animal);
- ``_process_actions`` atomic PLANT validation: ``plant_demand`` counts PLANT
  requests across the farmer AND every submitted hands entry (including
  entries beyond the hired hand count); when demand for a crop exceeds owned
  seeds, ALL of that crop's PLANT requests become PASS;
- ``_daily_refresh_plants`` (end-of-day: unwatered streak increment, >=2
  converts to WEED; ongoing interval accrual with fertilizer doubling only on
  watered days; ``max_lifespan_step`` set when production count reaches
  ``max_yield``);
- ``_decay_plants`` (per-step: at ``step >= max_lifespan_step`` with
  ``(step-mls)%2==0``, decrement yield unconditionally and convert to WEED
  when the result is <= 0).

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

# Literal transcription of the pinned official ``CROPS`` table (see module
# docstring). Locked against fast-engine behavior by the probes below.
OFFICIAL_CROPS_TABLE = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
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


def water_daily(days: int, hour: int = 0, start: int = 1):
    return [(day, hour, act(farmer=["WATER"])) for day in range(start, days)]


# ---------------------------------------------------------------------------
# Constants table
# ---------------------------------------------------------------------------


def test_official_crop_constants_table_is_locked() -> None:
    # The probes below derive every expectation from this literal table; it
    # must keep matching the pinned official source symbol ``CROPS``.
    assert OFFICIAL_CROPS_TABLE["WHEAT"]["interval"] == 0
    assert sum(entry["seed"] for entry in OFFICIAL_CROPS_TABLE.values()) == 260


def test_all_crop_seed_costs_against_official_constants() -> None:
    # One BUY_SEED 2 order per crop, one per hour; the money delta per hour
    # must equal 2x the official seed cost (fast-only semantic pin; the oracle
    # replay proves the fast engine agrees with the official engine).
    events = [
        (0, hour, act(market=[["BUY_SEED", crop, 2]]))
        for hour, crop in enumerate(OFFICIAL_CROPS_TABLE)
    ]
    trace = build_turns(events, total_days=1)
    replay("all_crop_seed_costs", trace)
    timeline = fast_timeline(CONFIG, trace)
    money = 3000.0
    for hour, (crop, data) in enumerate(OFFICIAL_CROPS_TABLE.items()):
        money -= 2 * data["seed"]
        assert timeline[hour + 1]["farms"][0]["money"] == money, crop
        assert timeline[-1]["privates"][0]["seeds"][crop] == 2


# ---------------------------------------------------------------------------
# PLANT
# ---------------------------------------------------------------------------


def test_plant_atomic_seed_consumption_and_new_plant_fields() -> None:
    # _new_plant: consecutive_unwatered=1 (planting day counts as unwatered),
    # yield_units=1 for non-ongoing, mls=(0+4+1)*24=120, fertilized=-1.
    events = [
        (0, 0, act(market=[["BUY_SEED", "WHEAT", 3]])),
        (0, 1, act(farmer=["PLANT", "WHEAT"])),
        (0, 2, act(farmer=["PLANT", "WHEAT"])),  # occupied tile: silent no-op
    ]
    trace = build_turns(events, total_days=1)
    replay("plant_atomic_consumption", trace)
    states = fast_timeline(CONFIG, trace)
    assert states[turn_index(0, 1)]["privates"][0]["seeds"]["WHEAT"] == 2
    assert tile(states[turn_index(0, 1)], 4, 4) == {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "max_lifespan_step": 120, "yield_units": 1, "watered_today": False,
        "consecutive_unwatered": 1, "fertilized_until_day": -1,
    }
    # Second PLANT onto the occupied tile consumed nothing.
    assert states[turn_index(0, 2)]["privates"][0]["seeds"]["WHEAT"] == 2
    assert tile(states[turn_index(0, 2)], 4, 4)["planted_day"] == 0


def test_plant_invalid_cases_are_silent_noops() -> None:
    # No seeds / unknown crop / locked tile all no-op (_apply_unit_action
    # PLANT arm guards); the locked-tile plant leaves "LOCKED" verbatim.
    events = [
        (0, 0, act(farmer=["PLANT", "WHEAT"])),   # no seeds
        (0, 1, act(farmer=["PLANT", "NOPE"])),    # unknown crop
        (0, 2, act(farmer=["SOUTH"])),            # stand on locked (4,5)
        (0, 3, act(farmer=["PLANT", "WHEAT"])),   # locked tile: no-op
        (0, 4, act(farmer=["NORTH"])),
        (0, 5, act(market=[["BUY_SEED", "WHEAT", 1]])),
        (0, 6, act(farmer=["PLANT", "WHEAT"])),   # now succeeds
    ]
    trace = build_turns(events, total_days=1)
    replay("plant_invalid_noops", trace)
    states = fast_timeline(CONFIG, trace)
    plant_turn = turn_index(0, 6)
    for index in range(len(states)):
        assert tile(states[index], 4, 5) == "LOCKED"
        assert (tile(states[index], 4, 4) is None) == (index < plant_turn)
    assert states[turn_index(0, 6)]["privates"][0]["seeds"]["WHEAT"] == 0
    assert tile(states[turn_index(0, 6)], 4, 4)["crop"] == "WHEAT"


def test_atomic_plant_group_blocks_on_short_supply_then_succeeds_at_supply() -> None:
    # _process_actions: demand counts farmer + submitted hands entries; 2
    # requests vs 1 seed blocks BOTH; 2 vs 2 lets both commit atomically.
    events = [
        (0, 0, act(market=[["HIRE"], ["BUY_SEED", "WHEAT", 1]])),
        # Hand spawned at (5,4); its PLANT request counts even though the
        # tile is locked, so demand 2 > 1 seed blocks BOTH requests.
        (0, 1, act(farmer=["PLANT", "WHEAT"], hands=[["PLANT", "WHEAT"]])),
        (0, 2, act(market=[["BUY_SEED", "WHEAT", 1]], hands=[["WEST"]])),
        (0, 3, act(hands=[["WEST"]])),  # hand to (3,4)
        (0, 4, act(farmer=["PLANT", "WHEAT"], hands=[["PLANT", "WHEAT"]])),
    ]
    trace = build_turns(events, total_days=1)
    replay("atomic_plant_group", trace)
    states = fast_timeline(CONFIG, trace)
    blocked = states[turn_index(0, 1)]
    assert blocked["privates"][0]["seeds"]["WHEAT"] == 1
    assert tile(blocked, 4, 4) is None and tile(blocked, 3, 4) is None
    committed = states[turn_index(0, 4)]
    assert committed["privates"][0]["seeds"]["WHEAT"] == 0
    assert tile(committed, 4, 4)["crop"] == "WHEAT"
    assert tile(committed, 3, 4)["crop"] == "WHEAT"


def test_phantom_hand_plant_request_counts_toward_demand() -> None:
    # A PLANT request in the hands list beyond the hired hand count never
    # executes but STILL counts toward plant_demand (official sums
    # [farmer_action, *hands_actions] before any existence check).
    events = [
        (0, 0, act(market=[["BUY_SEED", "WHEAT", 1]])),
        (0, 1, act(farmer=["PLANT", "WHEAT"], hands=[["PLANT", "WHEAT"]])),
        (0, 2, act(farmer=["PLANT", "WHEAT"])),
    ]
    trace = build_turns(events, total_days=1)
    replay("phantom_hand_demand", trace)
    states = fast_timeline(CONFIG, trace)
    blocked = states[turn_index(0, 1)]
    assert blocked["privates"][0]["seeds"]["WHEAT"] == 1
    assert tile(blocked, 4, 4) is None
    committed = states[turn_index(0, 2)]
    assert committed["privates"][0]["seeds"]["WHEAT"] == 0
    assert tile(committed, 4, 4)["crop"] == "WHEAT"


# ---------------------------------------------------------------------------
# WATER
# ---------------------------------------------------------------------------


def test_water_once_per_day_and_wheat_yield_window() -> None:
    # WHEAT window: window_start=(4+1)//2=2 .. max_yield_day=4; bonus 1
    # unfertilized; double WATER same day is a no-op; unwatered days still
    # advance consecutive_unwatered without killing a watered crop.
    events = [
        (0, 0, act(market=[["BUY_SEED", "WHEAT", 1]])),
        (0, 1, act(farmer=["PLANT", "WHEAT"])),
        (0, 2, act(farmer=["WATER"])),
        (0, 3, act(farmer=["WATER"])),  # already watered today
        (2, 0, act(farmer=["WATER"])),
        (3, 0, act(farmer=["WATER"])),
        (4, 0, act(farmer=["WATER"])),
    ]
    trace = build_turns(events, total_days=5)
    replay("water_window_wheat", trace)
    states = fast_timeline(CONFIG, trace)
    after_first = tile(states[turn_index(0, 2)], 4, 4)
    assert after_first["watered_today"] is True and after_first["yield_units"] == 1
    # Age 0 below window_start: watering alone grants no yield.
    assert tile(states[turn_index(0, 3)], 4, 4)["yield_units"] == 1
    # End of day 1 (unwatered): streak 1, watered flag cleared.
    overnight = tile(states[turn_index(1, 23)], 4, 4)
    assert overnight["consecutive_unwatered"] == 1 and overnight["watered_today"] is False
    assert tile(states[turn_index(2, 0)], 4, 4)["yield_units"] == 2
    assert tile(states[turn_index(3, 0)], 4, 4)["yield_units"] == 3
    assert tile(states[turn_index(4, 0)], 4, 4)["yield_units"] == 4


# ---------------------------------------------------------------------------
# FERTILIZE
# ---------------------------------------------------------------------------


def test_fertilizer_active_application_day_plus_two_expiry_on_melon() -> None:
    # MELON window 6..12 lets the +2/+1 boundary be observed directly:
    # fertilize day 6 (active 6,7,8), watering day 9 gets only +1. Carried
    # inventories drop to the shed at each day end, so the fertilizer is
    # picked up the same morning it is applied.
    events = [
        (0, 0, act(market=[["BUY_SEED", "MELON", 1], ["BUY_PRODUCT", "FERTILIZER", 1]])),
        (0, 2, act(farmer=["PLANT", "MELON"])),
        (0, 3, act(farmer=["WATER"])),
        *water_daily(6),
        (6, 0, act(farmer=["PICKUP", "FERTILIZER", 1])),
        (6, 1, act(farmer=["FERTILIZE"])),
        (6, 2, act(farmer=["WATER"])),
        (7, 0, act(farmer=["WATER"])),
        (8, 0, act(farmer=["WATER"])),
        (8, 1, act(farmer=["HARVEST"])),  # age 8 < first_yield_day 10: guarded
        (9, 0, act(farmer=["WATER"])),    # fertilizer expired (8 < 9): +1
        (10, 0, act(farmer=["HARVEST"])),
    ]
    trace = build_turns(events, total_days=11)
    replay("fertilizer_expiry_melon", trace)
    states = fast_timeline(CONFIG, trace)
    day6 = tile(states[turn_index(6, 2)], 4, 4)
    assert day6["fertilized_until_day"] == 8 and day6["yield_units"] == 3  # 1+2
    assert tile(states[turn_index(7, 0)], 4, 4)["yield_units"] == 5       # +2
    assert tile(states[turn_index(8, 0)], 4, 4)["yield_units"] == 6       # +2 cap
    # Immature HARVEST guard: positive yield but age 8 < 10 must not collect.
    guarded = states[turn_index(8, 1)]
    assert tile(guarded, 4, 4)["yield_units"] == 6
    assert guarded["privates"][0]["inventories"][0]["MELON"] == 0
    assert tile(states[turn_index(9, 0)], 4, 4)["yield_units"] == 6       # +1 cap
    harvested = states[turn_index(10, 0)]
    assert harvested["privates"][0]["inventories"][0]["MELON"] == 6
    assert tile(harvested, 4, 4) is None  # non-recurring crop removed


def test_fertilizer_requires_carried_inventory_and_caps_wheat_yield() -> None:
    # WHEAT with fertilizer on every window day: 1 +2+2+2 = 7 capped at
    # max_yield 6 (_apply_unit_action WATER arm min()); FERTILIZE without a
    # carried fertilizer is a silent no-op (covered further in the guards
    # scenario); harvest collects exactly the capped 6.
    events = [
        (0, 0, act(market=[["BUY_SEED", "WHEAT", 1], ["BUY_PRODUCT", "FERTILIZER", 1]])),
        (0, 2, act(farmer=["PLANT", "WHEAT"])),
        (0, 3, act(farmer=["WATER"])),
        (1, 0, act(farmer=["WATER"])),
        (2, 0, act(farmer=["PICKUP", "FERTILIZER", 1])),
        (2, 1, act(farmer=["FERTILIZE"])),  # active days 2,3,4
        (2, 2, act(farmer=["WATER"])),
        (3, 0, act(farmer=["WATER"])),
        (4, 0, act(farmer=["WATER"])),
        (4, 1, act(farmer=["HARVEST"])),
    ]
    trace = build_turns(events, total_days=5)
    replay("fertilizer_cap_wheat", trace)
    states = fast_timeline(CONFIG, trace)
    assert tile(states[turn_index(2, 2)], 4, 4)["yield_units"] == 3
    assert tile(states[turn_index(3, 0)], 4, 4)["yield_units"] == 5
    assert tile(states[turn_index(4, 0)], 4, 4)["yield_units"] == 6  # capped
    harvested = states[turn_index(4, 1)]
    assert harvested["privates"][0]["inventories"][0]["WHEAT"] == 6
    assert tile(harvested, 4, 4) is None


# ---------------------------------------------------------------------------
# Non-recurring removal, replant, stale-field reset
# ---------------------------------------------------------------------------


def test_carrot_cycle_replant_resets_stale_lifecycle_fields() -> None:
    # CARROT first_yield 2 / max 3: fertilized first cycle yields 3 at day 2;
    # the same-day replant must start from a brand-new _new_plant dict
    # (fertilized_until_day -1, watered_today False) -- not inherit the
    # fertilized/watered state of the harvested crop.
    events = [
        (0, 0, act(market=[["BUY_SEED", "CARROT", 2], ["BUY_PRODUCT", "FERTILIZER", 1]])),
        (0, 1, act(farmer=["PICKUP", "FERTILIZER", 1])),
        (0, 2, act(farmer=["PLANT", "CARROT"])),
        (0, 3, act(farmer=["FERTILIZE"])),
        (0, 4, act(farmer=["WATER"])),
        (1, 0, act(farmer=["WATER"])),
        (2, 0, act(farmer=["WATER"])),
        (2, 1, act(farmer=["HARVEST"])),
        (2, 2, act(farmer=["PLANT", "CARROT"])),
        (2, 3, act(farmer=["WATER"])),  # replant needs same-day water to survive
        (3, 0, act(farmer=["WATER"])),
        (4, 0, act(farmer=["WATER"])),
        (5, 0, act(farmer=["WATER"])),
        (5, 1, act(farmer=["HARVEST"])),
    ]
    trace = build_turns(events, total_days=6)
    replay("carrot_replant_reset", trace)
    states = fast_timeline(CONFIG, trace)
    assert states[turn_index(2, 1)]["privates"][0]["inventories"][0]["CARROT"] == 3
    # End of day 2: the carried harvest drops into the shed.
    dropped = states[turn_index(2, 23)]
    assert dropped["privates"][0]["shed"]["CARROT"] == 3
    assert dropped["privates"][0]["inventories"][0]["CARROT"] == 0
    replanted = tile(states[turn_index(2, 2)], 4, 4)
    assert replanted == {
        "kind": "PLANT", "crop": "CARROT", "planted_day": 2,
        "max_lifespan_step": (2 + 3 + 1) * 24, "yield_units": 1,
        "watered_today": False, "consecutive_unwatered": 1,
        "fertilized_until_day": -1,
    }
    assert states[turn_index(5, 1)]["privates"][0]["inventories"][0]["CARROT"] == 3
    assert states[turn_index(5, 1)]["privates"][0]["shed"]["CARROT"] == 3
    assert tile(states[turn_index(5, 1)], 4, 4) is None


# ---------------------------------------------------------------------------
# Unwatered decay to WEED and DIG
# ---------------------------------------------------------------------------


def test_unwatered_decay_to_weed_and_dig_recovery() -> None:
    # _daily_refresh_plants: planting day counts as unwatered, so a never-
    # watered plant becomes WEED after ONE end-of-day refresh; watering on the
    # planting day resets the streak, then two full unwatered days are needed.
    events = [
        (0, 0, act(market=[["BUY_SEED", "WHEAT", 2]])),
        (0, 1, act(farmer=["PLANT", "WHEAT"])),
        (1, 0, pas()),                    # weed by now (streak hit 2)
        (1, 1, act(farmer=["DIG"])),      # clear the weed
        (1, 2, act(farmer=["PLANT", "WHEAT"])),
        (1, 3, act(farmer=["WATER"])),    # streak reset on planting day
        # days 2 and 3 unwatered -> streak 2 at end of day 3 -> WEED
        (4, 0, pas()),
    ]
    trace = build_turns(events, total_days=5)
    replay("unwatered_weed_and_dig", trace)
    states = fast_timeline(CONFIG, trace)
    assert tile(states[turn_index(1, 0)], 4, 4) == {"kind": "WEED"}
    assert tile(states[turn_index(1, 1)], 4, 4) is None
    replanted = tile(states[turn_index(1, 3)], 4, 4)
    # WATER does not touch the streak itself; it only sets watered_today so
    # the END-OF-DAY refresh resets consecutive_unwatered to 0.
    assert replanted["consecutive_unwatered"] == 1 and replanted["watered_today"] is True
    assert tile(states[turn_index(1, 23)], 4, 4)["consecutive_unwatered"] == 0
    assert tile(states[turn_index(2, 23)], 4, 4)["consecutive_unwatered"] == 1
    assert tile(states[turn_index(4, 0)], 4, 4) == {"kind": "WEED"}


# ---------------------------------------------------------------------------
# Mature non-recurring per-step decay
# ---------------------------------------------------------------------------


def test_mature_melon_decays_one_unit_every_two_steps_to_weed() -> None:
    # Fully watered MELON has mls=(0+12+1)*24=312 (_new_plant non-ongoing
    # formula) with yield 6; _decay_plants decrements at steps 312,314,...
    # and the 0 crossing converts the tile to WEED at day 13 hour 10.
    events = [
        (0, 0, act(market=[["BUY_SEED", "MELON", 1]])),
        (0, 1, act(farmer=["PLANT", "MELON"])),
        (0, 2, act(farmer=["WATER"])),
        *water_daily(13),
    ]
    trace = build_turns(events, total_days=15)
    replay("mature_melon_decay", trace)
    states = fast_timeline(CONFIG, trace)
    assert tile(states[turn_index(12, 23)], 4, 4)["yield_units"] == 6
    assert tile(states[turn_index(13, 0)], 4, 4)["yield_units"] == 5
    assert tile(states[turn_index(13, 2)], 4, 4)["yield_units"] == 4
    assert tile(states[turn_index(13, 8)], 4, 4)["yield_units"] == 1
    assert tile(states[turn_index(13, 10)], 4, 4) == {"kind": "WEED"}


# ---------------------------------------------------------------------------
# Ongoing crops: survival, intervals, harvest-without-removal
# ---------------------------------------------------------------------------


def test_ongoing_tomato_daily_interval_harvest_survival_and_zero_yield_decay() -> None:
    # TOMATO (ongoing, interval 1, first 8, max 4): accrual ends day 7..10;
    # HARVEST drains yield but KEEPS the plant; once production completes
    # (mls=264 set at end of day 11) a zero-yield harvest-depleted plant
    # decays to WEED at step 264 like any other (_decay_plants decrements
    # unconditionally and converts at <= 0).
    events = [
        (0, 0, act(market=[["BUY_SEED", "TOMATO", 1]])),
        (0, 1, act(farmer=["PLANT", "TOMATO"])),
        (0, 2, act(farmer=["WATER"])),
        *water_daily(12),
        (8, 1, act(farmer=["HARVEST"])),
        (9, 1, act(farmer=["HARVEST"])),
        (10, 1, act(farmer=["HARVEST"])),
        (11, 1, act(farmer=["HARVEST"])),
        (12, 1, act(farmer=["DIG"])),
    ]
    trace = build_turns(events, total_days=13)
    replay("ongoing_tomato_lifecycle", trace)
    states = fast_timeline(CONFIG, trace)
    first = states[turn_index(8, 1)]
    assert first["privates"][0]["inventories"][0]["TOMATO"] == 1
    survived = tile(first, 4, 4)
    assert survived["kind"] == "PLANT" and survived["yield_units"] == 0
    # Each night drops the day's take into the shed; harvesting daily drains
    # each interval accrual immediately, so every take is 1 unit and the shed
    # accumulates the max_yield total of 4 by day 11.
    states_9 = states[turn_index(9, 1)]
    assert states_9["privates"][0]["inventories"][0]["TOMATO"] == 1
    assert states_9["privates"][0]["shed"]["TOMATO"] == 1
    states_10 = states[turn_index(10, 1)]
    assert states_10["privates"][0]["inventories"][0]["TOMATO"] == 1
    assert states_10["privates"][0]["shed"]["TOMATO"] == 2
    drained = states[turn_index(11, 1)]
    assert drained["privates"][0]["inventories"][0]["TOMATO"] == 1
    assert drained["privates"][0]["shed"]["TOMATO"] == 3
    assert tile(drained, 4, 4)["kind"] == "PLANT"
    # Zero-yield mature plant: decay to WEED at step 264, then dug clear.
    assert tile(states[turn_index(12, 0)], 4, 4) == {"kind": "WEED"}
    assert tile(states[turn_index(12, 1)], 4, 4) is None


def test_ongoing_strawberry_two_day_interval_and_gap_day() -> None:
    # STRAWBERRY (ongoing, interval 2, first 10, max 4): accrual only on
    # even days-since-first (ends of days 9, 11, 13, 15); a gap-day HARVEST
    # collects nothing; after max production the depleted plant decays at
    # step 408 (day 17 hour 0).
    events = [
        (0, 0, act(market=[["BUY_SEED", "STRAWBERRY", 1]])),
        (0, 1, act(farmer=["PLANT", "STRAWBERRY"])),
        (0, 2, act(farmer=["WATER"])),
        *water_daily(17),
        (10, 1, act(farmer=["HARVEST"])),
        (11, 1, act(farmer=["HARVEST"])),  # gap day: no accrual end of day 10
        (12, 1, act(farmer=["HARVEST"])),
        (14, 1, act(farmer=["HARVEST"])),
        (16, 1, act(farmer=["HARVEST"])),
        (17, 1, act(farmer=["DIG"])),
    ]
    trace = build_turns(events, total_days=18)
    replay("ongoing_strawberry_interval", trace)
    states = fast_timeline(CONFIG, trace)
    inv = lambda s: states[s]["privates"][0]["inventories"][0]["STRAWBERRY"]
    shed = lambda s: states[s]["privates"][0]["shed"]["STRAWBERRY"]
    assert inv(turn_index(10, 1)) == 1
    # Gap day: nothing accrued at end of day 10, and yesterday's take already
    # sits in the shed, so the carry is empty.
    assert inv(turn_index(11, 1)) == 0 and shed(turn_index(11, 1)) == 1
    # Daily takes drain each 2-day accrual to exactly 1 unit; the shed reaches
    # the max_yield total of 4 after the day-16 drop.
    assert inv(turn_index(12, 1)) == 1 and shed(turn_index(12, 1)) == 1
    assert inv(turn_index(14, 1)) == 1 and shed(turn_index(14, 1)) == 2
    assert inv(turn_index(16, 1)) == 1 and shed(turn_index(16, 1)) == 3
    assert shed(turn_index(16, 23)) == 4
    assert tile(states[turn_index(16, 1)], 4, 4)["kind"] == "PLANT"
    assert tile(states[turn_index(17, 0)], 4, 4) == {"kind": "WEED"}
    assert tile(states[turn_index(17, 1)], 4, 4) is None


# ---------------------------------------------------------------------------
# DIG interactions
# ---------------------------------------------------------------------------


def test_dig_clears_structures_but_never_a_placed_animal() -> None:
    # DIG arm: removes plants/weeds/empty structures; a placed animal makes
    # DIG a no-op ("animal" in tile early-return).
    events = [
        (0, 0, act(farmer=["BUILD_COOP"])),
        (1, 0, act(farmer=["DIG"])),
        (2, 0, act(farmer=["BUILD_COOP"])),
        (2, 1, act(market=[["BUY_ANIMAL", "GOOSE", 1]])),
        # Pickup and place within the SAME day: carried inventories drop to
        # the shed at each day end.
        (3, 0, act(farmer=["PICKUP", "GOOSE", 1])),
        (3, 1, act(farmer=["PLACE", "GOOSE"])),
        (4, 0, act(farmer=["DIG"])),
    ]
    trace = build_turns(events, total_days=5)
    replay("dig_interactions", trace)
    states = fast_timeline(CONFIG, trace)
    assert tile(states[turn_index(0, 0)], 4, 4) == {"kind": "COOP"}
    assert tile(states[turn_index(1, 0)], 4, 4) is None
    placed = states[turn_index(3, 1)]
    assert tile(placed, 4, 4) == {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 3, "yield_units": 0,
        "consecutive_unfed": 0, "fed_today": False, "cared_today": False,
        "fertilizer_available": False, "pending_care_bonus": 0,
    }
    after_dig = states[turn_index(4, 0)]
    assert tile(after_dig, 4, 4)["animal"] == "GOOSE"


# ---------------------------------------------------------------------------
# Silent no-op guards
# ---------------------------------------------------------------------------


def test_crop_guards_water_fertilize_harvest_on_wrong_tiles() -> None:
    # WATER/FERTILIZE/HARVEST require a PLANT tile; FERTILIZE additionally
    # requires carried fertilizer; HARVEST before first_yield_day is guarded
    # even for non-ongoing crops (WHEAT age 0 with yield 1).
    events = [
        (0, 0, act(farmer=["WATER"])),     # empty tile
        (0, 1, act(farmer=["FERTILIZE"])),  # empty tile, no fertilizer
        (0, 2, act(farmer=["HARVEST"])),   # empty tile
        (0, 3, act(farmer=["DIG"])),       # empty tile: no-op
        (0, 4, act(market=[["BUY_SEED", "WHEAT", 1]])),
        (0, 5, act(farmer=["PLANT", "WHEAT"])),
        (0, 6, act(farmer=["FERTILIZE"])),  # no carried fertilizer
        (0, 7, act(farmer=["HARVEST"])),   # age 0 < first_yield_day 2
    ]
    trace = build_turns(events, total_days=1)
    replay("crop_guard_noops", trace)
    states = fast_timeline(CONFIG, trace)
    guarded = tile(states[turn_index(0, 7)], 4, 4)
    assert guarded["fertilized_until_day"] == -1
    assert guarded["yield_units"] == 1
    assert states[turn_index(0, 7)]["privates"][0]["inventories"][0]["WHEAT"] == 0
