"""Focused tests for the canonical daily replay extractor.

Synthetic fixtures intentionally mirror the real 1.32.7 replay schema while
keeping boards tiny (2x2). The real-replay smoke test runs against the ignored
local sample when present.
"""

import json
from pathlib import Path

import pytest

from replay_daily.constants import fib, hire_cost, total_hire_cost
from replay_daily.extractor import (
    OPPONENT_PRIVATE_KEYS,
    VersionMismatch,
    extract_replay,
    load_manifest,
)
from replay_daily.lifecycle import canonical_tile, derive_animal, derive_plant

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "samples" / "2026-08-20" / "94735084.json"
MANIFEST = REPO_ROOT / "data" / "samples" / "2026-08-20" / "manifest.csv"

DEFAULT_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


# ---------------------------------------------------------------- fixtures


def empty_tiles(n: int = 2) -> list[list[None]]:
    return [[None] * n for _ in range(n)]


def make_farm(
    money: float = 3000.0,
    tiles: list | None = None,
    unlocked: tuple = ("NW",),
    hires_today: int = 0,
    hands: list | None = None,
) -> dict:
    return {
        "farmer": [0, 0],
        "hands": [list(h) for h in (hands or [])],
        "hires_today": hires_today,
        "money": money,
        "tiles": tiles if tiles is not None else empty_tiles(),
        "unlocked_quadrants": list(unlocked),
    }


def make_obs(
    player: int,
    day: int,
    hour: int,
    step: int,
    farms: list | None = None,
    privates: list | None = None,
    town: dict | None = None,
) -> dict:
    return {
        "day": day,
        "hour": hour,
        "step": step,
        "player": player,
        "farms": farms if farms is not None else [make_farm(), make_farm()],
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": town if town is not None else {"unlocked_shops": []},
        "private": (
            privates[player]
            if privates is not None
            else {"shed": {}, "seeds": {}, "inventories": [{}]}
        ),
        "remainingOverageTime": 60,
    }


def make_replay(specs: list[dict]) -> dict:
    """Build a synthetic replay faithful to the real schema.

    `specs[i]` may set: day, hour, action{0,1}, farms{0,1}, private{0,1},
    town{0,1}, step. The default outer `step` value is deliberately irregular
    (i * 7 + 3) so no test can accidentally depend on step % 24 or index==step.
    """
    steps = []
    for i, spec in enumerate(specs):
        agents = []
        for seat in (0, 1):
            obs = make_obs(
                player=seat,
                day=spec["day"],
                hour=spec["hour"],
                step=spec.get("step", i * 7 + 3),
                farms=spec.get("farms"),
                privates=spec.get("private"),
                town=spec.get("town"),
            )
            if spec.get(f"farms{seat}"):
                obs["farms"][seat] = spec[f"farms{seat}"]
            if spec.get(f"private{seat}"):
                obs["private"] = spec[f"private{seat}"]
            if spec.get(f"town{seat}"):
                obs["town"] = spec[f"town{seat}"]
            agents.append({
                "action": spec.get(f"action{seat}", DEFAULT_ACTION),
                "info": {},
                "observation": obs,
                "reward": 0,
                "status": "ACTIVE",
            })
        steps.append(agents)
    return {
        "id": "synthetic-uuid",
        "info": {"EpisodeId": 42, "TeamNames": ["Alpha", "Beta"], "seed": 12345},
        "module_version": "1.32.7",
        "name": "kaggriculture",
        "rewards": [100.0, 200.0],
        "statuses": ["DONE", "DONE"],
        "configuration": {"turnsPerDay": 24, "farmHandCostMult": 1},
        "schema_version": 1,
        "version": "0.1.0",
        "steps": steps,
    }


def assert_no_opponent_private(node) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            assert key not in OPPONENT_PRIVATE_KEYS, f"leaked opponent key: {key}"
            assert_no_opponent_private(value)
    elif isinstance(node, list):
        for value in node:
            assert_no_opponent_private(value)


# ------------------------------------------------------- day segmentation


def test_day_segmentation_explicit_boundaries_no_modulo():
    specs = [
        {"day": 0, "hour": 0},
        {"day": 0, "hour": 1},
        {"day": 0, "hour": 2},
        {"day": 1, "hour": 0},
        {"day": 1, "hour": 1},
    ]
    records = extract_replay(make_replay(specs))
    by_seat = {seat: sorted(r["day"] for r in records if r["metadata"]["seat"] == seat)
               for seat in (0, 1)}
    assert by_seat == {0: [0, 1], 1: [0, 1]}

    day0 = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 0)
    day1 = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 1)
    # Boundary states selected purely by explicit day/hour fields.
    assert day0["start"]["hour"] == 0 and day0["start"]["day"] == 0
    assert day0["end"]["boundary"] == "next_day_start"
    assert (day0["end"]["day"], day0["end"]["hour"]) == (1, 0)
    assert day1["end"]["boundary"] == "terminal"
    assert (day1["end"]["day"], day1["end"]["hour"]) == (1, 1)


def test_boundary_state_selection_uses_hour_zero_observation():
    marker_tiles = [[{"kind": "WEED"}, None], [None, None]]
    specs = [
        {"day": 0, "hour": 0, "farms0": make_farm(tiles=marker_tiles)},
        {"day": 0, "hour": 1},
        {"day": 1, "hour": 0},
    ]
    records = extract_replay(make_replay(specs))
    day0 = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 0)
    assert day0["start"]["self"]["board"][0][0] == {"kind": "WEED", "derived": None}


def test_lifecycle_uses_observation_step_not_outer_index():
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
            "yield_units": 1, "max_lifespan_step": 50,
            "fertilized_until_day": -1, "consecutive_unwatered": 0,
            "watered_today": False}
    specs = [
        {"day": 0, "hour": 0, "step": 100,
         "farms0": make_farm(tiles=[[tile, None], [None, None]])},
        {"day": 1, "hour": 0, "step": 124},
    ]
    record = next(r for r in extract_replay(make_replay(specs))
                  if r["metadata"]["seat"] == 0 and r["day"] == 0)
    assert record["start"]["self"]["board"][0][0]["derived"]["past_lifespan"] is True


# --------------------------------------------------- action alignment


def test_action_attribution_uses_preceding_observation():
    specs = [
        {"day": 0, "hour": 0},
        {"day": 0, "hour": 1},
        {"day": 0, "hour": 2,
         "action0": {"farmer": ["PASS"], "hands": [],
                     "market": [["SELL", "WHEAT", 5]]}},
        {"day": 0, "hour": 23},
        {"day": 1, "hour": 0,
         "action0": {"farmer": ["PASS"], "hands": [],
                     "market": [["SELL", "CARROT", 2]]}},
        {"day": 1, "hour": 1,
         "action0": {"farmer": ["PASS"], "hands": [],
                     "market": [["SELL", "MELON", 1]]}},
    ]
    records = extract_replay(make_replay(specs))
    day0 = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 0)
    day1 = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 1)
    # steps[2].action transformed obs[1] -> obs[2]: attributed to obs[1] hour 1.
    assert {"product": "WHEAT", "quantity": 5, "hour": 1} in day0["events"]["sells"]
    # steps[4].action (stored on the day-1 hour-0 entry) acted on obs hour 23:
    # belongs to day 0, not day 1.
    assert {"product": "CARROT", "quantity": 2, "hour": 23} in day0["events"]["sells"]
    assert day1["events"]["sells"] == [{"product": "MELON", "quantity": 1, "hour": 0}]
    assert all(s["hour"] >= 4 or s["product"] != "CARROT" for s in day1["events"]["sells"])


def test_initial_default_action_is_not_an_event():
    specs = [{"day": 0, "hour": 0}, {"day": 0, "hour": 1}]
    records = extract_replay(make_replay(specs))
    for rec in records:
        assert rec["events"]["sells"] == []
        assert rec["events"]["market_events_ordered"] == []


# --------------------------------------------------------- privacy


def test_two_seat_canonicalization_and_privacy():
    privates = [
        {"shed": {"WHEAT": 3}, "seeds": {"CARROT": 2}, "inventories": [{"FERTILIZER": 1}]},
        {"shed": {"WOOL": 1}, "seeds": {"MELON": 9}, "inventories": [{}]},
    ]
    specs = [
        {"day": 0, "hour": 0, "private": privates},
        {"day": 0, "hour": 1, "private": privates},
        {"day": 1, "hour": 0, "private": privates},
    ]
    records = extract_replay(make_replay(specs))
    assert {r["metadata"]["seat"] for r in records} == {0, 1}
    for rec in records:
        seat = rec["metadata"]["seat"]
        own_shed = privates[seat]["shed"]
        assert rec["start"]["self"]["shed"] == own_shed
        assert rec["end"]["self"]["shed"] == own_shed
        assert rec["metadata"]["player"] == ["Alpha", "Beta"][seat]
        assert rec["metadata"]["opponent"] == ["Alpha", "Beta"][1 - seat]
        for side in ("opponent_public",):
            assert_no_opponent_private(rec["start"][side])
            assert_no_opponent_private(rec["end"][side])
        # The opponent view must never equal the writer's own private state.
        assert rec["start"]["opponent_public"].get("shed", None) is None


# --------------------------------------------------------- sell bucketing


def test_sell_six_bin_bucketing_and_exact_hours():
    sells = [(0, "WHEAT", 1), (3, "WHEAT", 2), (4, "CARROT", 3), (7, "EGG", 1),
             (8, "MELON", 1), (11, "WOOL", 4), (12, "MILK", 2), (15, "EGG", 1),
             (16, "TOMATO", 1), (19, "WHEAT", 1), (20, "MELON", 5), (23, "EGG", 2)]
    # An action stored on the entry following observation hour H is attributed
    # to hour H. Build day-0 hours 0..23 plus the day-1 hour-0 boundary entry.
    sales_by_action_index: dict[int, list[tuple[str, int]]] = {}
    for hour, product, qty in sells:
        # action index = hour + 1 within specs[1..24]; hour 23 -> boundary entry.
        sales_by_action_index.setdefault(hour + 1, []).append((product, qty))
    specs = [{"day": 0, "hour": 0}]
    for h in range(1, 24):
        orders = [["SELL", p, q] for p, q in sales_by_action_index.get(h, [])]
        specs.append({"day": 0, "hour": h,
                      "action0": {"farmer": ["PASS"], "hands": [], "market": orders}})
    boundary_orders = [["SELL", p, q] for p, q in sales_by_action_index.get(24, [])]
    specs.append({"day": 1, "hour": 0,
                  "action0": {"farmer": ["PASS"], "hands": [], "market": boundary_orders}})
    records = extract_replay(make_replay(specs))
    rec = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 0)
    bins = rec["targets"]["sell_quantity"]
    assert sorted(bins) == ["0", "12", "16", "20", "4", "8"]
    assert bins["0"]["WHEAT"] == 3      # hours 0, 3
    assert bins["4"]["CARROT"] == 3 and bins["4"]["EGG"] == 1
    assert bins["8"]["MELON"] == 1 and bins["8"]["WOOL"] == 4
    assert bins["12"]["MILK"] == 2 and bins["12"]["EGG"] == 1
    assert bins["16"]["TOMATO"] == 1 and bins["16"]["WHEAT"] == 1
    assert bins["20"]["MELON"] == 5 and bins["20"]["EGG"] == 2
    # Exact primitive hours retained in the ledger.
    assert {"product": "EGG", "quantity": 2, "hour": 23} in rec["events"]["sells"]
    assert len(rec["events"]["sells"]) == len(sells)


# --------------------------------------------------------- composition


def test_crop_animal_composition_and_land_expansion():
    start_tiles = [[None, None], [None, None]]
    end_tiles = [
        [{"kind": "PLANT", "crop": "MELON", "planted_day": 0, "watered_today": False,
          "consecutive_unwatered": 0, "fertilized_until_day": -1,
          "max_lifespan_step": 312, "yield_units": 3},
         {"kind": "PLANT", "crop": "WHEAT", "planted_day": 4, "watered_today": False,
          "consecutive_unwatered": 0, "fertilized_until_day": -1,
          "max_lifespan_step": 216, "yield_units": 3}],
        [{"kind": "PLANT", "crop": "MELON", "planted_day": 0, "watered_today": False,
          "consecutive_unwatered": 0, "fertilized_until_day": -1,
          "max_lifespan_step": 312, "yield_units": 3},
         {"kind": "PASTURE", "animal": "COW", "placed_day": 0, "fed_today": False,
          "cared_today": False, "consecutive_unfed": 0, "fertilizer_available": False,
          "pending_care_bonus": 0, "yield_units": 2}],
    ]
    specs = [
        {"day": 0, "hour": 0, "farms0": make_farm(tiles=start_tiles)},
        {"day": 0, "hour": 1},
        {"day": 1, "hour": 0,
         "farms0": make_farm(tiles=end_tiles, unlocked=("NW", "NE"))},
    ]
    records = extract_replay(make_replay(specs))
    rec = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 0)
    assert rec["targets"]["crop_composition_end"] == {"MELON": 2, "WHEAT": 1}
    assert rec["targets"]["animal_counts_end"] == {"GOOSE": 0, "COW": 1, "SHEEP": 0}
    assert rec["targets"]["land_expansion"] == {"expanded": True, "new_quadrants": ["NE"]}
    assert rec["targets"]["unlocked_quadrants_end"] == ["NW", "NE"]


def _land_purchase_record(post_unlocked: tuple, orders: list[list]) -> dict:
    specs = [
        {"day": 0, "hour": 0, "farms0": make_farm(unlocked=("NW",))},
        {"day": 1, "hour": 0,
         "farms0": make_farm(unlocked=post_unlocked),
         "action0": {"farmer": ["PASS"], "hands": [], "market": orders}},
    ]
    return next(r for r in extract_replay(make_replay(specs))
                if r["metadata"]["seat"] == 0 and r["day"] == 0)


def test_buy_land_success_uses_observed_quadrant_transition():
    record = _land_purchase_record(("NW", "NE"), [["BUY_LAND"]])
    assert record["events"]["land_purchases"] == [{"quadrant": "NE", "hour": 0}]


def test_buy_land_failed_purchase_keeps_quadrant_unknown():
    record = _land_purchase_record(("NW",), [["BUY_LAND"]])
    assert record["events"]["land_purchases"] == [{"quadrant": None, "hour": 0}]


def test_buy_land_multiple_intents_only_observed_successes_get_quadrants():
    record = _land_purchase_record(
        ("NW", "NE"), [["BUY_LAND"], ["BUY_LAND"]]
    )
    assert record["events"]["land_purchases"] == [
        {"quadrant": "NE", "hour": 0},
        {"quadrant": None, "hour": 0},
    ]


# --------------------------------------------------------- hires/cost


def test_previous_day_hire_count_and_fibonacci_cost():
    def farm(hires: int) -> dict:
        return make_farm(hires_today=hires)

    specs = [
        {"day": 0, "hour": 0, "farms0": farm(0)},
        {"day": 0, "hour": 1, "farms0": farm(1),
         "action0": {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}},
        {"day": 0, "hour": 2, "farms0": farm(2),
         "action0": {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}},
        {"day": 0, "hour": 3, "farms0": farm(3),
         "action0": {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"]]}},
        {"day": 1, "hour": 0, "farms0": farm(0)},
    ]
    records = extract_replay(make_replay(specs))
    day0 = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 0)
    day1 = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 1)
    # Day 0 has no previous day: zeros.
    assert day0["start"]["previous_execution"] == {"workers_hired": 0, "hire_cost": 0}
    # Realized hires come from state (hires_today max = 3); cost = fib(0)+fib(1)+fib(2) = 4.
    assert day1["start"]["previous_execution"] == {"workers_hired": 3, "hire_cost": 4}
    # Submitted intent counted separately (4 HIRE orders submitted, 3 realized).
    assert day0["events"]["hires"]["submitted"] == 4
    assert day0["events"]["hires"]["realized"] == {"workers_hired": 3, "hire_cost": 4}
    assert total_hire_cost(3) == 4 and hire_cost(0) == 1 and fib(4) == 5


def test_hire_cost_multiplier_from_configuration():
    def farm(hires: int) -> dict:
        return make_farm(hires_today=hires)

    specs = [
        {"day": 0, "hour": 0, "farms0": farm(0)},
        {"day": 0, "hour": 1, "farms0": farm(2)},
        {"day": 1, "hour": 0, "farms0": farm(0)},
    ]
    replay = make_replay(specs)
    replay["configuration"]["farmHandCostMult"] = 3
    records = extract_replay(replay)
    day1 = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 1)
    assert day1["start"]["previous_execution"] == {"workers_hired": 2, "hire_cost": 6}


# --------------------------------------------------------- lifecycle timing


def test_crop_lifecycle_timing_cases():
    wheat_young = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 2,
                   "yield_units": 1, "max_lifespan_step": 168,
                   "fertilized_until_day": -1, "consecutive_unwatered": 0,
                   "watered_today": False}
    d = derive_plant(wheat_young, current_day=3, current_step=80)
    assert d["currently_harvestable"] is False
    assert d["days_until_next_harvest"] == 1  # first_yield_day 2 - age 1

    wheat_ready = dict(wheat_young, planted_day=0, yield_units=3)
    d = derive_plant(wheat_ready, current_day=4, current_step=100)
    assert d["currently_harvestable"] is True and d["days_until_next_harvest"] == 0

    # Ongoing strawberry planted day 5 at day 8: first production enters day 15.
    strawberry = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 5,
                  "yield_units": 0, "max_lifespan_step": -1,
                  "fertilized_until_day": -1, "consecutive_unwatered": 0,
                  "watered_today": True}
    d = derive_plant(strawberry, current_day=8, current_step=200)
    assert d["days_until_next_harvest"] == 7
    # Planted day 0 instead: production already due entering day 10 -> 2 days away.
    strawberry0 = dict(strawberry, planted_day=0)
    d = derive_plant(strawberry0, current_day=8, current_step=200)
    assert d["days_until_next_harvest"] == 2
    assert derive_plant(dict(strawberry, watered_today=False), current_day=8,
                        current_step=200)["days_until_next_harvest"] is None

    # Tomato production cap: last production enters day 11; afterwards null.
    tomato = {"kind": "PLANT", "crop": "TOMATO", "planted_day": 0,
              "yield_units": 0, "max_lifespan_step": -1,
              "fertilized_until_day": -1, "consecutive_unwatered": 0,
              "watered_today": True}
    assert derive_plant(tomato, current_day=10, current_step=0)["days_until_next_harvest"] == 1
    assert derive_plant(tomato, current_day=11, current_step=0)["days_until_next_harvest"] is None

    # Non-ongoing crop without units: watering-dependent, explicitly null.
    carrot = {"kind": "PLANT", "crop": "CARROT", "planted_day": 0,
              "yield_units": 0, "max_lifespan_step": 96,
              "fertilized_until_day": -1, "consecutive_unwatered": 0,
              "watered_today": False}
    assert derive_plant(carrot, current_day=1, current_step=30)["days_until_next_harvest"] is None

    # Decay flag once the global step reaches max_lifespan_step.
    melon_old = {"kind": "PLANT", "crop": "MELON", "planted_day": 0,
                 "yield_units": 2, "max_lifespan_step": 312,
                 "fertilized_until_day": -1, "consecutive_unwatered": 0,
                 "watered_today": False}
    d = derive_plant(melon_old, current_day=13, current_step=312)
    assert d["past_lifespan"] is True and d["currently_harvestable"] is True

    # Fertilizer active window (engine: fertilized through day+2 inclusive).
    melon_fert = dict(melon_old, fertilized_until_day=5)
    assert derive_plant(melon_fert, current_day=5, current_step=0)["fertilizer_active"] is True
    assert derive_plant(melon_fert, current_day=6, current_step=0)["fertilizer_active"] is False


def test_animal_lifecycle_timing_cases():
    cow_waiting = {"kind": "PASTURE", "animal": "COW", "placed_day": 0,
                   "yield_units": 0, "consecutive_unfed": 0, "fed_today": False,
                   "cared_today": False, "fertilizer_available": False,
                   "pending_care_bonus": 0}
    d = derive_animal(cow_waiting, current_day=8)
    assert d["days_until_next_product"] == 2  # production enters day 10 (interval 2)
    assert d["starving"] is False

    sheep_ready = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 0,
                   "yield_units": 3, "consecutive_unfed": 0, "fed_today": True,
                   "cared_today": False, "fertilizer_available": True,
                   "pending_care_bonus": 1}
    d = derive_animal(sheep_ready, current_day=9)
    assert d["currently_harvestable"] is True and d["days_until_next_product"] == 0

    goose_starving = {"kind": "COOP", "animal": "GOOSE", "placed_day": 2,
                      "yield_units": 0, "consecutive_unfed": 1, "fed_today": False,
                      "cared_today": False, "fertilizer_available": False,
                      "pending_care_bonus": 0}
    d = derive_animal(goose_starving, current_day=5)
    assert d["days_until_next_product"] is None  # escape enters day 6
    assert d["starving"] is True


def test_canonical_tile_preserves_raw_and_distinguishes_kinds():
    plant = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 1,
             "max_lifespan_step": 120, "fertilized_until_day": -1,
             "consecutive_unwatered": 0, "watered_today": False}
    out = canonical_tile(plant, current_day=1, current_step=30)
    assert out["crop"] == "WHEAT" and out["planted_day"] == 0  # raw preserved
    assert "derived" in out
    assert canonical_tile(None, 0, 0) is None
    assert canonical_tile("LOCKED", 0, 0) == "LOCKED"
    assert canonical_tile({"kind": "WEED"}, 0, 0) == {"kind": "WEED", "derived": None}


# --------------------------------------------------------- town duplicates


def test_town_duplicate_retention_and_counts():
    town = {"unlocked_shops": ["PIZZA_SHOP", "ICE_CREAM_SHOP", "PIZZA_SHOP"]}
    specs = [{"day": 0, "hour": 0, "town": town}, {"day": 1, "hour": 0}]
    records = extract_replay(make_replay(specs))
    rec = next(r for r in records if r["metadata"]["seat"] == 0 and r["day"] == 0)
    assert rec["start"]["town"]["unlocked_shops"] == \
        ["PIZZA_SHOP", "ICE_CREAM_SHOP", "PIZZA_SHOP"]
    assert rec["start"]["town"]["shop_counts"] == \
        {"PIZZA_SHOP": 2, "ICE_CREAM_SHOP": 1}


# --------------------------------------------------------- terminal day


def test_terminal_final_day_record_retained():
    specs = [{"day": 28, "hour": 23}, {"day": 29, "hour": 0}, {"day": 29, "hour": 23}]
    records = extract_replay(make_replay(specs))
    finals = [r for r in records if r["day"] == 29]
    assert len(finals) == 2
    for rec in finals:
        assert rec["end"]["boundary"] == "terminal"
        assert (rec["end"]["day"], rec["end"]["hour"]) == (29, 23)


# --------------------------------------------------------- version policy


def test_non_1327_replay_rejected():
    replay = make_replay([{"day": 0, "hour": 0}])
    replay["module_version"] = "1.32.6"
    with pytest.raises(VersionMismatch):
        extract_replay(replay)


# --------------------------------------------------------- real replay smoke


@pytest.mark.skipif(not SAMPLE.exists(), reason="local real sample not present")
def test_real_replay_smoke_60_records_alignment_and_privacy():
    out_path = REPO_ROOT / "data" / "temp" / "smoke_94735084.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from replay_daily.cli import main as cli_main

    rc = cli_main([
        "extract",
        "--input", str(SAMPLE),
        "--manifest", str(MANIFEST),
        "--source-dataset", "kaggle/kaggriculture-episodes-2026-08-20",
        "--partition-date", "2026-08-20",
        "--output", str(out_path),
    ])
    assert rc == 0

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 60  # 30 days x 2 seats
    records = [json.loads(line) for line in lines]  # JSONL parseability
    keys = {(r["metadata"]["seat"], r["day"]) for r in records}
    assert keys == {(seat, day) for seat in (0, 1) for day in range(30)}

    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for rec in records:
        meta = rec["metadata"]
        assert meta["module_version"] == "1.32.7"
        assert meta["episode_id"] == 94735084
        assert meta["avg_score"] is not None and meta["min_score"] is not None
        assert meta["seed"] == raw["info"]["seed"]
        assert_no_opponent_private(rec["start"]["opponent_public"])
        assert_no_opponent_private(rec["end"]["opponent_public"])

    # Action alignment on real data: every SELL order stored at steps[i>=1] must
    # be attributed to exactly one (seat, day) ledger with its preceding hour.
    expected_sells = []
    for i, step in enumerate(raw["steps"]):
        for seat in (0, 1):
            for order in step[seat]["action"].get("market") or []:
                if order[0] == "SELL":
                    pre = raw["steps"][i - 1][seat]["observation"]
                    expected_sells.append((seat, pre["day"], order[1], order[2], pre["hour"]))
    got_sells = []
    for rec in records:
        for sale in rec["events"]["sells"]:
            got_sells.append((
                rec["metadata"]["seat"], rec["day"],
                sale["product"], sale["quantity"], sale["hour"],
            ))
    assert sorted(got_sells) == sorted(expected_sells)

    # Known probe fact: steps[2][0] SELL WHEAT 3 acted on obs[1] (day 0, hour 1).
    day0_seat0 = next(r for r in records
                      if r["metadata"]["seat"] == 0 and r["day"] == 0)
    assert {"product": "WHEAT", "quantity": 3, "hour": 1} in day0_seat0["events"]["sells"]

    # Final banks flow into per-seat metadata.
    seat0 = next(r for r in records if r["metadata"]["seat"] == 0)
    assert seat0["metadata"]["final_bank_self"] == raw["rewards"][0]
    assert seat0["metadata"]["final_bank_opponent"] == raw["rewards"][1]


def test_manifest_loader_joins_scores():
    if not MANIFEST.exists():
        pytest.skip("local manifest not present")
    rows = load_manifest(str(MANIFEST))
    row = rows[94735084]
    assert float(row["avg_score"]) > 0 and float(row["min_score"]) > 0
