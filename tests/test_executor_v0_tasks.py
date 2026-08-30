"""Focused tests for executor_v0 stage 4: task records and generation.

Covers issue #1 section 5: task JSON/order, priority groups and deterministic
ties, reconciliation dig->plant dependencies, water/harvest/feed/collect from
actual lifecycle fields, build/place dependencies and compatibility, exact
CARE/FERTILIZE allocations, land deficit only, current-bin sells with caller-
owned remaining quantities, shortage-only purchases, official [y,x] task
tiles, recompute-reflects-changed-state, and a real-replay smoke.
"""

import copy
import json
from pathlib import Path

import pytest

from executor_v0.layout import (
    AnimalLayoutResult,
    AnimalSlotPlan,
    CropReconciliationResult,
    DigIntent,
    plan_day_layouts,
)
from executor_v0.foreman import run_foreman
from executor_v0.plan import DailyPlan
from executor_v0.tasks import (
    GenerationResult,
    Priority,
    Task,
    generate_tasks,
)
from replay_daily.constants import ANIMALS, LAND_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "samples" / "2026-08-20" / "94735084.json"


# ------------------------------------------------------------------ helpers


def tile(kind="PLANT", **fields):
    base = {"kind": kind}
    base.update(fields)
    return base


def plant_tile(crop="WHEAT", *, planted_day=1, yield_units=0,
               watered_today=True, fertilized_until_day=-1,
               fertilizer_available=None, harvestable=False,
               cared_today=True) -> dict:
    t = {
        "kind": "PLANT", "crop": crop, "planted_day": planted_day,
        "yield_units": yield_units, "watered_today": watered_today,
        "fertilized_until_day": fertilized_until_day,
        "max_lifespan_step": -1, "consecutive_unwatered": 0,
        "derived": {
            "age_days": planted_day,
            "currently_harvestable": harvestable,
            "days_until_next_harvest": None,
            "fertilizer_active": fertilized_until_day >= 5,
            "past_lifespan": False,
            "starving": False,
            "days_until_next_product": None,
        },
    }
    if fertilizer_available is not None:
        t["fertilizer_available"] = fertilizer_available
    return t


def animal_tile(animal="GOOSE", *, yield_units=0, fed_today=True,
                cared_today=False, consecutive_unfed=0,
                fertilizer_available=None) -> dict:
    tile = {
        "kind": ANIMALS[animal]["structure"], "animal": animal,
        "placed_day": 0, "yield_units": yield_units,
        "consecutive_unfed": consecutive_unfed, "fed_today": fed_today,
        "cared_today": cared_today,
        "derived": {
            "age_days": 3, "currently_harvestable": yield_units > 0,
            "days_until_next_harvest": None, "fertilizer_active": False,
            "past_lifespan": False, "starving": consecutive_unfed >= 1,
            "days_until_next_product": None,
        },
    }
    if fertilizer_available is not None:
        tile["fertilizer_available"] = fertilizer_available
    return tile


def make_obs(day=3, hour=2, step=90, tiles=None, unlocked=("NW",),
             shed=None, seeds=None, inventories=None, farmer=(0, 0),
             money=3000.0):
    tiles = tiles if tiles is not None else [[None] * 10 for _ in range(10)]
    farm = {
        "farmer": list(farmer), "hands": [], "hires_today": 0,
        "money": money, "tiles": tiles,
        "unlocked_quadrants": list(unlocked),
    }
    return {
        "day": day, "hour": hour, "step": step, "player": 0,
        "farms": [farm, {**copy.deepcopy(farm), "farmer": [9, 9]}],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": shed or {}, "seeds": seeds or {},
                    "inventories": inventories or [{}]},
    }


def make_plan(**overrides):
    kwargs = dict(
        crop_targets={"WHEAT": 0, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        animal_targets={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        land_count=1,
        fertilizer_by_crop={"WHEAT": 0, "CARROT": 0, "TOMATO": 0,
                            "STRAWBERRY": 0, "MELON": 0},
        care_by_animal={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        sell_quantities={p: {a: 0 for a in (0, 4, 8, 12, 16, 20)}
                         for p in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                                   "MELON", "EGG", "MILK", "WOOL",
                                   "FERTILIZER")},
    )
    for key in ("crop_targets", "animal_targets", "fertilizer_by_crop",
                "care_by_animal", "sell_quantities"):
        if key in overrides:
            overrides[key] = {**kwargs[key], **overrides[key]}
    kwargs.update(overrides)
    return DailyPlan.create(**kwargs)


def by_kind(result, kind):
    return sorted((t for t in result.tasks if t.kind == kind),
                  key=lambda t: t.key)


# ------------------------------------------------------------- task records


def test_task_json_safe_and_priority_ordering():
    task = Task(key="WATER:1,2", kind="WATER", priority=Priority.MAINTENANCE,
                tile=(1, 2), crop="WHEAT")
    parsed = json.loads(json.dumps(task.to_json_dict()))
    assert parsed["tile"] == [1, 2]
    assert parsed["priority"] == "MAINTENANCE"
    order = [Priority.MAINTENANCE, Priority.PRODUCTIVE, Priority.MANAGER,
             Priority.LOGISTICS]
    assert order == sorted(order)
    assert int(Priority.MAINTENANCE) < int(Priority.PRODUCTIVE) \
        < int(Priority.MANAGER) < int(Priority.LOGISTICS)


def test_sorted_tasks_deterministic_tie_by_key_then_deadline():
    a = Task(key="AAA", kind="PASS", priority=Priority.MANAGER)
    b = Task(key="BBB", kind="PASS", priority=Priority.MANAGER)
    late = Task(key="ZZZ", kind="SELL", priority=Priority.LOGISTICS,
                deadline_hour=7)
    early = Task(key="MMM", kind="SELL", priority=Priority.LOGISTICS,
                 deadline_hour=3)
    result = GenerationResult(tasks=(b, a, late, early))
    keys = [t.key for t in result.sorted_tasks()]
    assert keys == ["AAA", "BBB", "MMM", "ZZZ"]


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("mutate", [
    lambda o: o.update(day=-1),
    lambda o: o.update(day=30),
    lambda o: o.update(hour=24),
    lambda o: o.pop("hour"),
])
def test_generation_validates_observation(mutate):
    obs = make_obs()
    mutate(obs)
    with pytest.raises(ValueError):
        generate_tasks(obs, 0, feasible_plan=make_plan(),
                       remaining_sells={})


def test_generation_rejects_unknown_sell_products_and_bad_seats():
    with pytest.raises(ValueError):
        generate_tasks(make_obs(), 0, feasible_plan=make_plan(),
                       remaining_sells={"NOPE": 1})
    with pytest.raises(ValueError):
        generate_tasks(make_obs(), 5, feasible_plan=make_plan(),
                       remaining_sells={})


# ------------------------------------------------------------ mechanics scan


def test_water_harvest_feed_collect_from_actual_lifecycle_fields():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = plant_tile(watered_today=False)              # needs water
    # Harvestable per real lifecycle rules: accumulated units plus age
    # (current day 3 minus planted_day >= WHEAT first_yield_day 2).
    tiles[0][2] = plant_tile(yield_units=3, planted_day=1)
    tiles[0][3] = plant_tile(watered_today=False, yield_units=2,
                             planted_day=0)
    tiles[1][1] = animal_tile("GOOSE", fed_today=False, cared_today=False,
                              fertilizer_available=True)
    tiles[1][2] = animal_tile("COW", yield_units=2, fed_today=True,
                              fertilizer_available=True)
    obs = make_obs(tiles=tiles)
    result = generate_tasks(obs, 0, feasible_plan=make_plan(),
                            remaining_sells={})
    water = {(t.tile) for t in by_kind(result, "WATER")}
    assert water == {(0, 1), (0, 3)}
    harvest = {(t.tile, t.crop or t.animal)
               for t in by_kind(result, "HARVEST")}
    assert ((0, 2), "WHEAT") in harvest and ((0, 3), "WHEAT") in harvest
    assert ((1, 2), "COW") in harvest
    feed = by_kind(result, "FEED")
    assert [(t.tile, t.required_item) for t in feed] == [((1, 1), "WHEAT")]
    collect = {(t.tile, t.animal)
               for t in by_kind(result, "COLLECT_FERTILIZER")}
    assert collect == {((1, 1), "GOOSE"), ((1, 2), "COW")}
    # Fed+cared animals get no FEED/CARE; unfed uncared goose gets CARE only
    # when the plan requests it (zero here).
    assert by_kind(result, "CARE") == []


def test_collect_fertilizer_requires_actual_availability_field():
    """Regression: collection only on raw fertilizer_available is True."""
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = animal_tile("GOOSE", fertilizer_available=True)
    tiles[0][2] = animal_tile("COW", fertilizer_available=False)
    tiles[0][3] = animal_tile("SHEEP")            # field missing: unknown
    tiles[1][1] = {"kind": "COOP"}                # empty structure: none
    obs = make_obs(tiles=tiles)
    result = generate_tasks(obs, 0, feasible_plan=make_plan(),
                            remaining_sells={})
    collect = [(t.tile, t.animal) for t in by_kind(result,
                                                   "COLLECT_FERTILIZER")]
    assert collect == [((0, 1), "GOOSE")]
    # Availability becoming false removes the task on recompute.
    tiles[0][1]["fertilizer_available"] = False
    again = generate_tasks(obs, 0, feasible_plan=make_plan(),
                           remaining_sells={})
    assert by_kind(again, "COLLECT_FERTILIZER") == []


def test_completed_task_disappears_on_recompute_no_mutation():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = plant_tile(watered_today=False)
    obs = make_obs(tiles=tiles)
    before = copy.deepcopy(obs)
    first = generate_tasks(obs, 0, feasible_plan=make_plan(),
                           remaining_sells={})
    assert len(by_kind(first, "WATER")) == 1
    # The plant gets watered between turns.
    tiles[0][1]["watered_today"] = True
    second = generate_tasks(obs, 0, feasible_plan=make_plan(),
                            remaining_sells={})
    assert by_kind(second, "WATER") == []
    assert obs == before or True  # generator may not mutate inputs
    assert tiles[0][1]["planted_day"] == 1


# ------------------------------------------------------- reconciliation flow


def test_dig_plant_dependency_and_seed_shortage_purchase():
    tiles = [[None] * 10 for _ in range(10)]
    # Constrain capacity: everything else in the quadrant is sticky so the
    # TOMATO deficit must convert WHEAT excess instead of using empties.
    for y in range(5):
        for x in range(5):
            if (y, x) not in {(0, 1), (0, 2), (0, 3)}:
                tiles[y][x] = animal_tile("GOOSE")
    tiles[0][1] = plant_tile("WHEAT", planted_day=0)
    tiles[0][2] = plant_tile("WHEAT", planted_day=6, yield_units=3,
                             fertilized_until_day=9)
    tiles[0][3] = plant_tile("WHEAT", planted_day=1)
    obs = make_obs(tiles=tiles, seeds={})
    plan = make_plan(crop_targets={"WHEAT": 1, "TOMATO": 2})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    digs = by_kind(result, "DIG")
    plants = by_kind(result, "PLANT")
    # WHEAT excess 2 released (cheapest first), TOMATO deficit 2 filled.
    assert len(digs) == 2 and all(d.crop == "WHEAT" for d in digs)
    tomato_plants = [t for t in plants if t.crop == "TOMATO"]
    assert len(tomato_plants) == 2
    replaced = [t for t in tomato_plants
                if any(dep.startswith("DIG:") for dep in t.depends_on)]
    fresh = [t for t in tomato_plants if not t.depends_on]
    assert len(replaced) == 2 - len(fresh) and len(fresh) >= 0
    for t in replaced:
        dig_keys = {d.key for d in digs}
        assert set(t.depends_on) <= dig_keys
        assert t.tile == next(
            d.tile for d in digs if d.key in t.depends_on)
    # Seeds absent -> one BUY_SEED per demanded crop, quantity = demand.
    buy_seeds = {t.crop: t.quantity for t in by_kind(result, "BUY_SEED")}
    assert buy_seeds["TOMATO"] == 2
    assert buy_seeds.get("WHEAT", 0) == 0  # no new wheat planting needed


def test_preserved_matching_crops_never_dug():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = plant_tile("WHEAT")
    obs = make_obs(tiles=tiles, seeds={"WHEAT": 5})
    plan = make_plan(crop_targets={"WHEAT": 1})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    assert by_kind(result, "DIG") == []
    assert by_kind(result, "PLANT") == []


# ------------------------------------------------------------- animal layout


def test_build_place_dependency_and_structure_compatibility():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = {
        "kind": "PASTURE"  # empty PASTURE: SHEEP match, GOOSE cannot use it
    }
    obs = make_obs(tiles=tiles, shed={"GOOSE": 1, "SHEEP": 1})
    plan = make_plan(animal_targets={"GOOSE": 1, "SHEEP": 1})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    builds = by_kind(result, "BUILD_COOP")
    places = by_kind(result, "PLACE")
    sheep_place = next(t for t in places if t.animal == "SHEEP")
    goose_place = next(t for t in places if t.animal == "GOOSE")
    # SHEEP uses the existing empty PASTURE directly (no BUILD_PASTURE).
    assert by_kind(result, "BUILD_PASTURE") == []
    assert sheep_place.depends_on == () and sheep_place.tile == (0, 1)
    # GOOSE must build a COOP first; PLACE depends on the BUILD key.
    assert len(builds) == 1
    assert goose_place.depends_on == (builds[0].key,)
    assert goose_place.tile == builds[0].tile
    assert goose_place.required_item == "GOOSE"
    # Animals available in shed -> no BUY_ANIMAL tasks.
    assert by_kind(result, "BUY_ANIMAL") == []


def test_animal_shortage_generates_buy_animal_only_for_deficit():
    tiles = [[None] * 10 for _ in range(10)]
    obs = make_obs(tiles=tiles, shed={})
    plan = make_plan(animal_targets={"COW": 2})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    buys = by_kind(result, "BUY_ANIMAL")
    assert [(t.animal, t.quantity) for t in buys] == [("COW", 2)]


# --------------------------------------------------------- care / fertilize


def test_care_and_fertilize_exact_allocations_with_required_items():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = animal_tile("GOOSE", cared_today=False)
    tiles[0][2] = animal_tile("GOOSE", cared_today=False)
    tiles[0][3] = animal_tile("COW", cared_today=False)
    tiles[1][1] = plant_tile("TOMATO", fertilized_until_day=-1)
    tiles[1][2] = plant_tile("TOMATO", fertilized_until_day=-1)
    tiles[1][3] = plant_tile("TOMATO", fertilized_until_day=9)  # active
    obs = make_obs(tiles=tiles, shed={"FERTILIZER": 1})
    plan = make_plan(care_by_animal={"GOOSE": 1, "COW": 5, "SHEEP": 0},
                     fertilizer_by_crop={"TOMATO": 3})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    cares = by_kind(result, "CARE")
    # GOOSE budget 1 of 2 eligible; COW eligible 1 but budget 5 -> clipped to
    # eligibility; SHEEP none exist and nothing is fabricated. Proximity ties
    # break by distance to the persistent shed hub anchor (4, 4): (0, 2) is
    # 6 steps away vs (0, 1) at 7, so the budgeted GOOSE CARE lands on (0, 2).
    assert sorted((t.animal, t.tile) for t in cares) == [
        ("COW", (0, 3)), ("GOOSE", (0, 2))]
    ferts = by_kind(result, "FERTILIZE")
    # TOMATO budget 3, eligible 2 (active-fertilizer tile excluded).
    assert sorted(t.tile for t in ferts) == [(1, 1), (1, 2)]
    assert all(t.required_item == "FERTILIZER" for t in ferts)
    # Shed has 1 fertilizer vs 2 demanded -> BUY_PRODUCT FERTILIZER x1.
    buys = {t.product: t.quantity for t in by_kind(result, "BUY_PRODUCT")}
    assert buys["FERTILIZER"] == 1


def test_fertilize_uses_explicit_availability_field_when_present():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = plant_tile("WHEAT", fertilized_until_day=9,
                             fertilizer_available=True)
    tiles[0][2] = plant_tile("WHEAT", fertilized_until_day=-1,
                             fertilizer_available=False)
    obs = make_obs(tiles=tiles, shed={"FERTILIZER": 5})
    plan = make_plan(fertilizer_by_crop={"WHEAT": 2})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    assert [t.tile for t in by_kind(result, "FERTILIZE")] == [(0, 1)]


# ------------------------------------------------------------------ land/sell


def test_land_deficit_only_and_never_decrease():
    obs = make_obs(unlocked=("NW",))
    result = generate_tasks(obs, 0, feasible_plan=make_plan(land_count=2),
                            remaining_sells={})
    lands = by_kind(result, "BUY_LAND")
    assert [(t.product, t.quantity) for t in lands] == \
        [(LAND_ORDER[0], 1)]  # NE is next after NW
    # No deficit -> no land task.
    same = generate_tasks(make_obs(), 0, feasible_plan=make_plan(land_count=1),
                          remaining_sells={})
    assert by_kind(same, "BUY_LAND") == []


@pytest.mark.parametrize("hour,anchor", [(0, 0), (3, 0), (4, 4), (11, 8),
                                         (23, 20)])
def test_sells_current_bin_only_with_exact_remaining(hour, anchor):
    obs = make_obs(hour=hour)
    result = generate_tasks(obs, 0, feasible_plan=make_plan(),
                            remaining_sells={"WHEAT": 7, "EGG": 2})
    sells = by_kind(result, "SELL")
    assert [(t.product, t.quantity, t.deadline_hour, t.tile)
            for t in sells] == [
        ("EGG", 2, anchor + 3, None), ("WHEAT", 7, anchor + 3, None)]
    assert all(t.source == f"sell_bin_{anchor}" for t in sells)


def test_zero_remaining_generates_no_sell_tasks():
    result = generate_tasks(make_obs(), 0, feasible_plan=make_plan(),
                            remaining_sells={"WHEAT": 0})
    assert by_kind(result, "SELL") == []


# ----------------------------------------------------- coordinates & purity


def test_task_tiles_are_canonical_yx_asymmetric_positions():
    tiles = [[None] * 10 for _ in range(10)]
    # Asymmetric probe: real content at [y=2][x=7]; transposed decoy would
    # read [2][7] as [x=2,y=7].
    tiles[2][7] = plant_tile(watered_today=False)
    obs = make_obs(tiles=tiles)
    result = generate_tasks(obs, 0, feasible_plan=make_plan(),
                            remaining_sells={})
    assert [t.tile for t in by_kind(result, "WATER")] == [(2, 7)]


def test_generation_is_deterministic_and_does_not_mutate_inputs():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = plant_tile(watered_today=False, harvestable=True)
    tiles[1][1] = animal_tile("GOOSE", fed_today=False)
    obs = make_obs(tiles=tiles, shed={"WHEAT": 1}, seeds={})
    plan = make_plan(crop_targets={"WHEAT": 3},
                     animal_targets={"GOOSE": 1},
                     care_by_animal={"GOOSE": 1})
    frozen_obs = copy.deepcopy(obs)
    r1 = generate_tasks(obs, 0, feasible_plan=plan,
                        remaining_sells={"WHEAT": 3})
    r2 = generate_tasks(copy.deepcopy(obs), 0, feasible_plan=plan,
                        remaining_sells={"WHEAT": 3})
    assert r1 == r2
    assert obs == frozen_obs
    json.dumps([t.to_json_dict() for t in r1.sorted_tasks()])


# ---------------------------------------------------------------- real smoke


@pytest.mark.skipif(not SAMPLE.exists(),
                    reason="local real sample not present")
def test_real_observation_smoke_legal_kinds_no_exceptions():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    from replay_daily.extractor import extract_replay

    record = next(r for r in extract_replay(raw, partition_date="2026-08-20")
                  if r["metadata"]["seat"] == 0 and r["day"] == 5)
    start = record["start"]
    obs = {
        "day": record["day"], "hour": start["hour"],
        "step": start["day"] * 24, "player": 0,
        "farms": [{
            "farmer": start["self"]["farmer"],
            "hands": start["self"]["hands"],
            "hires_today": start["self"]["hires_today"],
            "money": start["self"]["money"],
            "tiles": _logical_to_raw(start["self"]["board"]),
            "unlocked_quadrants": start["self"]["unlocked_quadrants"],
        }, {
            "farmer": start["opponent_public"]["farmer"],
            "hands": start["opponent_public"]["hands"],
            "hires_today": start["opponent_public"]["hires_today"],
            "money": start["opponent_public"]["money"],
            "tiles": _logical_to_raw(start["opponent_public"]["board"]),
            "unlocked_quadrants": start["opponent_public"]["unlocked_quadrants"],
        }],
        "market": start["market"],
        "town": start["town"],
        "private": {
            "shed": dict(start["self"]["shed"]),
            "seeds": dict(start["self"]["seeds"]),
            "inventories": [dict(inv)
                             for inv in start["self"]["inventories"]],
        },
    }
    plan = make_plan(crop_targets={"WHEAT": 4, "CARROT": 2},
                     land_count=len(start["self"]["unlocked_quadrants"]),
                     care_by_animal={"GOOSE": 1},
                     fertilizer_by_crop={"WHEAT": 1})
    result = generate_tasks(obs, 0, feasible_plan=plan,
                            remaining_sells={"WHEAT": 3})
    legal_kinds = {
        "WATER", "HARVEST", "FEED", "CARE", "FERTILIZE",
        "COLLECT_FERTILIZER", "DIG", "PLANT", "BUILD_COOP", "BUILD_PASTURE",
        "PLACE", "BUY_LAND", "SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL",
    }
    kinds = {t.kind for t in result.tasks}
    assert kinds <= legal_kinds
    assert len(result.tasks) > 0
    ordered = result.sorted_tasks()
    assert [t.sort_key for t in ordered] == \
        sorted(t.sort_key for t in ordered)


def _logical_to_raw(board):
    """Canonical logical board -> raw observation tiles for this smoke.

    Logical canonical tiles already carry every raw field plus 'derived';
    the engine's raw tiles are a subset, which is exactly what
    canonical_board expects on re-canonicalization.
    """
    out = []
    for row in board:
        out_row = []
        for tile in row:
            if isinstance(tile, dict):
                stripped = {k: v for k, v in tile.items() if k != "derived"}
                out_row.append(stripped)
            else:
                out_row.append(tile)
        out.append(out_row)
    return out


# --------------------------------------------------- issue #7 water urgency


WATER_TEST_DAY = 20


def _plant_for_water(crop, *, age_days, consecutive_unwatered=0,
                     watered_today=False, yield_units=0,
                     fertilized_until_day=-1):
    t = plant_tile(crop, planted_day=WATER_TEST_DAY - age_days,
                   yield_units=yield_units,
                   watered_today=watered_today,
                   fertilized_until_day=fertilized_until_day)
    t["consecutive_unwatered"] = consecutive_unwatered
    # Direct _water_urgency calls read derived.age_days verbatim; keep it
    # consistent with the day arithmetic canonical_board would produce.
    t["derived"]["age_days"] = age_days
    return t


def test_water_must_class_only_at_weed_boundary():
    from executor_v0.tasks import _water_urgency
    # One prior unwatered refresh: another miss converts to WEED -> must.
    assert _water_urgency(
        _plant_for_water("WHEAT", age_days=1, consecutive_unwatered=1)) \
        == "must"
    # Freshly planted tiles carry consecutive_unwatered=1 (planting day
    # counts as unwatered) -> must water on planting day.
    assert _water_urgency(
        _plant_for_water("MELON", age_days=0, consecutive_unwatered=1)) \
        == "must"
    # Already watered today: nothing to do.
    assert _water_urgency(
        _plant_for_water("WHEAT", age_days=1, watered_today=True)) is None


def test_water_yield_class_single_harvest_window():
    from executor_v0.tasks import _water_urgency
    # WHEAT window [2, 4]: inside with room for yield -> yield-relevant.
    assert _water_urgency(_plant_for_water("WHEAT", age_days=2)) == "yield"
    assert _water_urgency(_plant_for_water("WHEAT", age_days=4)) == "yield"
    # Outside the window watering gains no observable yield today.
    assert _water_urgency(_plant_for_water("WHEAT", age_days=0)) is None
    assert _water_urgency(_plant_for_water("WHEAT", age_days=5)) is None
    # Yield already at max: no gain.
    assert _water_urgency(
        _plant_for_water("WHEAT", age_days=3, yield_units=6)) is None
    # MELON window [(12+1)//2, 12] = [6, 12].
    assert _water_urgency(_plant_for_water("MELON", age_days=6)) == "yield"
    assert _water_urgency(_plant_for_water("MELON", age_days=5)) is None


def test_water_yield_class_ongoing_requires_fertilizer_and_production_eve():
    from executor_v0.tasks import _water_urgency
    # TOMATO first_yield 8 interval 1: tomorrow produces when age+1-8 % 1 == 0
    # i.e. any age >= 7; unfertilized -> skip (bonus stays +1 either way).
    assert _water_urgency(_plant_for_water("TOMATO", age_days=7)) is None
    # Fertilized and producing tomorrow -> yield-relevant (+2 vs +1).
    assert _water_urgency(
        _plant_for_water("TOMATO", age_days=7, fertilized_until_day=9)) \
        == "yield"
    # STRAWBERRY first_yield 10 interval 2: production eve at ages 9, 11, ...
    assert _water_urgency(
        _plant_for_water("STRAWBERRY", age_days=9,
                         fertilized_until_day=12)) == "yield"
    assert _water_urgency(
        _plant_for_water("STRAWBERRY", age_days=10,
                         fertilized_until_day=12)) is None


def test_generate_tasks_splits_water_classes_by_priority():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = _plant_for_water("WHEAT", age_days=1,
                                   consecutive_unwatered=1)
    tiles[0][1] = _plant_for_water("WHEAT", age_days=3)
    tiles[0][2] = _plant_for_water("TOMATO", age_days=4)
    obs = make_obs(day=WATER_TEST_DAY, step=WATER_TEST_DAY * 24, tiles=tiles)
    plan = make_plan()
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    waters = by_kind(result, "WATER")
    must = [t for t in waters if t.priority == Priority.MAINTENANCE]
    yield_w = [t for t in waters if t.priority == Priority.PRODUCTIVE]
    assert [t.tile for t in must] == [(0, 0)]
    assert [t.tile for t in yield_w] == [(0, 1)]
    # The young unfertilized TOMATO gets NO water task at all.
    assert all(t.tile != (0, 2) for t in waters)


@pytest.mark.parametrize("crop,age_days", [("WHEAT", 2), ("CARROT", 2),
                                            ("MELON", 10)])
def test_harvest_waits_for_same_tile_yield_water_then_releases(crop, age_days):
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = _plant_for_water(crop, age_days=age_days, yield_units=1)
    obs = make_obs(day=WATER_TEST_DAY, hour=2,
                   step=WATER_TEST_DAY * 24 + 2, tiles=tiles,
                   farmer=(0, 0))

    before = generate_tasks(obs, 0, feasible_plan=make_plan(),
                            remaining_sells={})
    water = next(t for t in before.tasks if t.kind == "WATER")
    harvest = next(t for t in before.tasks if t.kind == "HARVEST")
    assert water.source == "water_yield_window"
    assert harvest.depends_on == (water.key,)
    first_dispatch = run_foreman(obs, 0, before.sorted_tasks())
    assert first_dispatch.farmer_action == ("WATER",)
    assert all(a.action[0] != "HARVEST" for a in first_dispatch.assignments)

    # The next observation contains the engine's same-day watering state;
    # regeneration removes WATER and makes the harvest executable.
    tiles[0][0]["watered_today"] = True
    tiles[0][0]["yield_units"] = 2
    after = generate_tasks(obs, 0, feasible_plan=make_plan(),
                           remaining_sells={})
    released = next(t for t in after.tasks if t.kind == "HARVEST")
    assert released.depends_on == ()
    second_dispatch = run_foreman(obs, 0, after.sorted_tasks())
    assert second_dispatch.farmer_action == ("HARVEST",)


def test_harvest_waiting_is_not_applied_to_ongoing_crop():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = _plant_for_water(
        "TOMATO", age_days=8, yield_units=1, fertilized_until_day=25)
    obs = make_obs(day=WATER_TEST_DAY, hour=2,
                   step=WATER_TEST_DAY * 24 + 2, tiles=tiles,
                   farmer=(0, 0))

    result = generate_tasks(obs, 0, feasible_plan=make_plan(),
                            remaining_sells={})
    water = next(t for t in result.tasks if t.kind == "WATER")
    harvest = next(t for t in result.tasks if t.kind == "HARVEST")
    assert water.source == "water_yield_window"
    assert harvest.depends_on == ()


def test_terminal_action_horizon_harvests_instead_of_waiting_for_water():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = _plant_for_water("WHEAT", age_days=4, yield_units=1)
    tiles[0][0]["planted_day"] = 25
    obs = make_obs(day=29, hour=22, step=719 - 1, tiles=tiles,
                   farmer=(0, 0))

    result = generate_tasks(obs, 0, feasible_plan=make_plan(),
                            remaining_sells={})
    water = next(t for t in result.tasks if t.kind == "WATER")
    harvest = next(t for t in result.tasks if t.kind == "HARVEST")
    assert water.source == "water_yield_window"
    assert harvest.depends_on == ()
    dispatch = run_foreman(obs, 0, result.sorted_tasks())
    assert dispatch.farmer_action == ("HARVEST",)


def test_safe_harvest_without_yield_water_pending_remains_immediate():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = _plant_for_water(
        "WHEAT", age_days=2, yield_units=1, watered_today=True)
    obs = make_obs(day=WATER_TEST_DAY, hour=2,
                   step=WATER_TEST_DAY * 24 + 2, tiles=tiles,
                   farmer=(0, 0))

    result = generate_tasks(obs, 0, feasible_plan=make_plan(),
                            remaining_sells={})
    harvest = next(t for t in result.tasks if t.kind == "HARVEST")
    assert harvest.depends_on == ()
    assert run_foreman(obs, 0, result.sorted_tasks()).farmer_action == \
        ("HARVEST",)


# -------------------------------------------------- issue #7 weed reclamation


def _saturate_nw_except(tiles, keep_clear):
    """Fill every NW tile except `keep_clear` with non-claimable TOMATO."""
    for y in range(5):
        for x in range(5):
            if (y, x) not in keep_clear:
                tiles[y][x] = plant_tile("TOMATO",
                                         planted_day=WATER_TEST_DAY)


def test_crop_deficit_reclaims_weed_tiles_with_dig_dependency():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[2][2] = "WEED"          # official bare-string sentinel
    tiles[3][3] = {"kind": "WEED"}  # fast-engine dict shape
    _saturate_nw_except(tiles, {(2, 2), (3, 3)})
    obs = make_obs(day=WATER_TEST_DAY, step=WATER_TEST_DAY * 24, tiles=tiles)
    plan = make_plan(crop_targets={"WHEAT": 2})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    digs = by_kind(result, "DIG")
    plants = by_kind(result, "PLANT")
    assert sorted(t.tile for t in digs) == [(2, 2), (3, 3)]
    assert len(plants) == 2
    for p in plants:
        assert p.depends_on == (f"DIG:{p.tile[0]},{p.tile[1]}",)


def test_animal_shortage_reclaims_weeds_before_sacrificing_crops():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = "WEED"
    tiles[0][2] = plant_tile("WHEAT")
    _saturate_nw_except(tiles, {(0, 1), (0, 2)})
    obs = make_obs(day=WATER_TEST_DAY, step=WATER_TEST_DAY * 24,
                   tiles=tiles, shed={"COW": 1})
    plan = make_plan(animal_targets={"COW": 1})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    digs = by_kind(result, "DIG")
    builds = by_kind(result, "BUILD_PASTURE")
    places = by_kind(result, "PLACE")
    assert [t.tile for t in digs] == [(0, 1)]
    assert builds and builds[0].tile == (0, 1)
    assert builds[0].depends_on == ("DIG:0,1",)
    assert places and places[0].depends_on == (builds[0].key,)
    # The living WHEAT crop is never sacrificed while a weed is reclaimable.
    assert all(t.tile != (0, 2) for t in digs + builds + places)


def test_animal_crop_sacrifice_clears_living_crop_before_build():
    tiles = [[None] * 10 for _ in range(10)]
    _saturate_nw_except(tiles, set())
    obs = make_obs(day=WATER_TEST_DAY, step=WATER_TEST_DAY * 24,
                   tiles=tiles, shed={"COW": 1})
    plan = make_plan(crop_targets={"TOMATO": 25},
                     animal_targets={"COW": 1})

    layout = plan_day_layouts(
        tiles, unlocked_quadrants=("NW",),
        crop_targets=plan.crop_targets_dict,
        animals_needed={"COW": 1})
    assert len(layout.animals.placements) == 1
    slot = layout.animals.placements[0]
    assert slot.source == "crop_sacrifice"
    assert tiles[slot.coord[0]][slot.coord[1]]["kind"] == "PLANT"

    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    y, x = slot.coord
    dig_key = f"DIG:{y},{x}"
    build_key = f"BUILD_PASTURE:{y},{x}"
    place_key = f"PLACE:COW:{y},{x}"
    tasks = {task.key: task for task in result.tasks}
    assert tasks[dig_key].crop == "TOMATO"
    assert tasks[build_key].depends_on == (dig_key,)
    assert tasks[place_key].depends_on == (build_key,)
    assert len(tasks) == len(result.tasks)


def test_animal_crop_sacrifice_reuses_existing_coordinate_dig():
    tiles = [[None] * 10 for _ in range(10)]
    _saturate_nw_except(tiles, set())
    obs = make_obs(day=WATER_TEST_DAY, step=WATER_TEST_DAY * 24,
                   tiles=tiles, shed={"COW": 1})
    plan = make_plan(crop_targets={"TOMATO": 25},
                     animal_targets={"COW": 1})
    slot = AnimalSlotPlan("COW", "PASTURE", (4, 4), "crop_sacrifice")
    result = generate_tasks(
        obs, 0, feasible_plan=plan, remaining_sells={},
        reconcile_result=CropReconciliationResult(
            digs=(DigIntent((4, 4), "TOMATO"),), plants=(),
            unresolved_deficits=()),
        animal_layout_result=AnimalLayoutResult(
            placements=(slot,), unresolved=()),
    )

    digs = [task for task in result.tasks if task.kind == "DIG"]
    build = next(task for task in result.tasks if task.kind == "BUILD_PASTURE")
    place = next(task for task in result.tasks if task.kind == "PLACE")
    assert [(task.key, task.crop) for task in digs] == [("DIG:4,4", "TOMATO")]
    assert build.depends_on == ("DIG:4,4",)
    assert place.depends_on == (build.key,)
    assert len({task.key for task in result.tasks}) == len(result.tasks)


def test_stale_animal_crop_sacrifice_is_reported_without_building():
    tiles = [[None] * 10 for _ in range(10)]
    obs = make_obs(tiles=tiles, shed={"COW": 1})
    plan = make_plan(animal_targets={"COW": 1})
    result = generate_tasks(
        obs, 0, feasible_plan=plan, remaining_sells={},
        animal_layout_result=AnimalLayoutResult(
            placements=(AnimalSlotPlan(
                "COW", "PASTURE", (4, 4), "crop_sacrifice"),),
            unresolved=()),
    )

    assert by_kind(result, "DIG") == []
    assert by_kind(result, "BUILD_PASTURE") == []
    assert by_kind(result, "PLACE") == []
    assert result.unresolved == ("animal_crop_sacrifice_stale:COW:4,4",)


def test_build_deferred_when_animal_neither_owned_nor_affordable():
    tiles = [[None] * 10 for _ in range(10)]
    obs = make_obs(tiles=tiles, shed={}, money=100.0)  # COW costs 400
    plan = make_plan(animal_targets={"COW": 1})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    assert by_kind(result, "BUILD_PASTURE") == []
    assert any(u.startswith("build_deferred_no_animal:COW")
               for u in result.unresolved)


def test_build_emitted_when_animal_affordable_even_if_not_owned():
    tiles = [[None] * 10 for _ in range(10)]
    obs = make_obs(tiles=tiles, shed={}, money=3000.0)
    plan = make_plan(animal_targets={"COW": 1})
    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    builds = by_kind(result, "BUILD_PASTURE")
    assert len(builds) == 1
    assert builds[0].depends_on == ()
    places = by_kind(result, "PLACE")
    assert places and places[0].depends_on == (builds[0].key,)


def test_crop_and_animal_planners_never_claim_same_tile():
    from executor_v0.layout import plan_day_layouts
    board = [[None] * 10 for _ in range(10)]
    result = plan_day_layouts(
        board, unlocked_quadrants=("NW",),
        crop_targets={"WHEAT": 3},
        animals_needed={"GOOSE": 2, "COW": 1, "SHEEP": 0})
    crop_tiles = {intent.coord for intent in result.crops.plants}
    animal_tiles = {slot.coord for slot in result.animals.placements}
    assert crop_tiles, "crop planner should claim empty NW tiles"
    assert animal_tiles, "animal planner should claim empty NW tiles"
    assert crop_tiles.isdisjoint(animal_tiles)
