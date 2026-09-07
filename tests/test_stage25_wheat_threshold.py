"""Focused opt-in wheat threshold behavior and forecast tests."""

from executor_v0.tasks import Priority, generate_tasks
from executor_v0.upkeep import (
    FINAL_ACTIONABLE_STEP,
    fertilizer_extra_units,
    wheat_harvest_eligibility,
)
from test_executor_v0_tasks import (
    _plant_for_water,
    animal_tile,
    make_obs,
    make_plan,
)


def _fast_pass():
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _fast_action(op, argument=None, market=None):
    return {"farmer": [op] + ([] if argument is None else [argument]),
            "hands": [], "market": market or []}


def _result(*, age=3, yield_units=1, watered_today=False, step=None,
            crop="WHEAT", max_lifespan_step=-1, extra_tiles=(), plan=None):
    day = 20
    if step is None:
        step = day * 24
    tiles = [[None] * 10 for _ in range(10)]
    tile = _plant_for_water(crop, age_days=age, yield_units=yield_units,
                            watered_today=watered_today)
    tile["max_lifespan_step"] = max_lifespan_step
    tiles[0][0] = tile
    for coord, extra in extra_tiles:
        tiles[coord[0]][coord[1]] = extra
    obs = make_obs(day=day, step=step, tiles=tiles, farmer=(0, 0))
    return generate_tasks(obs, 0, feasible_plan=plan or make_plan(),
                          remaining_sells={}, wheat_harvest_threshold=True)


def test_flag_off_preserves_existing_wheat_harvest_behavior():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = _plant_for_water("WHEAT", age_days=3, yield_units=1)
    obs = make_obs(day=20, step=480, tiles=tiles, farmer=(0, 0))
    old = generate_tasks(obs, 0, feasible_plan=make_plan(), remaining_sells={})
    new = generate_tasks(obs, 0, feasible_plan=make_plan(), remaining_sells={},
                         wheat_harvest_threshold=False)
    assert [t.to_json_dict() for t in old.tasks] == [
        t.to_json_dict() for t in new.tasks]
    assert new.diagnostics == ()


def test_subthreshold_wheat_defers_but_keeps_useful_water():
    result = _result(age=3, yield_units=1)
    assert not any(t.kind == "HARVEST" for t in result.tasks)
    water = next(t for t in result.tasks if t.kind == "WATER")
    assert water.priority == Priority.PRODUCTIVE
    assert any("wheat_harvest:0,0:deferred:future_growth" == item
               for item in result.diagnostics)


def test_threshold_wheat_is_ordinary_productive_harvest():
    for yield_units in (3, 4):
        result = _result(age=3, yield_units=yield_units)
        harvest = next(t for t in result.tasks if t.kind == "HARVEST")
        assert harvest.priority == Priority.PRODUCTIVE
        assert harvest.source == "mechanical"
        assert "wheat_harvest:0,0:eligible:threshold_met" in result.diagnostics


def test_subthreshold_wheat_is_allowed_when_no_growth_remains():
    result = _result(age=5, yield_units=1, watered_today=False)
    harvest = next(t for t in result.tasks if t.kind == "HARVEST")
    assert harvest.priority == Priority.PRODUCTIVE
    assert "wheat_harvest:0,0:eligible:no_further_growth" in result.diagnostics


def test_expiry_and_terminal_boundaries_are_inclusive_exceptions():
    before = _result(age=3, yield_units=1, step=98,
                     max_lifespan_step=100)
    at_boundary = _result(age=3, yield_units=1, step=99,
                          max_lifespan_step=100)
    assert not any(t.kind == "HARVEST" for t in before.tasks)
    assert any(t.kind == "HARVEST" for t in at_boundary.tasks)
    assert "wheat_harvest:0,0:eligible:expiry" in at_boundary.diagnostics

    terminal_before = _result(age=3, yield_units=1, step=FINAL_ACTIONABLE_STEP - 1)
    terminal = _result(age=3, yield_units=1, step=FINAL_ACTIONABLE_STEP)
    assert not any(t.kind == "HARVEST" for t in terminal_before.tasks)
    assert any(t.kind == "HARVEST" for t in terminal.tasks)
    assert "wheat_harvest:0,0:eligible:terminal_horizon" in terminal.diagnostics


def test_manager_requested_removal_and_feed_buy_survive_threshold():
    tiles = [[None] * 10 for _ in range(10)]
    for y in range(5):
        for x in range(5):
            tiles[y][x] = animal_tile("GOOSE")
    tiles[0][0] = _plant_for_water("WHEAT", age_days=3, yield_units=1)
    cow = animal_tile("COW", fed_today=False)
    tiles[0][1] = cow
    obs = make_obs(day=20, step=480, tiles=tiles, farmer=(0, 0))
    result = generate_tasks(
        obs, 0, feasible_plan=make_plan(crop_targets={"WHEAT": 0,
                                                      "TOMATO": 1}),
        remaining_sells={}, wheat_harvest_threshold=True)
    assert any(t.kind == "DIG" and t.tile == (0, 0) for t in result.tasks)
    assert any(t.kind == "FEED" and t.tile == (0, 1) for t in result.tasks)
    assert any(t.kind == "BUY_PRODUCT" and t.product == "WHEAT"
               for t in result.tasks)


def test_other_crops_keep_existing_harvest_rules():
    result = _result(crop="CARROT", age=3, yield_units=1)
    assert any(t.kind == "HARVEST" for t in result.tasks)


def test_threshold_forecast_crosses_at_different_times_and_stops():
    tile = _plant_for_water("WHEAT", age_days=3, yield_units=2)
    assert fertilizer_extra_units(tile, 20) == 1
    assert fertilizer_extra_units(tile, 20,
                                 wheat_harvest_threshold=True) == 1
    # At yield 2, treatment reaches 4 while untreated stops at the first
    # threshold crossing at 3; no post-harvest watering is counted.
    capped = _plant_for_water("WHEAT", age_days=3, yield_units=5)
    assert fertilizer_extra_units(capped, 20,
                                  wheat_harvest_threshold=True) == 0


def test_eligibility_helper_distinguishes_yield_from_plant_age():
    tile = _plant_for_water("WHEAT", age_days=3, yield_units=1)
    assert wheat_harvest_eligibility(tile, 20, 480) == (False, "future_growth")
    tile["yield_units"] = 3
    assert wheat_harvest_eligibility(tile, 20, 480) == (True, "threshold_met")


def test_fast_engine_scripted_wheat_lifecycle_threshold_crossing():
    """Fast-engine-only lifecycle evidence; official parity is not claimed here."""
    from fast_env import FastKaggricultureEnv

    env = FastKaggricultureEnv({"seed": 7, "weedSpawnChance": 0})
    obs = env.reset()[0]
    obs = env.step([_fast_action("PASS", market=[["BUY_SEED", "WHEAT", 1]]),
                    _fast_pass()])[0][0]
    obs = env.step([_fast_action("PLANT", "WHEAT"), _fast_pass()])[0][0]
    x, y = obs["farms"][0]["farmer"]

    # Protect planting day, then water at the first turn of days 1, 2, and 3.
    obs = env.step([_fast_action("WATER"), _fast_pass()])[0][0]
    while obs["day"] == 0:
        obs = env.step([_fast_pass(), _fast_pass()])[0][0]
    for target_day in (1, 2):
        obs = env.step([_fast_action("WATER"), _fast_pass()])[0][0]
        while obs["day"] == target_day:
            obs = env.step([_fast_pass(), _fast_pass()])[0][0]

    tile = obs["farms"][0]["tiles"][y][x]
    assert tile["kind"] == "PLANT" and tile["yield_units"] == 2
    below = generate_tasks(
        obs, 0, feasible_plan=make_plan(crop_targets={"WHEAT": 1}),
        remaining_sells={}, wheat_harvest_threshold=True)
    assert not any(task.kind == "HARVEST" for task in below.tasks)

    obs = env.step([_fast_action("WATER"), _fast_pass()])[0][0]
    while obs["day"] == 3:
        obs = env.step([_fast_pass(), _fast_pass()])[0][0]
    tile = obs["farms"][0]["tiles"][y][x]
    assert tile["kind"] == "PLANT" and tile["yield_units"] == 3
    eligible = generate_tasks(
        obs, 0, feasible_plan=make_plan(crop_targets={"WHEAT": 1}),
        remaining_sells={}, wheat_harvest_threshold=True)
    assert any(task.kind == "HARVEST" for task in eligible.tasks)
