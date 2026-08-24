"""Focused V0.6 survival/work-debt regression tests."""

import copy

from executor_v0.agent import ExecutorAgent
from executor_v0.manager import FixedPlanProvider
from executor_v0.plan import DailyPlan


PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER")


def empty_tiles():
    return [[None] * 10 for _ in range(10)]


def plant_tile(crop="WHEAT", **extra):
    tile = {
        "kind": "PLANT", "crop": crop, "planted_day": 0,
        "yield_units": 1, "max_lifespan_step": 500,
        "fertilized_until_day": -1, "consecutive_unwatered": 0,
        "watered_today": True,
    }
    tile.update(extra)
    return tile


def pasture_tile(animal="GOOSE", **extra):
    tile = {
        "kind": "PASTURE", "animal": animal, "placed_day": 0,
        "yield_units": 0, "fed_today": False, "cared_today": False,
        "consecutive_unfed": 0, "pending_care_bonus": 0,
        "fertilizer_available": False,
    }
    tile.update(extra)
    return tile


def make_obs(*, day=3, hour=0, farmer=(4, 4), hands=(), money=3000.0,
             tiles=None, shed=None, seeds=None, inventories=None,
             unlocked=("NW", "NE", "SW", "SE"), market_inventory=None,
             market_prices=None):
    tiles = tiles if tiles is not None else empty_tiles()
    farm = {
        "farmer": list(farmer),
        "hands": [list(h) for h in hands],
        "hires_today": len(hands),
        "money": money,
        "tiles": tiles,
        "unlocked_quadrants": list(unlocked),
    }
    return {
        "day": day, "hour": hour, "step": day * 24 + hour, "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "market": {
            "inventory": dict(market_inventory or {p: 100 for p in PRODUCTS}),
            "prices": dict(market_prices or {p: 10 for p in PRODUCTS}),
        },
        "town": {"unlocked_shops": []},
        "private": {
            "shed": dict(shed or {}),
            "seeds": dict(seeds or {}),
            "inventories": inventories if inventories is not None
            else [{} for _ in range(1 + len(hands))],
        },
    }


def plan(*, crop_targets=None, animal_targets=None, land_count=1,
         wheat_sell=0):
    sells = {
        product: {anchor: 0 for anchor in (0, 4, 8, 12, 16, 20)}
        for product in PRODUCTS
    }
    sells["WHEAT"][0] = wheat_sell
    return DailyPlan.create(
        crop_targets=crop_targets or {
            "WHEAT": 0, "CARROT": 0, "TOMATO": 0,
            "STRAWBERRY": 0, "MELON": 0,
        },
        animal_targets=animal_targets or {"GOOSE": 0, "COW": 0, "SHEEP": 0},
        land_count=land_count,
        fertilizer_by_crop={
            "WHEAT": 0, "CARROT": 0, "TOMATO": 0,
            "STRAWBERRY": 0, "MELON": 0,
        },
        care_by_animal={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        sell_quantities=sells,
    )


def agent_for(daily_plan):
    return ExecutorAgent(FixedPlanProvider(daily_plan), seat=0)


def test_wheat_sells_preserve_current_day_feed_reserve():
    tiles = empty_tiles()
    tiles[4][4] = pasture_tile("GOOSE")
    tiles[4][5] = pasture_tile("GOOSE")
    agent = agent_for(plan(
        animal_targets={"GOOSE": 2, "COW": 0, "SHEEP": 0},
        wheat_sell=5))
    obs = make_obs(tiles=tiles, shed={"WHEAT": 5})
    action = agent(obs)

    wheat_sells = [order for order in action["market"]
                   if order[:2] == ["SELL", "WHEAT"]]
    assert wheat_sells == [["SELL", "WHEAT", 3]]
    survival = agent.diagnostics_json()["days"]["3"]["survival"]
    assert survival["feed_reserve_protected_units"] == 2


def test_starving_animal_preempts_non_survival_tile_work():
    tiles = empty_tiles()
    tiles[4][4] = pasture_tile("GOOSE", consecutive_unfed=1)
    tiles[2][2] = plant_tile("WHEAT", yield_units=3)
    agent = agent_for(plan(
        crop_targets={"WHEAT": 1, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        animal_targets={"GOOSE": 1, "COW": 0, "SHEEP": 0}))
    obs = make_obs(tiles=tiles, inventories=[{"WHEAT": 1}])
    action = agent(obs)

    assert action["farmer"] == ["FEED"]
    survival = agent.diagnostics_json()["days"]["3"]["survival"]
    assert survival["starvation_preemption_turns"] == 1


def test_feed_shortage_buy_precedes_hiring_and_blocks_expansion():
    tiles = empty_tiles()
    tiles[4][4] = pasture_tile("GOOSE", consecutive_unfed=1)
    agent = agent_for(plan(
        animal_targets={"GOOSE": 2, "COW": 0, "SHEEP": 0},
        land_count=2))
    obs = make_obs(tiles=tiles, shed={}, inventories=[{}], money=3000.0,
                   unlocked=("NW",))
    action = agent(obs)

    assert action["market"]
    assert action["market"][0][:2] == ["BUY_PRODUCT", "WHEAT"]
    assert not any(order[0] in ("BUY_ANIMAL", "BUY_LAND")
                   for order in action["market"])
    survival = agent.diagnostics_json()["days"]["3"]["survival"]
    assert survival["expansion_suppressed_current"] is True
    assert survival["feed_shortage_turns"] == 1


def test_temporary_waiting_that_finishes_on_hour23_is_not_work_debt():
    tiles = empty_tiles()
    tiles[2][2] = plant_tile("WHEAT", yield_units=3)
    daily = plan(crop_targets={
        "WHEAT": 1, "CARROT": 0, "TOMATO": 0,
        "STRAWBERRY": 0, "MELON": 0,
    })
    agent = agent_for(daily)

    # Earlier in the day the harvest is pending while the worker travels.
    agent(make_obs(day=3, hour=22, farmer=(0, 0), tiles=tiles))
    # On the final turn the worker is underfoot and HARVEST executes before
    # refresh, so earlier waiting must not be labeled unfinished.
    action = agent(make_obs(day=3, hour=23, farmer=(2, 2), tiles=tiles))
    assert action["farmer"] == ["HARVEST"]

    day = agent.diagnostics_json()["days"]["3"]
    assert day["pending_task_turns"]
    assert day["end_of_day_work_debt"]["all"] == []
    assert day["unfinished_tasks"] == []
    assert day["unfinished_task_turns"] == {}


def test_movement_on_hour23_remains_real_work_debt_and_suppresses_next_day():
    tiles = empty_tiles()
    tiles[2][2] = plant_tile("WHEAT", yield_units=3)
    daily = plan(
        crop_targets={"WHEAT": 1, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        animal_targets={"GOOSE": 1, "COW": 0, "SHEEP": 0},
        land_count=2)
    agent = agent_for(daily)

    action = agent(make_obs(day=3, hour=23, farmer=(0, 0), tiles=tiles))
    assert action["farmer"][0] in ("SOUTH", "EAST")
    day3 = agent.diagnostics_json()["days"]["3"]
    assert any(key.startswith("HARVEST:")
               for key in day3["end_of_day_work_debt"]["all"])

    agent(make_obs(day=4, hour=0, farmer=(0, 0), tiles=tiles))
    diag = agent.diagnostics_json()
    assert diag["days"]["3"]["next_day_expansion_suppressed"] is True
    assert diag["days"]["4"]["survival"][
        "expansion_suppressed_from_prior_debt"] is True
