"""Focused V0.6 survival/work-debt regression tests."""

import copy

import pytest

from executor_v0.agent import AgentConfig, ExecutorAgent
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


def agent_for(daily_plan, *, config=None):
    return ExecutorAgent(FixedPlanProvider(daily_plan), seat=0, config=config)


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


def test_partial_affordable_feed_buy_is_submitted():
    tiles = empty_tiles()
    tiles[4][4] = pasture_tile("GOOSE")
    tiles[4][5] = pasture_tile("GOOSE")
    agent = agent_for(plan(animal_targets={"GOOSE": 2, "COW": 0,
                                           "SHEEP": 0}))

    # At this market inventory one WHEAT costs 125; the full two-unit order
    # costs more than the available cash, but one unit is affordable.
    action = agent(make_obs(tiles=tiles, money=125.0, shed={},
                            inventories=[{}]))

    assert action["market"] == [["BUY_PRODUCT", "WHEAT", 1]]
    assert agent.diagnostics_json()["days"]["3"]["survival"][
        "partial_feed_buys"] == 1


def test_full_shed_suppresses_survival_feed_buy_without_changing_shortage_guardrail():
    tiles = empty_tiles()
    tiles[4][4] = pasture_tile("GOOSE")
    agent = agent_for(plan(
        animal_targets={"GOOSE": 2, "COW": 0, "SHEEP": 0}, land_count=2))

    action = agent(make_obs(tiles=tiles, shed={"CARROT": 100}, unlocked=("NW",)))

    assert not any(order[:2] == ["BUY_PRODUCT", "WHEAT"]
                   for order in action["market"])
    survival = agent.diagnostics_json()["days"]["3"]["survival"]
    assert survival["feed_shortage_turns"] == 1
    assert survival["expansion_suppressed_current"] is True


def test_survival_feed_buy_is_capped_to_partial_shed_room():
    tiles = empty_tiles()
    for row, column in ((4, 4), (4, 5), (4, 6), (5, 4), (5, 5)):
        tiles[row][column] = pasture_tile("GOOSE")
    agent = agent_for(plan(animal_targets={"GOOSE": 5, "COW": 0, "SHEEP": 0}))

    action = agent(make_obs(tiles=tiles, shed={"CARROT": 98}))

    assert [order for order in action["market"]
            if order[:2] == ["BUY_PRODUCT", "WHEAT"]] == [
                ["BUY_PRODUCT", "WHEAT", 2]]
    assert agent.diagnostics_json()["days"]["3"]["survival"][
        "partial_feed_buys"] == 1


def test_carried_wheat_does_not_consume_shed_room_for_survival_buy():
    tiles = empty_tiles()
    for row, column in ((4, 4), (4, 5), (4, 6), (5, 4), (5, 5)):
        tiles[row][column] = pasture_tile("GOOSE")
    agent = agent_for(plan(animal_targets={"GOOSE": 5, "COW": 0, "SHEEP": 0}))

    action = agent(make_obs(
        tiles=tiles, shed={"CARROT": 98}, inventories=[{"WHEAT": 2}]))

    assert [order for order in action["market"]
            if order[:2] == ["BUY_PRODUCT", "WHEAT"]] == [
                ["BUY_PRODUCT", "WHEAT", 2]]


def test_missing_or_none_shed_preserves_full_affordable_survival_buy():
    tiles = empty_tiles()
    tiles[4][4] = pasture_tile("GOOSE")
    tiles[4][5] = pasture_tile("GOOSE")

    for missing in (True, False):
        agent = agent_for(plan(animal_targets={"GOOSE": 2, "COW": 0, "SHEEP": 0}))
        obs = make_obs(tiles=tiles, shed={})
        if missing:
            del obs["private"]["shed"]
        else:
            obs["private"]["shed"] = None

        action = agent(obs)

        assert [order for order in action["market"]
                if order[:2] == ["BUY_PRODUCT", "WHEAT"]] == [
                    ["BUY_PRODUCT", "WHEAT", 2]]


def test_zero_shed_capacity_is_rejected():
    with pytest.raises(ValueError, match="config.shed_capacity must be a positive integer"):
        agent_for(plan(), config=AgentConfig(shed_capacity=0))


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
    assert day["pending_task_turns"] == {"HARVEST:2,2": 1}
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
    assert diag["config"]["suppress_expansion_from_prior_debt"] is True


def _prior_debt_plan():
    return plan(
        crop_targets={"WHEAT": 1, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        animal_targets={"GOOSE": 1, "COW": 0, "SHEEP": 0},
        land_count=2,
    )


def _prior_debt_observations():
    tiles = empty_tiles()
    tiles[2][2] = plant_tile("WHEAT", yield_units=3)
    return (
        make_obs(day=3, hour=23, farmer=(0, 0), tiles=tiles),
        make_obs(day=4, hour=0, farmer=(0, 0), tiles=empty_tiles()),
    )


def test_prior_debt_toggle_false_preserves_debt_and_allows_expansion():
    agent = agent_for(
        _prior_debt_plan(),
        config=AgentConfig(
            suppress_expansion_from_prior_debt=False, turn_trace=True),
    )
    day3, day4 = _prior_debt_observations()
    agent(day3)
    action = agent(day4)
    diagnostics = agent.diagnostics_json()

    assert diagnostics["config"]["suppress_expansion_from_prior_debt"] is False
    assert diagnostics["days"]["3"]["end_of_day_work_debt"]["all"]
    assert diagnostics["days"]["3"]["next_day_expansion_suppressed"] is False
    assert diagnostics["days"]["4"]["survival"][
        "expansion_suppressed_from_prior_debt"] is False
    assert any(order[0] in ("BUY_ANIMAL", "BUY_LAND")
               for order in action["market"])
    expansion = diagnostics["days"]["4"]["turn_trace"][0]["expansion"]
    assert expansion["suppressed_from_prior"] is False
    assert "prior_day_work_debt" not in expansion["reasons"]


def test_prior_debt_toggle_false_keeps_current_feed_suppression():
    tiles = empty_tiles()
    tiles[4][4] = pasture_tile("GOOSE", consecutive_unfed=1)
    agent = agent_for(
        plan(animal_targets={"GOOSE": 2, "COW": 0, "SHEEP": 0},
             land_count=2),
        config=AgentConfig(suppress_expansion_from_prior_debt=False),
    )

    action = agent(make_obs(tiles=tiles, shed={}, inventories=[{}],
                            unlocked=("NW",)))
    diagnostics = agent.diagnostics_json()

    assert action["market"]
    assert action["market"][0][:2] == ["BUY_PRODUCT", "WHEAT"]
    assert not any(order[0] in ("BUY_ANIMAL", "BUY_LAND")
                   for order in action["market"])
    assert diagnostics["days"]["3"]["survival"][
        "expansion_suppressed_current"] is True


def test_prior_debt_keeps_land_but_suppresses_animal_expansion():
    agent = agent_for(_prior_debt_plan(), config=AgentConfig(turn_trace=True))
    day3, day4 = _prior_debt_observations()
    agent(day3)
    action = agent(make_obs(
        day=4, hour=0, money=3000.0, unlocked=("NW",),
        tiles=empty_tiles()))

    assert [order for order in action["market"] if order[0] == "BUY_LAND"] == [
        ["BUY_LAND"]]
    assert not any(order[0] == "BUY_ANIMAL" for order in action["market"])
    day = agent.diagnostics_json()["days"]["4"]
    assert day["end_of_day_work_debt"] == {
        "all": [], "survival": [], "maintenance": [],
        "productive": [], "manager": [],
    }
    assert day["survival"]["expansion_suppressed_from_prior_debt"] is True
    assert day["land_purchase"] == {
        "requested": True, "task_present": True,
        "suppressed_prior_debt": False,
        "suppressed_current_survival": False,
        "affordable_before_hires": True,
        "unaffordable_before_hires": False,
        "submitted": True,
        "land_cost": 1000.0,
        "cash_before_land": 3000.0,
    }
    assert agent.diagnostics_json()["days"]["4"]["turn_trace"][0][
        "land_purchase"]["suppressed_prior_debt"] is False


def test_current_starvation_suppresses_requested_land():
    tiles = empty_tiles()
    tiles[4][4] = pasture_tile("GOOSE", consecutive_unfed=1)
    agent = agent_for(plan(
        animal_targets={"GOOSE": 1, "COW": 0, "SHEEP": 0}, land_count=2),
        config=AgentConfig(turn_trace=True))

    action = agent(make_obs(
        unlocked=("NW",), money=10_000.0, tiles=tiles,
        inventories=[{"WHEAT": 1}]))

    assert not any(order[0] == "BUY_LAND" for order in action["market"])
    land = agent.diagnostics_json()["days"]["3"]["land_purchase"]
    assert land["requested"] is True
    assert land["task_present"] is True
    assert land["suppressed_prior_debt"] is False
    assert land["suppressed_current_survival"] is True
    assert land["submitted"] is False


def test_current_feed_shortage_buys_feed_and_suppresses_requested_land():
    tiles = empty_tiles()
    tiles[4][4] = pasture_tile("GOOSE")
    agent = agent_for(plan(
        animal_targets={"GOOSE": 1, "COW": 0, "SHEEP": 0}, land_count=2),
        config=AgentConfig(turn_trace=True))

    action = agent(make_obs(unlocked=("NW",), money=10_000.0, tiles=tiles))

    assert action["market"][0][:2] == ["BUY_PRODUCT", "WHEAT"]
    assert not any(order[0] == "BUY_LAND" for order in action["market"])
    land = agent.diagnostics_json()["days"]["3"]["land_purchase"]
    assert land["suppressed_current_survival"] is True
    assert land["submitted"] is False


def test_land_is_reserved_before_workload_hiring():
    tiles = empty_tiles()
    tiles[1][0] = plant_tile(watered_today=False)
    agent = agent_for(plan(
        crop_targets={"WHEAT": 0, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        land_count=2), config=AgentConfig(tasks_per_worker=10))

    action = agent(make_obs(unlocked=("NW",), money=1000.0, tiles=tiles))

    assert action["market"] == [["BUY_LAND"]]
    land = agent.diagnostics_json()["days"]["3"]["land_purchase"]
    assert land["affordable_before_hires"] is True
    assert land["submitted"] is True


def test_land_and_workload_hire_both_fit_in_cash():
    tiles = empty_tiles()
    tiles[1][0] = plant_tile(watered_today=False)
    agent = agent_for(plan(land_count=2), config=AgentConfig(tasks_per_worker=10))

    action = agent(make_obs(unlocked=("NW",), money=1001.0, tiles=tiles))

    assert action["market"][:2] == [["BUY_LAND"], ["HIRE"]]
    assert agent.diagnostics_json()["days"]["3"]["land_purchase"][
        "submitted"] is True


def test_unaffordable_land_does_not_reserve_cash_from_hiring():
    tiles = empty_tiles()
    tiles[1][0] = plant_tile(watered_today=False)
    agent = agent_for(plan(land_count=2), config=AgentConfig(tasks_per_worker=10))

    action = agent(make_obs(unlocked=("NW",), money=999.0, tiles=tiles))

    assert action["market"] == [["HIRE"]]
    land = agent.diagnostics_json()["days"]["3"]["land_purchase"]
    assert land["affordable_before_hires"] is False
    assert land["unaffordable_before_hires"] is True
    assert land["submitted"] is False


def test_market_cap_keeps_land_ahead_of_workload_hiring():
    tiles = empty_tiles()
    tiles[1][0] = plant_tile(watered_today=False)
    daily = plan(land_count=2, wheat_sell=1)
    agent = agent_for(daily, config=AgentConfig(
        tasks_per_worker=10, max_market_orders=2))

    action = agent(make_obs(
        unlocked=("NW",), money=1000.0, shed={"WHEAT": 1}, tiles=tiles))

    assert action["market"] == [["SELL", "WHEAT", 1], ["BUY_LAND"]]
    assert agent.diagnostics_json()["days"]["3"]["land_purchase"][
        "submitted"] is True


def test_market_queue_places_land_before_hire_and_other_buy():
    tiles = empty_tiles()
    tiles[1][0] = plant_tile(watered_today=False)
    daily = plan(
        crop_targets={"WHEAT": 2, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        land_count=2, wheat_sell=1)
    agent = agent_for(daily, config=AgentConfig(tasks_per_worker=10))

    action = agent(make_obs(
        unlocked=("NW",), money=1001.0, shed={"WHEAT": 1},
        seeds={}, tiles=tiles))

    assert action["market"] == [
        ["SELL", "WHEAT", 1], ["BUY_LAND"], ["HIRE"],
        ["BUY_SEED", "WHEAT", 1],
    ]


def test_prior_debt_toggle_does_not_change_safe_actions_or_ordinary_diagnostics():
    observations = [
        make_obs(day=3, hour=0),
        make_obs(day=3, hour=1),
    ]
    enabled = agent_for(
        plan(), config=AgentConfig(suppress_expansion_from_prior_debt=True))
    disabled = agent_for(
        plan(), config=AgentConfig(suppress_expansion_from_prior_debt=False))

    enabled_actions = [enabled(copy.deepcopy(obs)) for obs in observations]
    disabled_actions = [disabled(copy.deepcopy(obs)) for obs in observations]
    enabled_diag = enabled.diagnostics_json()
    disabled_diag = disabled.diagnostics_json()
    enabled_diag.pop("config")
    disabled_diag.pop("config")

    assert enabled_actions == disabled_actions
    assert enabled_diag == disabled_diag
