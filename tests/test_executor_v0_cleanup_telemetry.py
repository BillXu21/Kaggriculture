"""Bounded aggregate telemetry for PASS-only cleanup."""

import copy

from executor_v0.agent import AgentConfig, ExecutorAgent
from executor_v0.manager import FixedPlanProvider
from executor_v0.plan import DailyPlan


def plant():
    return {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 3,
        "yield_units": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "fertilized_until_day": -1,
        "max_lifespan_step": -1,
    }


def empty_plan():
    crops = {name: 0 for name in
             ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}
    animals = {name: 0 for name in ("GOOSE", "COW", "SHEEP")}
    products = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER")
    return DailyPlan.create(
        crop_targets=crops,
        animal_targets=animals,
        land_count=1,
        fertilizer_by_crop=crops,
        care_by_animal=animals,
        sell_quantities={product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)}
                         for product in products},
    )


def make_obs(*, hour=2, farmer=(0, 0), hands=(), tiles=None):
    tiles = tiles if tiles is not None else [[None] * 10 for _ in range(10)]
    farm = {
        "farmer": list(farmer),
        "hands": [list(position) for position in hands],
        "hires_today": len(hands),
        "money": 3000.0,
        "tiles": tiles,
        "unlocked_quadrants": ["NW", "NE", "SW", "SE"],
    }
    return {
        "day": 3,
        "hour": hour,
        "step": 3 * 24 + hour,
        "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {
            "shed": {},
            "seeds": {},
            "inventories": [{} for _ in range(1 + len(hands))],
        },
    }


def test_cleanup_aggregates_count_each_action_class_and_replacement_rate():
    first_tiles = [[None] * 10 for _ in range(10)]
    first_tiles[0][0] = "WEED"
    first_tiles[0][2] = plant()
    second_tiles = [[None] * 10 for _ in range(10)]
    second_tiles[0][0] = plant()
    agent = ExecutorAgent(
        FixedPlanProvider(empty_plan()), seat=0,
        config=AgentConfig(optional_idle_cleanup=True))

    first = agent(make_obs(farmer=(0, 0), hands=((1, 0),),
                           tiles=first_tiles))
    second = agent(make_obs(hour=3, farmer=(0, 0), hands=((9, 9),),
                            tiles=second_tiles))
    metrics = agent.diagnostics_json()["cleanup_metrics"]

    assert first == {
        "farmer": ["DIG"], "hands": [["EAST"]], "market": []}
    assert second == {
        "farmer": ["WATER"], "hands": [["PASS"]], "market": []}
    assert metrics == {
        "baseline_pass_worker_actions": 4,
        "cleanup_replacements": 3,
        "weed_dig_cleanup_interactions": 1,
        "optional_water_cleanup_interactions": 1,
        "cleanup_movement_actions": 1,
        "remaining_pass_worker_actions": 1,
        "normal_non_pass_actions_changed": 0,
        "cleanup_replacement_rate": 0.75,
    }


def test_disabled_cleanup_still_reports_baseline_and_remaining_passes():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = "WEED"
    agent = ExecutorAgent(FixedPlanProvider(empty_plan()), seat=0)

    assert agent(make_obs(tiles=tiles))["farmer"] == ["PASS"]
    assert agent.diagnostics_json()["cleanup_metrics"] == {
        "baseline_pass_worker_actions": 1,
        "cleanup_replacements": 0,
        "weed_dig_cleanup_interactions": 0,
        "optional_water_cleanup_interactions": 0,
        "cleanup_movement_actions": 0,
        "remaining_pass_worker_actions": 1,
        "normal_non_pass_actions_changed": 0,
        "cleanup_replacement_rate": 0.0,
    }
