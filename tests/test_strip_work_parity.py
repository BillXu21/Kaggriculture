"""Presence of the experimental forecast must not alter V0 task generation."""

from copy import deepcopy

from executor_v0.plan import DailyPlan
from executor_v0.strip_work import build_strip_work_plan
from executor_v0.tasks import generate_tasks


def _plan() -> DailyPlan:
    crops = {name: 0 for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}
    animals = {name: 0 for name in ("GOOSE", "COW", "SHEEP")}
    products = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER")
    return DailyPlan.create(
        crop_targets={**crops, "WHEAT": 1},
        animal_targets=animals,
        land_count=1,
        fertilizer_by_crop=crops,
        care_by_animal=animals,
        sell_quantities={p: {h: 0 for h in (0, 4, 8, 12, 16, 20)} for p in products},
    )


def _obs():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 1,
        "yield_units": 0, "watered_today": False,
        "fertilized_until_day": -1, "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    farm = {
        "farmer": [0, 0], "hands": [], "hires_today": 0, "money": 3000.0,
        "tiles": board, "unlocked_quadrants": ["NW"],
    }
    return {
        "day": 3, "hour": 0, "step": 72, "player": 0,
        "farms": [farm, deepcopy(farm)],
        "market": {"inventory": {}, "prices": {}}, "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}, {}]},
    }


def test_strip_forecast_does_not_change_existing_task_generation():
    observation = _obs()
    plan = _plan()
    before = generate_tasks(deepcopy(observation), 0, feasible_plan=plan,
                            remaining_sells={})
    forecast = build_strip_work_plan(observation, plan)
    after = generate_tasks(deepcopy(observation), 0, feasible_plan=plan,
                           remaining_sells={})
    assert before == after
    assert observation["farms"][0]["tiles"][0][0]["watered_today"] is False
    assert forecast.acting_seat == 0
