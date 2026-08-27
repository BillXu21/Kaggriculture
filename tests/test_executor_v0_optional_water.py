"""Focused coverage for the config-gated spare-capacity watering core."""

import copy

import pytest

from executor_v0.agent import AgentConfig, ExecutorAgent
from executor_v0.foreman import run_foreman
from executor_v0.manager import FixedPlanProvider
from executor_v0.plan import DailyPlan
from executor_v0.tasks import (
    Priority,
    Task,
    generate_optional_water_tasks,
)


def plant(crop="WHEAT", *, day=3, watered_today=False,
          consecutive_unwatered=0, yield_units=0,
          fertilized_until_day=-1):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": day,
        "yield_units": yield_units,
        "watered_today": watered_today,
        "consecutive_unwatered": consecutive_unwatered,
        "fertilized_until_day": fertilized_until_day,
        "max_lifespan_step": -1,
    }


def make_obs(*, day=3, hour=2, farmer=(0, 0), hands=(), tiles=None,
             unlocked=("NW", "NE", "SW", "SE"), shed=None,
             inventories=None, money=3000.0):
    tiles = tiles if tiles is not None else [[None] * 10 for _ in range(10)]
    farm = {
        "farmer": list(farmer),
        "hands": [list(position) for position in hands],
        "hires_today": len(hands),
        "money": money,
        "tiles": tiles,
        "unlocked_quadrants": list(unlocked),
    }
    return {
        "day": day,
        "hour": hour,
        "step": day * 24 + hour,
        "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {
            "shed": shed or {},
            "seeds": {},
            "inventories": inventories if inventories is not None
            else [{} for _ in range(1 + len(hands))],
        },
    }


def plan(*, crop_targets=None, land_count=1):
    crops = {name: 0 for name in (
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}
    if crop_targets is not None:
        crops.update(crop_targets)
    return DailyPlan.create(
        crop_targets=crops,
        animal_targets={name: 0 for name in ("GOOSE", "COW", "SHEEP")},
        land_count=land_count,
        fertilizer_by_crop={name: 0 for name in (
            "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
        care_by_animal={name: 0 for name in ("GOOSE", "COW", "SHEEP")},
        sell_quantities={product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)}
                         for product in (
                             "WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                             "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")},
    )


def optional_task(coord):
    return Task(
        key=f"WATER_OPTIONAL:{coord[0]},{coord[1]}",
        kind="WATER", priority=Priority.OPTIONAL, tile=coord,
        source="water_optional_spare")


def test_optional_generation_is_strictly_gated_and_filters_ineligible_tiles():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = plant()
    tiles[0][1] = plant(watered_today=True)
    tiles[0][2] = plant(consecutive_unwatered=1)
    tiles[0][3] = plant(day=1)  # WHEAT age 2: yield window.
    tiles[0][4] = {"kind": "COOP"}
    tiles[0][5] = plant()  # NE is locked below.
    tiles[5][0] = plant()  # SW is locked below.
    obs = make_obs(tiles=tiles, unlocked=("NW",))

    candidates = generate_optional_water_tasks(obs, 0)

    assert [(task.key, task.tile, task.priority, task.source)
            for task in candidates] == [
                ("WATER_OPTIONAL:0,0", (0, 0), Priority.OPTIONAL,
                 "water_optional_spare")]


def test_malformed_plant_is_not_an_optional_candidate():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = {"kind": "PLANT", "crop": "NOT_A_CROP",
                   "watered_today": False, "consecutive_unwatered": 0}
    assert generate_optional_water_tasks(make_obs(tiles=tiles), 0) == ()


@pytest.mark.parametrize("planted_day", [14, 8])
def test_melon_yield_window_is_not_optional(planted_day):
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = plant("MELON", day=planted_day)

    assert generate_optional_water_tasks(
        make_obs(day=20, tiles=tiles), 0) == ()


def test_fertilized_ongoing_production_timing_is_not_optional():
    tiles = [[None] * 10 for _ in range(10)]
    # TOMATO age 7 is production-eve; active fertilizer makes WATER yield-positive.
    tiles[0][0] = plant("TOMATO", day=3, fertilized_until_day=10)

    assert generate_optional_water_tasks(
        make_obs(day=10, tiles=tiles), 0) == ()


def test_default_off_passes_and_enabled_spare_worker_waters():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = plant()
    obs = make_obs(tiles=tiles)
    provider = FixedPlanProvider(plan(crop_targets={"WHEAT": 1}))

    default_agent = ExecutorAgent(provider, seat=0)
    enabled_agent = ExecutorAgent(
        FixedPlanProvider(plan(crop_targets={"WHEAT": 1})), seat=0,
        config=AgentConfig(optional_spare_watering=True))

    assert default_agent(obs)["farmer"] == ["PASS"]
    assert enabled_agent(obs)["farmer"] == ["WATER"]
    assert default_agent.diagnostics_json()["config"][
        "optional_spare_watering"] is False
    assert enabled_agent.diagnostics_json()["config"][
        "optional_spare_watering"] is True


@pytest.mark.parametrize(
    "kind, priority, extra",
    [
        ("WATER", Priority.MAINTENANCE, {}),
        ("WATER", Priority.PRODUCTIVE, {}),
        ("HARVEST", Priority.PRODUCTIVE, {}),
        ("FEED", Priority.MAINTENANCE, {"required_item": "WHEAT"}),
        ("COLLECT_FERTILIZER", Priority.MAINTENANCE, {}),
        ("DIG", Priority.MANAGER, {}),
        ("WATER", Priority.LOGISTICS, {}),
    ],
)
def test_every_existing_priority_class_beats_optional(kind, priority, extra):
    obs = make_obs(farmer=(0, 0), inventories=[{"WHEAT": 1}])
    normal = Task(
        key=f"{kind}:9,9", kind=kind, priority=priority, tile=(9, 9),
        **extra)

    result = run_foreman(obs, 0, [optional_task((0, 1)), normal])

    assert result.assignments[0].task_key == normal.key
    assert result.farmer_action == ("SOUTH",)


def test_optional_choice_is_nearest_and_two_workers_claim_distinct_tiles():
    obs = make_obs(
        farmer=(0, 0), hands=((9, 9),), inventories=[{}, {}])
    tasks = [optional_task((0, 1)), optional_task((9, 8))]

    result = run_foreman(obs, 0, tasks)

    assert [assignment.task_key for assignment in result.assignments] == [
        "WATER_OPTIONAL:0,1", "WATER_OPTIONAL:9,8"]
    assert result.farmer_action == ("EAST",)
    assert result.hands_actions == (("WEST",),)


def test_optional_dispatch_stays_out_of_hiring_debt_and_is_traceable():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = plant()
    agent = ExecutorAgent(
        FixedPlanProvider(plan(crop_targets={"WHEAT": 1})), seat=0,
        config=AgentConfig(optional_spare_watering=True, turn_trace=True))

    action = agent(make_obs(day=3, hour=23, tiles=tiles))
    diagnostics = agent.diagnostics_json()
    record = diagnostics["days"]["3"]
    snapshot = agent.debug_trace_turn

    assert action["farmer"] == ["WATER"]
    assert record["hires"] == {"requested": 0, "submitted": 0,
                                "observed_max": 0}
    assert record["end_of_day_work_debt"]["all"] == []
    assert record["unfinished_tasks"] == []
    assert record["missed_maintenance"] == []
    assert record["pending_task_turns"] == {}
    agent(make_obs(day=4, hour=0, tiles=tiles))
    assert record["next_day_expansion_suppressed"] is False
    assignment = next(item for item in snapshot["assignments"]
                       if item["task_key"] == "WATER_OPTIONAL:0,0")
    assert assignment["source"] == "water_optional_spare"
    assert assignment["target"] == [0, 0]
    turn_assignment = next(item for item in record["turn_trace"][0]["assignments"]
                           if item["task_key"] == "WATER_OPTIONAL:0,0")
    assert turn_assignment["source"] == "water_optional_spare"
