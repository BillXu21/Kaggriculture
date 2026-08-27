"""PASS-only weed-first idle cleanup coverage."""

import copy

import pytest

from executor_v0.agent import AgentConfig, ExecutorAgent
from executor_v0.foreman import apply_idle_cleanup, run_foreman
from executor_v0.manager import FixedPlanProvider
from executor_v0.plan import DailyPlan
from executor_v0.tasks import (
    Priority,
    Task,
    generate_optional_idle_cleanup_tasks,
)


def plant(crop="WHEAT", *, watered_today=False, consecutive_unwatered=0):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": 3,
        "yield_units": 0,
        "watered_today": watered_today,
        "consecutive_unwatered": consecutive_unwatered,
        "fertilized_until_day": -1,
        "max_lifespan_step": -1,
    }


def make_obs(*, day=3, hour=2, farmer=(0, 0), hands=(), tiles=None,
             unlocked=None, money=3000.0):
    tiles = tiles if tiles is not None else [[None] * 10 for _ in range(10)]
    farm = {
        "farmer": list(farmer),
        "hands": [list(position) for position in hands],
        "hires_today": len(hands),
        "money": money,
        "tiles": tiles,
        "unlocked_quadrants": list(unlocked or ("NW", "NE", "SW", "SE")),
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
            "shed": {},
            "seeds": {},
            "inventories": [{} for _ in range(1 + len(hands))],
        },
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


def cleanup_task(kind, coord):
    return Task(
        key=f"{kind}_CLEANUP:{coord[0]},{coord[1]}",
        kind=kind,
        priority=Priority.OPTIONAL,
        tile=coord,
        crop="WEED" if kind == "DIG" else "WHEAT",
        source="dig_cleanup" if kind == "DIG" else "water_optional_spare",
    )


def test_generation_is_weed_first_and_water_remains_strictly_filtered():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = "WEED"
    tiles[0][1] = plant()
    tiles[0][2] = plant(watered_today=True)
    tiles[0][3] = plant(consecutive_unwatered=1)
    tiles[0][5] = plant()  # NE is locked below.
    result = generate_optional_idle_cleanup_tasks(
        make_obs(tiles=tiles, unlocked=("NW",)), 0)

    assert [(task.kind, task.key, task.source) for task in result] == [
        ("DIG", "DIG_CLEANUP:0,0", "dig_cleanup"),
        ("WATER", "WATER_OPTIONAL:0,1", "water_optional_spare"),
    ]


def test_pass_becomes_underfoot_water():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = plant()
    obs = make_obs(tiles=tiles)
    normal = run_foreman(obs, 0, [])
    result = apply_idle_cleanup(obs, 0, normal,
                                generate_optional_idle_cleanup_tasks(obs, 0))

    assert result.farmer_action == ("WATER",)
    assert result.assignments[0].task_key == "WATER_OPTIONAL:0,0"


def test_pass_moves_toward_distant_optional_water():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][3] = plant()
    obs = make_obs(tiles=tiles)
    normal = run_foreman(obs, 0, [])
    result = apply_idle_cleanup(obs, 0, normal,
                                generate_optional_idle_cleanup_tasks(obs, 0))

    assert result.farmer_action == ("EAST",)
    assert result.assignments[0].task_key == "WATER_OPTIONAL:0,3"


def test_nonpass_normal_action_is_an_hard_invariant():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = "WEED"
    tiles[0][1] = plant()
    obs = make_obs(tiles=tiles)
    normal_task = Task("HARVEST:0,1", "HARVEST", Priority.PRODUCTIVE,
                       tile=(0, 1))
    normal = run_foreman(obs, 0, [normal_task])
    result = apply_idle_cleanup(obs, 0, normal,
                                generate_optional_idle_cleanup_tasks(obs, 0))

    assert normal.farmer_action != ("PASS",)
    assert result.farmer_action == normal.farmer_action
    assert result.assignments[0].task_key == normal.assignments[0].task_key


def test_two_pass_workers_claim_distinct_weeds_deterministically():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = "WEED"
    tiles[9][8] = "WEED"
    obs = make_obs(farmer=(0, 0), hands=((9, 9),), tiles=tiles)
    normal = run_foreman(obs, 0, [])
    cleanup = generate_optional_idle_cleanup_tasks(obs, 0)
    result = apply_idle_cleanup(obs, 0, normal, cleanup)

    assert [a.task_key for a in result.assignments] == [
        "DIG_CLEANUP:0,1", "DIG_CLEANUP:9,8"]
    assert result.farmer_action == ("EAST",)
    assert result.hands_actions == (("WEST",),)


def test_weed_beats_water_even_when_water_is_underfoot():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = plant()
    tiles[0][2] = "WEED"
    obs = make_obs(tiles=tiles)
    normal = run_foreman(obs, 0, [])
    result = apply_idle_cleanup(obs, 0, normal,
                                generate_optional_idle_cleanup_tasks(obs, 0))

    assert result.farmer_action == ("EAST",)
    assert result.assignments[0].task_key == "DIG_CLEANUP:0,2"


def test_normal_work_preempts_weed_and_no_pass_means_no_cleanup():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = "WEED"
    obs = make_obs(tiles=tiles)
    normal_task = Task("WATER:9,9", "WATER", Priority.MAINTENANCE,
                       tile=(9, 9))
    normal = run_foreman(obs, 0, [normal_task])
    result = apply_idle_cleanup(obs, 0, normal,
                                generate_optional_idle_cleanup_tasks(obs, 0))
    assert result.farmer_action == normal.farmer_action == ("SOUTH",)
    assert result.assignments[0].task_key == normal_task.key

    underfoot = Task("WATER:0,0", "WATER", Priority.MAINTENANCE,
                      tile=(0, 0))
    normal = run_foreman(obs, 0, [underfoot])
    result = apply_idle_cleanup(obs, 0, normal,
                                generate_optional_idle_cleanup_tasks(obs, 0))
    assert result.farmer_action == ("WATER",)
    assert result.assignments[0].task_key == underfoot.key


def test_cleanup_claims_are_not_persistent_between_turns():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = "WEED"
    obs = make_obs(tiles=tiles)
    normal = run_foreman(obs, 0, [])
    first = apply_idle_cleanup(obs, 0, normal,
                               generate_optional_idle_cleanup_tasks(obs, 0))
    second = apply_idle_cleanup(obs, 0, normal, ())

    assert first.farmer_action == ("DIG",)
    assert second.farmer_action == ("PASS",)
    assert normal.farmer_action == ("PASS",)


def test_legacy_watering_alias_enables_generalized_cleanup():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = "WEED"
    agent = ExecutorAgent(
        FixedPlanProvider(empty_plan()), seat=0,
        config=AgentConfig(optional_spare_watering=True))

    assert agent(make_obs(tiles=tiles))["farmer"] == ["DIG"]
    config = agent.diagnostics_json()["config"]
    assert config["optional_spare_watering"] is True
    assert config["optional_idle_cleanup"] is True
    assert config["optional_idle_cleanup_mode"] == "weed_first"


def test_cleanup_does_not_create_hire_debt_or_market_work():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = "WEED"
    agent = ExecutorAgent(
        FixedPlanProvider(empty_plan()), seat=0,
        config=AgentConfig(optional_idle_cleanup=True, turn_trace=True))

    action = agent(make_obs(day=3, hour=23, tiles=tiles, money=0.0))
    record = agent.diagnostics_json()["days"]["3"]

    assert action["farmer"] == ["DIG"]
    assert action["market"] == []
    assert record["hires"] == {"requested": 0, "submitted": 0,
                                "observed_max": 0}
    assert record["end_of_day_work_debt"]["all"] == []
    assert record["pending_task_turns"] == {}
    assignment = next(item for item in agent.debug_trace_turn["assignments"]
                       if item["task_key"] == "DIG_CLEANUP:0,0")
    assert assignment["source"] == "dig_cleanup"
    assert assignment["target"] == [0, 0]
    trace_assignment = next(
        item for item in record["turn_trace"][0]["assignments"]
        if item["task_key"] == "DIG_CLEANUP:0,0")
    assert trace_assignment["source"] == "dig_cleanup"
    assert trace_assignment["target"] == [0, 0]


@pytest.mark.parametrize("normal_kind", ["WATER", "HARVEST", "DIG", "CARE"])
def test_each_nonpass_normal_kind_survives_cleanup_layer(normal_kind):
    obs = make_obs()
    normal_task = Task(f"{normal_kind}:0,0", normal_kind,
                       Priority.MAINTENANCE, tile=(0, 0))
    normal = run_foreman(obs, 0, [normal_task])
    cleanup = (cleanup_task("DIG", (0, 1)),)
    result = apply_idle_cleanup(obs, 0, normal, cleanup)
    assert normal.farmer_action != ("PASS",)
    assert result.farmer_action == normal.farmer_action
