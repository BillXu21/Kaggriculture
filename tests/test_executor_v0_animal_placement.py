"""Regression coverage for purchased-animal placement progress (issue #25)."""

import copy

import pytest

from executor_v0.agent import AgentConfig, ExecutorAgent
from executor_v0.foreman import run_foreman
from executor_v0.layout import plan_animal_layout, tile_role
from executor_v0.manager import FixedPlanProvider
from executor_v0.plan import DailyPlan
from executor_v0.tasks import generate_tasks
from oracle.closed_loop import _executor_observation


ANIMALS = ("GOOSE", "COW", "SHEEP")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")


def make_plan(*, animals=None, crops=None):
    targets = {name: 0 for name in ANIMALS}
    targets.update(animals or {})
    crop_targets = {name: 0 for name in CROPS}
    crop_targets.update(crops or {})
    return DailyPlan.create(
        crop_targets=crop_targets,
        animal_targets=targets,
        land_count=4,
        fertilizer_by_crop={name: 0 for name in CROPS},
        care_by_animal={name: 0 for name in ANIMALS},
        sell_quantities={
            name: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)}
            for name in PRODUCTS
        },
    )


def make_obs(*, day=4, hour=0, farmer=(4, 4), hands=(), tiles=None,
             shed=None, inventories=None):
    board = tiles if tiles is not None else [[None] * 10 for _ in range(10)]
    farm = {
        "farmer": list(farmer),
        "hands": [list(position) for position in hands],
        "hires_today": len(hands),
        "money": 3000.0,
        "tiles": board,
        "unlocked_quadrants": ["NW", "NE", "SW", "SE"],
    }
    return {
        "day": day,
        "hour": hour,
        "step": day * 24 + hour,
        "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "market": {
            "inventory": {name: 10_000 for name in PRODUCTS},
            "prices": {name: 10 for name in PRODUCTS},
        },
        "town": {"unlocked_shops": []},
        "private": {
            "shed": dict(shed or {}),
            "seeds": {},
            "inventories": inventories if inventories is not None
            else [{} for _ in range(1 + len(hands))],
        },
    }


def animal_tile(animal="GOOSE", *, starving=False):
    return {
        "kind": "COOP" if animal == "GOOSE" else "PASTURE",
        "animal": animal,
        "placed_day": 0,
        "yield_units": 0,
        "fed_today": False,
        "cared_today": True,
        "consecutive_unfed": 1 if starving else 0,
        "fertilizer_available": False,
    }


def hard_water_tile():
    return {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 4,
        "yield_units": 1,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "fertilized_until_day": -1,
        "max_lifespan_step": 500,
    }


def living_wheat_tile():
    return {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 4,
        "yield_units": 0,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "fertilized_until_day": -1,
        "max_lifespan_step": 500,
    }


@pytest.mark.parametrize("zero_filled", [False, True])
def test_empty_pasture_carried_cow_traces_through_next_observed_state(zero_filled):
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {"kind": "PASTURE"}
    inventory = ({name: int(name == "COW") for name in (*PRODUCTS, *ANIMALS)}
                 if zero_filled else {"COW": 1})
    obs = make_obs(tiles=board, inventories=[inventory])
    plan = make_plan(animals={"COW": 1})

    layout = plan_animal_layout(
        board, unlocked_quadrants=obs["farms"][0]["unlocked_quadrants"],
        animals_needed={"COW": 1}, anchor=(4, 4))
    assert [(slot.animal, slot.coord, slot.source) for slot in layout.placements] == [
        ("COW", (4, 4), "empty_structure")]

    generated = generate_tasks(
        obs, 0, feasible_plan=plan, remaining_sells={},
        animal_layout_result=layout)
    place = [task for task in generated.tasks if task.kind == "PLACE"]
    assert [(task.key, task.depends_on) for task in place] == [
        ("PLACE:COW:4,4", ())]
    assert not any(task.kind in ("BUILD_PASTURE", "BUY_ANIMAL")
                   for task in generated.tasks)

    dispatched = run_foreman(obs, 0, generated.sorted_tasks())
    assert dispatched.assignments[0].task_key == "PLACE:COW:4,4"
    assert dispatched.farmer_action == ("PLACE", "COW", 1)

    next_obs = copy.deepcopy(obs)
    next_obs["hour"] += 1
    next_obs["step"] += 1
    next_obs["farms"][0]["tiles"][4][4] = animal_tile("COW")
    next_obs["private"]["inventories"][0]["COW"] = 0
    after = generate_tasks(next_obs, 0, feasible_plan=plan, remaining_sells={})
    assert not any(task.kind in ("PLACE", "BUY_ANIMAL") for task in after.tasks)


def test_purchased_cow_crop_sacrifice_completes_across_regenerated_tasks():
    board = [[None] * 10 for _ in range(10)]
    for y in range(5):
        for x in range(5):
            board[y][x] = living_wheat_tile()
    obs = make_obs(tiles=board, farmer=(4, 4), shed={})
    obs["farms"][0]["unlocked_quadrants"] = ["NW"]
    plan = make_plan(animals={"COW": 1}, crops={"WHEAT": 25})

    first = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    assert not any(task.source == "crop_sacrifice" for task in first.tasks)
    assert not any(task.kind in ("DIG", "BUILD_PASTURE", "PLACE")
                   for task in first.tasks)
    assert [(task.key, task.quantity) for task in first.tasks
            if task.kind == "BUY_ANIMAL"] == [("BUY_ANIMAL:COW", 1)]

    purchased = copy.deepcopy(obs)
    purchased["private"]["shed"]["COW"] = 1
    second = generate_tasks(purchased, 0, feasible_plan=plan, remaining_sells={})
    assert any(task.source == "crop_sacrifice" for task in second.tasks)
    assert run_foreman(purchased, 0, second.sorted_tasks()).farmer_action == (
        "DIG",)
    target = next(task.tile for task in second.tasks if task.kind == "DIG")
    assert target == (4, 4)

    dug = copy.deepcopy(purchased)
    dug["farms"][0]["tiles"][4][4] = None
    third = generate_tasks(dug, 0, feasible_plan=plan, remaining_sells={})
    assert run_foreman(dug, 0, third.sorted_tasks()).farmer_action == (
        "BUILD_PASTURE",)

    built = copy.deepcopy(dug)
    built["farms"][0]["tiles"][4][4] = {"kind": "PASTURE"}
    built["private"]["shed"]["COW"] = 0
    built["private"]["inventories"][0]["COW"] = 1
    fourth = generate_tasks(built, 0, feasible_plan=plan, remaining_sells={})
    assert run_foreman(built, 0, fourth.sorted_tasks()).farmer_action == (
        "PLACE", "COW", 1)

    placed = copy.deepcopy(built)
    placed["private"]["inventories"][0]["COW"] = 0
    placed["farms"][0]["tiles"][4][4] = animal_tile("COW")
    final = generate_tasks(placed, 0, feasible_plan=plan, remaining_sells={})
    assert sum(
        tile_role(tile) == "animal_structure"
        and tile.get("animal") == "COW"
        for row in placed["farms"][0]["tiles"] for tile in row
        if isinstance(tile, dict)
    ) == 1
    assert not any(task.kind in ("BUILD_PASTURE", "PLACE", "BUY_ANIMAL")
                   for task in final.tasks)


def test_multiple_livestock_claim_distinct_compatible_structures():
    board = [[None] * 10 for _ in range(10)]
    for coord in ((4, 4), (4, 5), (5, 4)):
        board[coord[0]][coord[1]] = {"kind": "PASTURE"}
    obs = make_obs(
        farmer=(4, 4), hands=((5, 4), (4, 5)), tiles=board,
        inventories=[{"COW": 1}, {"COW": 1}, {"SHEEP": 1}])
    result = generate_tasks(
        obs, 0, feasible_plan=make_plan(animals={"COW": 2, "SHEEP": 1}),
        remaining_sells={})

    place = [task for task in result.tasks if task.kind == "PLACE"]
    assert len(place) == 3
    assert len({task.tile for task in place}) == 3
    dispatched = run_foreman(obs, 0, result.sorted_tasks())
    claimed = [assignment.task_key for assignment in dispatched.assignments]
    assert len(set(claimed)) == 3
    assert all(action[0] == "PLACE" for action in (
        dispatched.farmer_action, *dispatched.hands_actions))


def test_carried_animal_never_returns_to_pickup_and_spare_worker_does_not_pass():
    board = [[None] * 10 for _ in range(10)]
    board[1][1] = {"kind": "PASTURE"}
    obs = make_obs(farmer=(0, 0), hands=((1, 1),), tiles=board,
                   inventories=[{}, {"SHEEP": 1}])
    result = generate_tasks(
        obs, 0, feasible_plan=make_plan(animals={"SHEEP": 1}),
        remaining_sells={})
    dispatched = run_foreman(obs, 0, result.sorted_tasks())

    assert dispatched.hands_actions == (("PLACE", "SHEEP", 1),)
    assert dispatched.assignments[1].reason == "underfoot_execution"
    assert dispatched.assignments[1].action[0] != "PICKUP"


def test_shed_owned_animal_pickup_then_regenerated_place():
    board = [[None] * 10 for _ in range(10)]
    board[1][1] = {"kind": "PASTURE"}
    obs = make_obs(farmer=(4, 4), tiles=board, shed={"COW": 1})
    expected = make_plan(animals={"COW": 1})
    generated = generate_tasks(obs, 0, feasible_plan=expected,
                                remaining_sells={})

    pickup = run_foreman(obs, 0, generated.sorted_tasks())
    assert pickup.farmer_action == ("PICKUP", "COW", 1)

    carried = copy.deepcopy(obs)
    carried["private"]["shed"]["COW"] = 0
    carried["private"]["inventories"][0]["COW"] = 1
    placed = run_foreman(carried, 0, generated.sorted_tasks())
    assert placed.farmer_action == ("NORTH",)
    assert placed.assignments[0].task_key == "PLACE:COW:1,1"


def test_hard_water_beats_underfoot_place():
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {"kind": "PASTURE"}
    board[4][5] = hard_water_tile()
    obs = make_obs(tiles=board, inventories=[{"COW": 1}])
    result = generate_tasks(
        obs, 0, feasible_plan=make_plan(animals={"COW": 1}),
        remaining_sells={})

    dispatched = run_foreman(obs, 0, result.sorted_tasks())
    assert dispatched.assignments[0].task_key == "WATER:4,5"
    assert dispatched.farmer_action == ("EAST",)


def test_starvation_feed_preempts_underfoot_place():
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {"kind": "PASTURE"}
    board[4][5] = animal_tile("GOOSE", starving=True)
    agent = ExecutorAgent(
        FixedPlanProvider(make_plan(animals={"GOOSE": 1, "COW": 1})),
        seat=0, config=AgentConfig(strict=True))

    action = agent(make_obs(tiles=board, inventories=[{"COW": 1, "WHEAT": 1}]))
    assert action["farmer"] == ["EAST"]
    assert "PLACE" not in action["farmer"]


def test_unplaced_owned_livestock_does_not_duplicate_buy_order():
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {"kind": "PASTURE"}
    result = generate_tasks(
        make_obs(tiles=board, shed={"COW": 1}), 0,
        feasible_plan=make_plan(animals={"COW": 1}), remaining_sells={})

    assert sum(task.kind == "PLACE" for task in result.tasks) == 1
    assert not any(task.kind == "BUY_ANIMAL" for task in result.tasks)


def test_official_and_fast_private_inventory_shapes_generate_same_place_task():
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {"kind": "PASTURE"}
    official = make_obs(tiles=copy.deepcopy(board), shed={"COW": 0},
                        inventories=[{"COW": 1}])
    fast = make_obs(tiles=copy.deepcopy(board), shed={"COW": 0},
                    inventories=[{name: int(name == "COW")
                                  for name in (*PRODUCTS, *ANIMALS)}])

    official = _executor_observation(official, from_fast=False)
    fast = _executor_observation(fast, from_fast=True)
    expected = make_plan(animals={"COW": 1})
    official_tasks = generate_tasks(official, 0, feasible_plan=expected,
                                    remaining_sells={})
    fast_tasks = generate_tasks(fast, 0, feasible_plan=expected,
                                remaining_sells={})
    assert [(task.key, task.depends_on) for task in official_tasks.tasks
            if task.kind == "PLACE"] == [
                ("PLACE:COW:4,4", ())]
    assert [(task.key, task.depends_on) for task in fast_tasks.tasks
            if task.kind == "PLACE"] == [
                ("PLACE:COW:4,4", ())]


def test_prior_debt_does_not_strand_owned_animal_in_ready_structure():
    plan = make_plan(animals={"COW": 1})
    agent = ExecutorAgent(FixedPlanProvider(plan), seat=0,
                          config=AgentConfig(strict=True, turn_trace=True))
    debt_board = [[None] * 10 for _ in range(10)]
    debt_board[1][1] = hard_water_tile()
    agent(make_obs(day=3, hour=23, farmer=(0, 0), tiles=debt_board))

    ready_board = [[None] * 10 for _ in range(10)]
    ready_board[4][4] = {"kind": "PASTURE"}
    action = agent(make_obs(day=4, hour=0, tiles=ready_board,
                            inventories=[{"COW": 1}]))

    trace = agent.debug_trace_turn
    generated_place = next(task for task in trace["tasks"]
                           if task["key"] == "PLACE:COW:4,4")
    assert generated_place["depends_on"] == []
    assert trace["survival"]["expansion_suppressed"] is True
    assert action["farmer"] == ["PLACE", "COW", 1]
