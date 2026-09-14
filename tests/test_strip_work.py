"""Self-contained contract tests for the pure strip work forecast."""

import copy
import json

from executor_v0.plan import DailyPlan
from executor_v0.strip_work import (
    BlockReason,
    StripWorkConfig,
    WorkStatus,
    build_strip_work_plan,
    row_key_for_tile,
)


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")


def plan(**changes):
    value = {
        "crop_targets": {c: 0 for c in CROPS},
        "animal_targets": {a: 0 for a in ANIMALS},
        "land_count": 1,
        "fertilizer_by_crop": {c: 0 for c in CROPS},
        "care_by_animal": {a: 0 for a in ANIMALS},
        "sell_quantities": {p: {h: 0 for h in (0, 4, 8, 12, 16, 20)} for p in PRODUCTS},
    }
    for key, update in changes.items():
        if key == "sell_quantities":
            value[key] = {p: {**value[key][p], **update.get(p, {})} for p in PRODUCTS}
        else:
            value[key] = (
                {**value[key], **update} if isinstance(value[key], dict) else update
            )
    return DailyPlan.create(**value)


def plant(
    crop="WHEAT",
    planted_day=0,
    *,
    yield_units=0,
    watered_today=False,
    fertilized_until_day=-1,
):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "yield_units": yield_units,
        "watered_today": watered_today,
        "fertilized_until_day": fertilized_until_day,
        "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }


def animal(name="GOOSE", *, fed_today=False, cared_today=False):
    return {
        "kind": "COOP" if name == "GOOSE" else "PASTURE",
        "animal": name,
        "placed_day": 0,
        "yield_units": 0,
        "fed_today": fed_today,
        "cared_today": cared_today,
        "consecutive_unfed": 0,
    }


def obs(
    board=None,
    *,
    day=3,
    player=0,
    unlocked=("NW",),
    shed=None,
    seeds=None,
    inventories=None,
    money=5000.0,
    step=None,
):
    board = board or [[None] * 10 for _ in range(10)]
    farm = {
        "tiles": board,
        "unlocked_quadrants": list(unlocked),
        "farmer": [9, 9],
        "hands": [],
        "money": money,
    }
    return {
        "day": day,
        "hour": 0,
        "step": day * 24 if step is None else step,
        "player": player,
        "farms": [farm, copy.deepcopy(farm)],
        "private": {
            "shed": shed or {},
            "seeds": seeds or {},
            "inventories": inventories if inventories is not None else [{}, {}],
        },
        "market": {"inventory": {}, "prices": {}},
        "town": {},
    }


def kinds(result, kind):
    return [item for item in result.items if item.kind == kind]


def test_empty_crop_is_plant_then_water_and_seeds_are_global():
    result = build_strip_work_plan(
        obs(), plan(crop_targets={"WHEAT": 1}), config=StripWorkConfig()
    )
    chain = next(c for c in result.chains if c.kind == "CROP_GROWTH")
    kinds_in_chain = [
        next(x for x in result.items if x.id == i).kind for i in chain.item_ids
    ]
    assert kinds_in_chain == ["PLANT", "WATER"]
    planting = next(x for x in result.items if x.kind == "PLANT")
    assert planting.required_supply_dict == {"WHEAT": 1}
    assert not any(x.kind == "PICKUP" for x in result.items)


def test_replacement_has_harvest_plant_water_but_reduction_has_no_replacement():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=0, yield_units=3)
    for y in range(5):
        for x in range(5):
            if board[y][x] is None:
                board[y][x] = "LOCKED"
    replacement = build_strip_work_plan(
        obs(board, day=3, seeds={"TOMATO": 1}), plan(crop_targets={"TOMATO": 1})
    )
    chain = next(c for c in replacement.chains if c.kind == "CROP_GROWTH")
    assert [
        next(x for x in replacement.items if x.id == i).kind for i in chain.item_ids
    ] == ["HARVEST", "PLANT", "WATER"]
    board[0][1] = plant("WHEAT", planted_day=0, yield_units=1)
    reduced = build_strip_work_plan(obs(board, day=3), plan(crop_targets={"WHEAT": 1}))
    assert not kinds(reduced, "PLANT")
    assert [(item.kind, item.tile) for item in kinds(reduced, "DIG")] == [
        ("DIG", (0, 1))
    ]
    assert not [item for item in kinds(reduced, "WATER") if item.tile == (0, 1)]


def test_animal_build_and_place_are_retained_when_purchase_is_missing():
    result = build_strip_work_plan(obs(), plan(animal_targets={"GOOSE": 1}))
    assert {x.kind for x in result.items} == {"BUILD_COOP", "BUY_ANIMAL", "PLACE"}
    purchase = next(x for x in result.items if x.kind == "BUY_ANIMAL")
    assert purchase.animal == "GOOSE" and purchase.status == WorkStatus.READY
    place = kinds(result, "PLACE")[0]
    assert place.status == WorkStatus.BLOCKED
    assert place.block_reason == BlockReason.MISSING_PURCHASE
    assert purchase.id in place.depends_on


def test_land_request_is_explicit_locked_land_without_coordinates():
    result = build_strip_work_plan(
        obs(),
        plan(crop_targets={"WHEAT": 101}, land_count=2),
    )
    land = kinds(result, "BUY_LAND")
    assert len(land) == 1 and land[0].tile is None
    assert land[0].land == "NE"
    assert land[0].block_reason == BlockReason.LOCKED_LAND
    unresolved = next(x for x in result.items if x.id == "UNRESOLVED_PLANT:WHEAT")
    assert unresolved.block_reason == BlockReason.LOCKED_LAND
    assert land[0].id in unresolved.depends_on
    assert result.diagnostics.unresolved_land_delta == 1


def test_feed_then_care_separates_missing_feed_from_care_dependency():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = animal("GOOSE")
    result = build_strip_work_plan(obs(board), plan(care_by_animal={"GOOSE": 1}))
    feed, care = kinds(result, "FEED")[0], kinds(result, "CARE")[0]
    assert feed.block_reason == BlockReason.MISSING_SUPPLY
    assert care.block_reason == BlockReason.DEPENDENCY_BLOCKED
    ready = build_strip_work_plan(
        obs(board, shed={"WHEAT": 1}), plan(care_by_animal={"GOOSE": 1})
    )
    assert all(
        x.status == WorkStatus.READY
        for x in kinds(ready, "FEED") + kinds(ready, "CARE")
    )
    late = build_strip_work_plan(
        obs(board, day=29, shed={"WHEAT": 1}),
        plan(care_by_animal={"GOOSE": 1}),
    )
    assert not kinds(late, "CARE")


def test_fertilizer_policy_is_limited_to_wheat_and_strawberry_and_does_not_duplicate():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=1)
    board[0][1] = plant("STRAWBERRY", planted_day=-6)
    board[0][2] = plant("CARROT", planted_day=1)
    result = build_strip_work_plan(
        obs(board, day=3, shed={"FERTILIZER": 2}),
        plan(fertilizer_by_crop={"WHEAT": 1, "STRAWBERRY": 1, "CARROT": 1}),
    )
    assert {(x.crop, x.tile) for x in kinds(result, "FERTILIZE")} == {
        ("WHEAT", (0, 0)),
        ("STRAWBERRY", (0, 1)),
    }
    for crop, tile in (("WHEAT", (0, 0)), ("STRAWBERRY", (0, 1))):
        water = [x for x in kinds(result, "WATER") if x.tile == tile]
        assert len(water) == 1
        assert water[0].source == "fertilizer_linked_productive"
    active = copy.deepcopy(board)
    active[0][0]["fertilized_until_day"] = 9
    again = build_strip_work_plan(
        obs(active, day=3, shed={"FERTILIZER": 2}),
        plan(fertilizer_by_crop={"WHEAT": 1}),
    )
    assert not kinds(again, "FERTILIZE")


def test_routine_watering_uses_default_ages_and_stable_reasons():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=15)  # age 0
    board[0][1] = plant("CARROT", planted_day=13)  # age 2
    board[0][2] = plant("MELON", planted_day=11)  # age 4
    board[0][3] = plant("STRAWBERRY", planted_day=13)  # age 2
    board[0][4] = plant("TOMATO", planted_day=7)  # age 8, later productive
    board[1][0] = plant("MELON", planted_day=4)  # age 11, retained
    board[1][1] = plant("MELON", planted_day=4, yield_units=6)  # full, skip
    board[1][2] = plant("CARROT", planted_day=14)  # age 1, survival
    board[1][2]["consecutive_unwatered"] = 1
    result = build_strip_work_plan(
        obs(board, day=15),
        plan(
            crop_targets={
                "WHEAT": 1,
                "CARROT": 2,
                "MELON": 3,
                "STRAWBERRY": 1,
                "TOMATO": 1,
            }
        ),
    )
    water = {(x.tile, x.source) for x in kinds(result, "WATER")}
    assert ((0, 0), "planting_continuation") in water
    assert ((0, 1), "yield_improving") in water
    assert ((0, 2), "optional_deferrable") in water
    assert ((0, 3), "optional_deferrable") in water
    assert ((0, 4), "yield_improving") in water
    assert ((1, 0), "yield_improving") in water
    assert ((1, 1), "optional_deferrable") not in water
    assert ((1, 2), "survival_weed_prevention") in water


def test_wheat_harvest_uses_authoritative_terminal_exception():
    board = [["LOCKED"] * 5 + [None] * 5 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=0, yield_units=0)
    for y in range(5):
        for x in range(5):
            if (y, x) != (0, 0):
                board[y][x] = "LOCKED"
    result = build_strip_work_plan(
        obs(board, day=29, step=718, seeds={"TOMATO": 1}),
        plan(crop_targets={"TOMATO": 1}),
    )
    assert kinds(result, "HARVEST")[0].crop == "WHEAT"


def test_sell_includes_carried_delivery_and_retains_shortage_sell():
    result = build_strip_work_plan(
        obs(shed={"WHEAT": 1}, inventories=[{"WHEAT": 2}, {}]),
        plan(sell_quantities={"WHEAT": {0: 3}}),
    )
    assert kinds(result, "DELIVERY")[0].quantity == 2
    assert kinds(result, "SELL")[0].status == WorkStatus.READY
    short = build_strip_work_plan(
        obs(shed={}, inventories=[{}, {}]), plan(sell_quantities={"WHEAT": {0: 2}})
    )
    assert kinds(short, "SELL")[0].block_reason == BlockReason.MISSING_SUPPLY


def test_animal_purchase_is_blocked_by_authoritative_farm_money():
    result = build_strip_work_plan(obs(money=0), plan(animal_targets={"GOOSE": 1}))
    purchase = kinds(result, "BUY_ANIMAL")[0]
    place = kinds(result, "PLACE")[0]
    assert purchase.block_reason == BlockReason.MISSING_GLOBAL_RESOURCE
    assert place.block_reason == BlockReason.DEPENDENCY_BLOCKED


def test_coordinates_purity_determinism_and_json_safety():
    board = [[None] * 10 for _ in range(10)]
    board[6][3] = plant("WHEAT", planted_day=5)
    for y in range(5):
        for x in range(5):
            board[y][x] = "LOCKED"
    board[6][6] = None
    for y in range(5, 10):
        for x in range(5):
            if (y, x) != (6, 4):
                board[y][x] = "LOCKED"
    before = copy.deepcopy(board)
    observation = obs(board, unlocked=("NW", "SW"), seeds={"WHEAT": 1})
    first = build_strip_work_plan(
        observation, plan(crop_targets={"WHEAT": 2}), acting_seat=0
    )
    second = build_strip_work_plan(
        copy.deepcopy(observation), plan(crop_targets={"WHEAT": 2}), acting_seat=0
    )
    assert first == second and board == before
    assert next(x for x in first.items if x.kind == "PLANT").tile == (6, 4)
    assert row_key_for_tile((6, 7)).quadrant == "SE"
    json.dumps(first.to_json_dict())
