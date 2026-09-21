"""Self-contained contract tests for the pure strip work forecast."""

import copy
import json

import pytest

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


def test_new_crop_water_waits_for_plant_but_chain_counts_both():
    result = build_strip_work_plan(
        obs(seeds={"WHEAT": 1}),
        plan(crop_targets={"WHEAT": 1}),
        config=StripWorkConfig(),
    )
    planting = next(x for x in result.items if x.kind == "PLANT")
    watering = next(x for x in result.items if x.kind == "WATER")
    assert planting.status == WorkStatus.READY
    assert watering.status == WorkStatus.BLOCKED
    assert watering.block_reason == BlockReason.DEPENDENCY_BLOCKED
    assert watering.depends_on == (planting.id,)
    chain = next(c for c in result.chains if c.kind == "CROP_GROWTH")
    assert chain.interaction_turns == 2


def test_retained_crop_preferred_slot_is_reserved_without_duplicate_plant():
    result = build_strip_work_plan(
        obs(seeds={"WHEAT": 1, "CARROT": 1}),
        plan(crop_targets={"WHEAT": 1, "CARROT": 1}),
        preferred_crop_slots={"WHEAT": ((0, 0),), "CARROT": ((0, 0),)},
    )
    plants = [item for item in result.items if item.kind == "PLANT"]
    assert {item.crop: item.tile for item in plants} == {
        "WHEAT": (0, 0),
        "CARROT": (4, 4),
    }
    assert len({item.tile for item in plants}) == len(plants)
    assert next(item for item in plants if item.crop == "WHEAT").source == (
        "retained_crop_maintenance"
    )


def test_preferred_tomato_slot_does_not_get_retained_seed_retry_provenance():
    result = build_strip_work_plan(
        obs(seeds={"TOMATO": 1}),
        plan(crop_targets={"TOMATO": 1}),
        preferred_crop_slots={"TOMATO": ((0, 0),)},
    )
    planting = next(item for item in result.items if item.kind == "PLANT")
    assert planting.tile == (0, 0)
    assert planting.source == "crop_reconciliation"


def test_seed_observation_unlocks_only_the_affordable_crop_stage():
    initial = build_strip_work_plan(
        obs(money=16),
        plan(crop_targets={"WHEAT": 1, "STRAWBERRY": 17}),
    )
    initial_plants = [item for item in initial.items if item.kind == "PLANT"]
    assert len([item for item in initial_plants if item.crop == "WHEAT"]) == 1
    assert len([item for item in initial_plants if item.crop == "STRAWBERRY"]) == 17
    assert all(
        item.status == WorkStatus.BLOCKED
        and item.block_reason == BlockReason.MISSING_GLOBAL_RESOURCE
        for item in initial_plants
    )

    refreshed = build_strip_work_plan(
        obs(money=6, seeds={"WHEAT": 1}),
        plan(crop_targets={"WHEAT": 1, "STRAWBERRY": 17}),
    )
    wheat = next(
        item for item in refreshed.items if item.kind == "PLANT" and item.crop == "WHEAT"
    )
    strawberries = [
        item
        for item in refreshed.items
        if item.kind == "PLANT" and item.crop == "STRAWBERRY"
    ]
    assert wheat.status == WorkStatus.READY
    assert all(item.block_reason == BlockReason.MISSING_GLOBAL_RESOURCE for item in strawberries)
    assert refreshed.diagnostics.represented_crop_delta_dict["STRAWBERRY"] == 17


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
    by_id = {x.id: x for x in replacement.items}
    harvest = by_id[chain.item_ids[0]]
    plant_step = by_id[chain.item_ids[1]]
    water_step = by_id[chain.item_ids[2]]
    assert harvest.status == WorkStatus.READY
    assert plant_step.status == WorkStatus.BLOCKED
    assert plant_step.block_reason == BlockReason.DEPENDENCY_BLOCKED
    assert plant_step.depends_on == (harvest.id,)
    assert water_step.status == WorkStatus.BLOCKED
    assert water_step.block_reason == BlockReason.DEPENDENCY_BLOCKED
    assert water_step.depends_on == (plant_step.id,)
    assert chain.interaction_turns == 3
    board[0][1] = plant("WHEAT", planted_day=0, yield_units=1)
    reduced = build_strip_work_plan(obs(board, day=3), plan(crop_targets={"WHEAT": 1}))
    assert not kinds(reduced, "PLANT")
    assert [(item.kind, item.tile) for item in kinds(reduced, "DIG")] == [
        ("DIG", (0, 1))
    ]
    assert not [item for item in kinds(reduced, "WATER") if item.tile == (0, 1)]


@pytest.mark.parametrize(
    ("crop", "day"),
    [("CARROT", 3), ("TOMATO", 8), ("STRAWBERRY", 10), ("MELON", 10)],
)
def test_retained_mature_non_wheat_crop_gets_one_routine_harvest(crop, day):
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant(crop, planted_day=0, yield_units=1)

    result = build_strip_work_plan(
        obs(board, day=day), plan(crop_targets={crop: 1})
    )

    harvests = kinds(result, "HARVEST")
    assert [(item.id, item.tile, item.crop, item.source) for item in harvests] == [
        ("HARVEST:0,0", (0, 0), crop, "routine_harvest")
    ]


def test_retained_immature_crop_does_not_get_routine_harvest():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("MELON", planted_day=2, yield_units=1)

    result = build_strip_work_plan(
        obs(board, day=3), plan(crop_targets={"MELON": 1})
    )

    assert not kinds(result, "HARVEST")


def test_retained_wheat_uses_threshold_and_horizon_harvestability():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=0, yield_units=2)
    below_threshold = build_strip_work_plan(
        obs(board, day=3, step=72), plan(crop_targets={"WHEAT": 1})
    )
    assert not kinds(below_threshold, "HARVEST")

    board[0][0]["yield_units"] = 3
    at_threshold = build_strip_work_plan(
        obs(board, day=3, step=72), plan(crop_targets={"WHEAT": 1})
    )
    assert [item.source for item in kinds(at_threshold, "HARVEST")] == [
        "routine_harvest"
    ]

    board[0][0]["yield_units"] = 0
    terminal = build_strip_work_plan(
        obs(board, day=29, step=718), plan(crop_targets={"WHEAT": 1})
    )
    assert [item.source for item in kinds(terminal, "HARVEST")] == [
        "routine_harvest"
    ]


def test_replacement_and_reduction_harvests_are_not_duplicated_by_routine_scan():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=0, yield_units=3)
    for y in range(5):
        for x in range(5):
            if (y, x) != (0, 0):
                board[y][x] = "LOCKED"
    replacement = build_strip_work_plan(
        obs(board, day=3, step=72, seeds={"TOMATO": 1}),
        plan(crop_targets={"TOMATO": 1}),
    )
    replacement_harvests = kinds(replacement, "HARVEST")
    assert len(replacement_harvests) == 1
    assert replacement_harvests[0].source == "crop_replacement"

    reduction = build_strip_work_plan(
        obs(board, day=3, step=72), plan(crop_targets={"WHEAT": 0})
    )
    reduction_harvests = kinds(reduction, "HARVEST")
    assert len(reduction_harvests) == 1
    assert reduction_harvests[0].source == "crop_reduction"


def test_routine_and_replacement_harvests_share_stable_unique_ids():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=0, yield_units=3)
    board[0][1] = plant("MELON", planted_day=0, yield_units=1)
    for y in range(5):
        for x in range(5):
            if (y, x) not in {(0, 0), (0, 1)}:
                board[y][x] = "LOCKED"
    result = build_strip_work_plan(
        obs(board, day=10, step=240, seeds={"TOMATO": 1}),
        plan(crop_targets={"TOMATO": 1, "MELON": 1}),
    )

    harvests = {item.tile: item for item in kinds(result, "HARVEST")}
    assert harvests[(0, 0)].source == "crop_replacement"
    assert harvests[(0, 1)].source == "routine_harvest"
    assert len({item.id for item in harvests.values()}) == len(harvests)


def test_retained_harvest_output_is_deterministic():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("MELON", planted_day=0, yield_units=1)
    observation = obs(board, day=10, step=240)
    daily_plan = plan(crop_targets={"MELON": 1})

    first = build_strip_work_plan(observation, daily_plan)
    second = build_strip_work_plan(copy.deepcopy(observation), daily_plan)
    assert first == second


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
    assert land[0].status == WorkStatus.READY
    assert land[0].block_reason is None
    unresolved = next(x for x in result.items if x.id == "UNRESOLVED_PLANT:WHEAT")
    assert unresolved.block_reason == BlockReason.LOCKED_LAND
    assert land[0].id in unresolved.depends_on
    assert result.diagnostics.unresolved_land_delta == 1


def test_land_purchase_is_blocked_without_money_but_demand_remains():
    result = build_strip_work_plan(
        obs(money=0.0),
        plan(crop_targets={"WHEAT": 101}, land_count=2),
    )
    land = kinds(result, "BUY_LAND")
    assert len(land) == 1 and land[0].land == "NE"
    assert land[0].status == WorkStatus.BLOCKED
    assert land[0].block_reason == BlockReason.MISSING_GLOBAL_RESOURCE
    unresolved = next(x for x in result.items if x.id == "UNRESOLVED_PLANT:WHEAT")
    assert unresolved.block_reason == BlockReason.LOCKED_LAND
    assert land[0].id in unresolved.depends_on


def test_multi_land_purchases_are_ordered_not_independent():
    result = build_strip_work_plan(obs(), plan(land_count=3))
    land = sorted(kinds(result, "BUY_LAND"), key=lambda x: x.id)
    assert [x.land for x in land] == ["NE", "SW"]
    first, second = land
    assert first.status == WorkStatus.READY
    assert first.depends_on == ()
    assert second.status == WorkStatus.BLOCKED
    assert first.id in second.depends_on


def test_feed_then_care_separates_missing_feed_from_care_dependency():
    # Strip CARE is executor-owned: zero care_by_animal counts still forecast.
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = animal("GOOSE")
    result = build_strip_work_plan(obs(board), plan())
    feed, care = kinds(result, "FEED")[0], kinds(result, "CARE")[0]
    assert feed.block_reason == BlockReason.MISSING_SUPPLY
    assert care.block_reason == BlockReason.DEPENDENCY_BLOCKED
    assert care.depends_on == (feed.id,)
    ready = build_strip_work_plan(obs(board, shed={"WHEAT": 1}), plan())
    feed, care = kinds(ready, "FEED")[0], kinds(ready, "CARE")[0]
    assert feed.status == WorkStatus.READY
    assert care.status == WorkStatus.BLOCKED
    assert care.block_reason == BlockReason.DEPENDENCY_BLOCKED
    assert care.depends_on == (feed.id,)
    chain = next(c for c in ready.chains if c.kind == "ANIMAL_CARE")
    assert chain.interaction_turns == 2
    late = build_strip_work_plan(obs(board, day=29, shed={"WHEAT": 1}), plan())
    assert not kinds(late, "CARE")


def test_fed_animal_gets_standalone_ready_care_and_cared_gets_none():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = animal("GOOSE", fed_today=True)
    fed = build_strip_work_plan(obs(board), plan())
    assert not kinds(fed, "FEED")
    care = kinds(fed, "CARE")[0]
    assert care.status == WorkStatus.READY
    cared_board = [[None] * 10 for _ in range(10)]
    cared_board[0][0] = animal("GOOSE", fed_today=True, cared_today=True)
    cared = build_strip_work_plan(obs(cared_board), plan())
    assert not kinds(cared, "CARE")
    assert not kinds(cared, "FEED")


def test_fertilizer_policy_is_limited_to_wheat_and_strawberry_and_does_not_duplicate():
    # Strip fertilizer is executor-owned: zero fertilizer_by_crop counts still
    # forecast under ON permissions.
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=1)
    board[0][1] = plant("STRAWBERRY", planted_day=-6)
    board[0][2] = plant("CARROT", planted_day=1)
    result = build_strip_work_plan(
        obs(board, day=3, shed={"FERTILIZER": 2}),
        plan(),
    )
    assert {(x.crop, x.tile) for x in kinds(result, "FERTILIZE")} == {
        ("WHEAT", (0, 0)),
        ("STRAWBERRY", (0, 1)),
    }
    for crop, tile in (("WHEAT", (0, 0)), ("STRAWBERRY", (0, 1))):
        treatment = next(
            x for x in kinds(result, "FERTILIZE") if x.tile == tile
        )
        assert treatment.status == WorkStatus.READY
        water = [x for x in kinds(result, "WATER") if x.tile == tile]
        assert len(water) == 1
        assert water[0].source == "fertilizer_linked_productive"
        assert water[0].status == WorkStatus.BLOCKED
        assert water[0].block_reason == BlockReason.DEPENDENCY_BLOCKED
        assert water[0].depends_on == (treatment.id,)
    active = copy.deepcopy(board)
    active[0][0]["fertilized_until_day"] = 9
    again = build_strip_work_plan(
        obs(active, day=3, shed={"FERTILIZER": 2}),
        plan(),
    )
    # Active wheat treatment suppresses wheat only; strawberry remains.
    assert {(x.crop, x.tile) for x in kinds(again, "FERTILIZE")} == {
        ("STRAWBERRY", (0, 1))
    }


def test_fertilizer_permission_off_suppresses_only_that_crop():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=1)
    board[0][1] = plant("STRAWBERRY", planted_day=-6)
    no_wheat = build_strip_work_plan(
        obs(board, day=3, shed={"FERTILIZER": 2}),
        plan(),
        config=StripWorkConfig(allow_wheat_fertilizer=False),
    )
    assert {(x.crop, x.tile) for x in kinds(no_wheat, "FERTILIZE")} == {
        ("STRAWBERRY", (0, 1))
    }
    no_berry = build_strip_work_plan(
        obs(board, day=3, shed={"FERTILIZER": 2}),
        plan(),
        config=StripWorkConfig(allow_strawberry_fertilizer=False),
    )
    assert {(x.crop, x.tile) for x in kinds(no_berry, "FERTILIZE")} == {
        ("WHEAT", (0, 0))
    }


def test_scarce_fertilizer_represents_all_demand_with_stable_order():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("STRAWBERRY", planted_day=-6)
    board[0][1] = plant("STRAWBERRY", planted_day=-6)
    board[0][2] = plant("STRAWBERRY", planted_day=-6)
    result = build_strip_work_plan(
        obs(board, day=3, shed={"FERTILIZER": 1}),
        plan(),
    )
    treatments = sorted(kinds(result, "FERTILIZE"), key=lambda x: x.tile)
    assert [x.tile for x in treatments] == [(0, 0), (0, 1), (0, 2)]
    assert treatments[0].status == WorkStatus.READY
    assert treatments[1].block_reason == BlockReason.MISSING_SUPPLY
    assert treatments[2].block_reason == BlockReason.MISSING_SUPPLY
    assert result.diagnostics.supply_demand is not None
    demand = {
        (d.item, d.scope): d
        for d in result.diagnostics.supply_demand
    }
    assert demand[("FERTILIZER", "inventory")].requested == 3
    assert demand[("FERTILIZER", "inventory")].available == 1
    assert demand[("FERTILIZER", "inventory")].missing == 2
    again = build_strip_work_plan(
        obs(copy.deepcopy(board), day=3, shed={"FERTILIZER": 1}),
        plan(),
    )
    assert [x.id for x in again.items] == [x.id for x in result.items]


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


def test_wheat_threshold_harvest_suppresses_same_turn_routine_water():
    for age in (2, 3, 4):
        board = [[None] * 10 for _ in range(10)]
        board[0][0] = plant("WHEAT", planted_day=15 - age, yield_units=3)
        result = build_strip_work_plan(
            obs(board, day=15, step=360), plan(crop_targets={"WHEAT": 1})
        )
        assert [item.kind for item in kinds(result, "HARVEST") if item.tile == (0, 0)] == [
            "HARVEST"
        ]
        assert not [item for item in kinds(result, "WATER") if item.tile == (0, 0)]


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
    delivery = kinds(result, "DELIVERY")[0]
    assert delivery.quantity == 2
    assert delivery.status == WorkStatus.READY
    sell = kinds(result, "SELL")[0]
    assert sell.status == WorkStatus.BLOCKED
    assert sell.block_reason == BlockReason.DEPENDENCY_BLOCKED
    assert sell.depends_on == (delivery.id,)
    chain = next(c for c in result.chains if c.kind == "SELL_INTENT")
    assert chain.interaction_turns == 2
    short = build_strip_work_plan(
        obs(shed={}, inventories=[{}, {}]), plan(sell_quantities={"WHEAT": {0: 2}})
    )
    assert kinds(short, "SELL")[0].block_reason == BlockReason.MISSING_SUPPLY


def test_blocked_chain_workload_still_counts_complete_forecast():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=0, yield_units=3)
    for y in range(5):
        for x in range(5):
            if board[y][x] is None:
                board[y][x] = "LOCKED"
    result = build_strip_work_plan(
        obs(board, day=3, seeds={"TOMATO": 1}), plan(crop_targets={"TOMATO": 1})
    )
    chain = next(c for c in result.chains if c.kind == "CROP_GROWTH")
    assert chain.status == WorkStatus.BLOCKED
    assert chain.interaction_turns == 3
    row = result.row_summary_by_key[row_key_for_tile((0, 0))]
    assert row.ready_interactions == 1
    assert row.future_interactions == 2
    assert row.nontravel_turns == 3


def test_animal_purchase_is_blocked_by_authoritative_farm_money():
    result = build_strip_work_plan(obs(money=0), plan(animal_targets={"GOOSE": 1}))
    purchase = kinds(result, "BUY_ANIMAL")[0]
    place = kinds(result, "PLACE")[0]
    assert purchase.block_reason == BlockReason.MISSING_GLOBAL_RESOURCE
    assert place.block_reason == BlockReason.DEPENDENCY_BLOCKED


def test_escaped_animal_deficit_reuses_empty_matching_structure():
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {"kind": "PASTURE"}
    result = build_strip_work_plan(
        obs(board, shed={}, money=1000), plan(animal_targets={"SHEEP": 1})
    )
    assert {item.kind for item in result.items} == {"BUY_ANIMAL", "PLACE"}
    purchase = kinds(result, "BUY_ANIMAL")[0]
    place = kinds(result, "PLACE")[0]
    assert purchase.status == WorkStatus.READY
    assert place.status == WorkStatus.BLOCKED
    assert place.block_reason == BlockReason.MISSING_PURCHASE
    assert place.tile == (4, 4)


def test_observed_purchased_animal_makes_place_ready_without_duplicate_buy():
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {"kind": "PASTURE"}
    result = build_strip_work_plan(
        obs(board, shed={"SHEEP": 1}, money=1000),
        plan(animal_targets={"SHEEP": 1}),
    )
    assert not kinds(result, "BUY_ANIMAL")
    place = kinds(result, "PLACE")[0]
    assert place.status == WorkStatus.READY
    assert place.required_supply_dict == {"SHEEP": 1}


def test_animal_target_counts_are_authoritative_and_structure_type_is_exact():
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {
        "kind": "PASTURE", "animal": "SHEEP", "placed_day": 1,
        "yield_units": 0, "fed_today": False, "cared_today": False,
        "consecutive_unfed": 0,
    }
    retained = build_strip_work_plan(
        obs(board, shed={"SHEEP": 1}, money=1000),
        plan(animal_targets={"SHEEP": 1}),
    )
    assert not kinds(retained, "PLACE")
    assert not kinds(retained, "BUY_ANIMAL")

    wrong_structure = [[None] * 10 for _ in range(10)]
    wrong_structure[4][4] = {"kind": "COOP"}
    rebuilt = build_strip_work_plan(
        obs(wrong_structure, shed={"SHEEP": 1}, money=1000),
        plan(animal_targets={"SHEEP": 1}),
    )
    assert [item.kind for item in rebuilt.items] == ["BUILD_PASTURE", "PLACE"]


def test_multiple_animal_deficits_use_canonical_deterministic_order():
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {"kind": "PASTURE"}
    board[3][4] = {"kind": "PASTURE"}
    result = build_strip_work_plan(
        obs(board, shed={}, money=5000),
        plan(animal_targets={"COW": 1, "SHEEP": 1}),
    )
    placements = [item for item in result.items if item.kind == "PLACE"]
    assert [(item.animal, item.tile) for item in placements] == [
        ("COW", (4, 4)), ("SHEEP", (3, 4))
    ]


def test_zero_animal_target_stops_replacement_attempts():
    board = [[None] * 10 for _ in range(10)]
    board[4][4] = {"kind": "PASTURE"}
    result = build_strip_work_plan(obs(board, money=1000), plan())
    assert not any(item.kind in {"BUY_ANIMAL", "PLACE", "BUILD_PASTURE"}
                   for item in result.items)


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
